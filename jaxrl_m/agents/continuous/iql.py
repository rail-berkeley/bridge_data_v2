import copy
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch, PRNGKey
from jaxrl_m.common.common import ModuleDict, JaxRLTrainState, nonpytree_field
from jaxrl_m.common.encoding import EncodingWrapper
from jaxrl_m.networks.actor_critic_nets import ValueCritic
from jaxrl_m.networks.actor_critic_nets import Policy
from jaxrl_m.networks.actor_critic_nets import Critic
from jaxrl_m.networks.mlp import MLP
import numpy as np

import flax
import flax.linen as nn


def expectile_loss(diff, expectile=0.5):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def iql_value_loss(q, v, expectile):
    value_loss = expectile_loss(q - v, expectile)
    return value_loss.mean(), {
        "value_loss": value_loss.mean(),
        "uncentered_loss": jnp.mean((q - v) ** 2),
        "v": v.mean(),
    }


def iql_critic_loss(q, q_target):
    critic_loss = jnp.square(q - q_target)
    return critic_loss.mean(), {
        "td_loss": critic_loss.mean(),
        "q": q.mean(),
    }


def iql_actor_loss(q, v, dist, actions, temperature=1.0, adv_clip_max=100.0, mask=None):
    adv = q - v

    exp_adv = jnp.exp(adv / temperature)
    exp_adv = jnp.minimum(exp_adv, adv_clip_max)

    log_probs = dist.log_prob(actions)
    actor_loss = -(exp_adv * log_probs)

    if mask is not None:
        actor_loss *= mask
        actor_loss = jnp.sum(actor_loss) / jnp.sum(mask)
    else:
        actor_loss = jnp.mean(actor_loss)

    behavior_mse = jnp.square(dist.mode() - actions).sum(-1)

    return actor_loss, {
        "actor_loss": actor_loss,
        "behavior_logprob": log_probs.mean(),
        "behavior_mse": behavior_mse.mean(),
        "adv_mean": adv.mean(),
        "adv_max": adv.max(),
        "adv_min": adv.min(),
    }


class IQLAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        new_rng, dropout_rng = jax.random.split(self.state.rng)

        def critic_loss_fn(params):
            next_v = self.state.apply_fn(
                {"params": self.state.target_params},
                batch["next_observations"],
                name="value",
            )
            target_q = (
                batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
            )
            q = self.state.apply_fn(
                {"params": params},
                batch["observations"],
                batch["actions"],
                name="critic",
            )
            return iql_critic_loss(q, target_q)

        def value_loss_fn(params):
            q = self.state.apply_fn(
                {"params": self.state.params},  # no gradient flows through here
                batch["observations"],
                batch["actions"],
                name="critic",
            )
            v = self.state.apply_fn(
                {"params": params},  # gradient flows through here
                batch["observations"],
                name="value",
            )
            return iql_value_loss(q, v, self.config["expectile"])

        def actor_loss_fn(params):
            next_v = self.state.apply_fn(
                {"params": self.state.target_params},
                batch["next_observations"],
                name="value",
            )
            target_q = (
                batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
            )

            v = self.state.apply_fn(
                {"params": self.state.params},  # no gradient flows through here
                batch["observations"],
                name="value",
            )
            dist = self.state.apply_fn(
                {"params": params},  # gradient flows through here
                batch["observations"],
                train=True,
                rngs={"dropout": dropout_rng},
                name="actor",
            )
            mask = batch.get("actor_loss_mask", None)
            return iql_actor_loss(
                target_q,
                v,
                dist,
                batch["actions"],
                self.config["temperature"],
                mask=mask,
            )

        loss_fns = {
            "critic": critic_loss_fn,
            "value": value_loss_fn,
            "actor": actor_loss_fn,
        }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # update rng
        new_state = new_state.replace(rng=new_rng)

        # log learning rates
        info["actor_lr"] = self.lr_schedules["actor"](self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        argmax=False,
    ) -> jnp.ndarray:
        dist = self.state.apply_fn(observations, temperature=temperature, name="actor")
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, gripper_close_val=None, **kwargs):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        v = self.state.apply_fn(
            {"params": self.state.params}, batch["observations"], name="value"
        )
        next_v = self.state.apply_fn(
            {"params": self.state.target_params},
            batch["next_observations"],
            name="value",
        )
        target_q = batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
        q = self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            batch["actions"],
            name="critic",
        )

        metrics = {
            "log_probs": log_probs,
            "mse": ((dist.mode() - batch["actions"]) ** 2).sum(-1),
            "pi_actions": pi_actions,
            "online_v": v,
            "online_q": q,
            "target_q": target_q,
            "value_err": expectile_loss(target_q - v, self.config["expectile"]),
            "td_err": jnp.square(target_q - q),
            "advantage": target_q - v,
            "qf_advantage": q - v,
        }

        if gripper_close_val is not None:
            gripper_close_q = self.state.apply_fn(
                {"params": self.state.params},
                batch["observations"],
                jnp.broadcast_to(gripper_close_val, batch["actions"].shape),
                name="critic",
            )
            metrics.update(
                {
                    "gripper_close_q": gripper_close_q,
                    "gripper_close_adv": gripper_close_q - v,
                }
            )

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "state_dependent_std": False,
            "dropout": 0.0,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        actor_decay_steps: Optional[int] = None,
        # Algorithm config
        discount=0.95,
        expectile=0.9,
        temperature=1.0,
        target_update_rate=0.002,
    ):
        encoder_def = EncodingWrapper(
            encoder=encoder_def,
            use_proprio=use_proprio,
            stop_gradient=False,
        )

        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "value": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": encoder_def,
                "value": copy.deepcopy(encoder_def),
                "critic": copy.deepcopy(encoder_def),
            }

        network_kwargs["activate_final"] = True
        networks = {
            "actor": Policy(
                encoders["actor"],
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs,
            ),
            "value": ValueCritic(encoders["value"], MLP(**network_kwargs)),
            "critic": Critic(encoders["critic"], MLP(**network_kwargs)),
        }

        model_def = ModuleDict(networks)

        # no decay
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {
            "actor": lr_schedule,
            "value": lr_schedule,
            "critic": lr_schedule,
        }
        if actor_decay_steps is not None:
            lr_schedules["actor"] = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=actor_decay_steps,
                end_value=0.0,
            )
        txs = {k: optax.adam(v) for k, v in lr_schedules.items()}

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=observations,
            value=observations,
            critic=[observations, actions],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                discount=discount,
                temperature=temperature,
                target_update_rate=target_update_rate,
                expectile=expectile,
            )
        )
        return cls(state, config)
