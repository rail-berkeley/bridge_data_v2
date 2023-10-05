#!/usr/bin/env python3

import sys
import os
import time
from datetime import datetime
import traceback
from collections import deque
import json

from absl import app, flags, logging

import numpy as np
import tensorflow as tf

import cv2
import jax
from PIL import Image
import imageio

from flax.training import checkpoints
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
from jaxrl_m.data.text_processing import text_processors

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXStatus
from utils import state_to_eep, stack_obs

##############################################################################

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_multi_string(
    "checkpoint_config_path", None, "Path to checkpoint config JSON", required=True
)
flags.DEFINE_string("goal_type", None, "Goal type", required=True)
flags.DEFINE_integer("im_size", None, "Image size", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_string("goal_image_path", None, "Path to a single goal image")  # not used by lc
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")  # not used by lc
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_integer("act_exec_horizon", 1, "Action sequence length")
flags.DEFINE_bool("deterministic", True, "Whether to sample action deterministically")
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
FIXED_STD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

##############################################################################


def load_checkpoint(checkpoint_weights_path, checkpoint_config_path):
    with open(checkpoint_config_path, "r") as f:
        config = json.load(f)

    # create encoder from wandb config
    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])

    act_pred_horizon = config["dataset_kwargs"].get("act_pred_horizon")
    obs_horizon = config["dataset_kwargs"].get("obs_horizon")

    # Set action
    if act_pred_horizon is not None:
        example_actions = np.zeros((1, act_pred_horizon, 7), dtype=np.float32)
    else:
        example_actions = np.zeros((1, 7), dtype=np.float32)

    # Set observations
    if obs_horizon is None:
        img_obs_shape = (1, FLAGS.im_size, FLAGS.im_size, 3)
    else:
        img_obs_shape = (1, obs_horizon, FLAGS.im_size, FLAGS.im_size, 3)
    example_obs = {"image": np.zeros(img_obs_shape, dtype=np.uint8)}

    # Set goals
    if FLAGS.goal_type == "gc":
        example_goals = {
            "image": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
        }
    elif FLAGS.goal_type == "lc":
        example_goals = {"language": np.zeros((1, 512), dtype=np.float32)}
    else:
        raise ValueError(f"Unknown goal type: {FLAGS.goal_type}")

    # create agent from wandb config
    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[config["agent"]].create(
        rng=construct_rng,
        observations=example_obs,
        goals=example_goals,
        actions=example_actions,
        encoder_def=encoder_def,
        **config["agent_kwargs"],
    )

    # load action metadata from wandb
    action_proprio_metadata = config["bridgedata_config"]["action_proprio_metadata"]
    action_mean = np.array(action_proprio_metadata["action"]["mean"])
    action_std = np.array(action_proprio_metadata["action"]["std"])

    # hydrate agent with parameters from checkpoint
    agent = checkpoints.restore_checkpoint(checkpoint_weights_path, agent)

    def get_action(obs, goal_obs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        action = jax.device_get(
            agent.sample_actions(obs, goal_obs, seed=key, argmax=FLAGS.deterministic)
        )
        action = action * action_std + action_mean
        return action

    text_processor = None
    if FLAGS.goal_type == "lc":
        text_processor = text_processors[config["text_processor"]](
            **config["text_processor_kwargs"]
        )

    return get_action, obs_horizon, text_processor


def request_goal_image(image_goal, widowx_client):
    """
    Request a new goal image from the user.
    """
    # ask for new goal
    if image_goal is None:
        print("Taking a new goal...")
        ch = "y"
    else:
        ch = input("Taking a new goal? [y/n]")
    if ch == "y":
        assert isinstance(FLAGS.goal_eep, list)
        goal_eep = [float(e) for e in FLAGS.goal_eep]
        widowx_client.move_gripper(1.0)  # open gripper

        # retry move action until success
        goal_eep = state_to_eep(goal_eep, 0)
        move_status = None
        while move_status != WidowXStatus.SUCCESS:
            move_status = widowx_client.move(goal_eep, duration=1.5)

        input("Press [Enter] when ready for taking the goal image. ")

        obs = widowx_client.get_observation()
        while obs is None:
            print("WARNING retrying to get observation...")
            obs = widowx_client.get_observation()
            time.sleep(1)

        image_goal = (
            obs["image"].reshape(3, FLAGS.im_size, FLAGS.im_size).transpose(1, 2, 0)
            * 255
        ).astype(np.uint8)
    return image_goal


def request_goal_language(instruction, text_processor):
    """
    Request a new goal language from the user.
    """
    # ask for new instruction
    if instruction is None:
        ch = "y"
    else:
        ch = input("New instruction? [y/n]")
    if ch == "y":
        instruction = text_processor.encode(input("Instruction?"))
    return instruction


##############################################################################

def main(_):
    assert len(FLAGS.checkpoint_weights_path) == len(FLAGS.checkpoint_config_path)

    # policies is a dict from run_name to get_action function
    policies = {}
    for checkpoint_weights_path, checkpoint_config_path in zip(
        FLAGS.checkpoint_weights_path, FLAGS.checkpoint_config_path
    ):
        assert tf.io.gfile.exists(checkpoint_weights_path), checkpoint_weights_path
        checkpoint_num = int(checkpoint_weights_path.split("_")[-1])
        run_name = checkpoint_config_path.split("/")[-1]
        policies[f"{run_name}-{checkpoint_num}"] = load_checkpoint(
            checkpoint_weights_path, checkpoint_config_path
        )

    assert isinstance(FLAGS.initial_eep, list)
    initial_eep = [float(e) for e in FLAGS.initial_eep]
    start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])

    # set up environment
    env_params = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": WORKSPACE_BOUNDS,
        "action_clipping": "xyz",
        "catch_environment_except": False,
        "start_state": list(start_state),
        "return_full_image": False,
        "camera_topics": CAMERA_TOPICS,
    }
    widowx_client = WidowXClient(FLAGS.ip, FLAGS.port)
    widowx_client.init(env_params, image_size=FLAGS.im_size)

    # load goals
    if FLAGS.goal_type == "gc":
        image_goal = None
        if FLAGS.goal_image_path is not None:
            image_goal = np.array(Image.open(FLAGS.goal_image_path))
    elif FLAGS.goal_type == "lc":
        instruction = None

    # goal sampling loop
    while True:
        # ask for which policy to use
        if len(policies) == 1:
            policy_idx = 0
        else:
            print("policies:")
            for i, name in enumerate(policies.keys()):
                print(f"{i}) {name}")
            policy_idx = int(input("select policy: "))

        policy_name = list(policies.keys())[policy_idx]
        get_action, obs_horizon, text_processors = policies[policy_name]

        # show img for monitoring
        if FLAGS.show_image:
            obs = widowx_client.get_observation()
            while obs is None:
                print("Waiting for observations...")
                obs = widowx_client.get_observation()
                time.sleep(1)
            bgr_img = cv2.cvtColor(obs["full_image"], cv2.COLOR_RGB2BGR)
            cv2.imshow("img_view", bgr_img)
            cv2.waitKey(100)

        # request goal
        if FLAGS.goal_type == "gc":
            image_goal = request_goal_image(image_goal, widowx_client)
            goal_obs = {"image": image_goal}
            input("Press [Enter] to start.")
        elif FLAGS.goal_type == "lc":
            instruction = request_goal_language(None, text_processors)
            goal_obs = {"language": instruction}
        else:
            raise ValueError(f"Unknown goal type: {FLAGS.goal_type}")

        # reset env
        widowx_client.reset()
        time.sleep(2.5)

        # move to initial position
        if FLAGS.initial_eep is not None:
            assert isinstance(FLAGS.initial_eep, list)
            initial_eep = [float(e) for e in FLAGS.initial_eep]
            eep = state_to_eep(initial_eep, 0)

            # retry move action until success
            move_status = None
            while move_status != WidowXStatus.SUCCESS:
                move_status = widowx_client.move(eep, duration=1.5)

        # do rollout
        last_tstep = time.time()
        images = []
        image_goals = []  # only used when goal_type == "gc"
        t = 0
        if obs_horizon is not None:
            obs_hist = deque(maxlen=obs_horizon)
        # keep track of our own gripper state to implement sticky gripper
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0
        try:
            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:

                    obs = widowx_client.get_observation()
                    if obs is None:
                        print("WARNING retrying to get observation...")
                        continue

                    if FLAGS.show_image:
                        bgr_img = cv2.cvtColor(obs["full_image"], cv2.COLOR_RGB2BGR)
                        cv2.imshow("img_view", bgr_img)
                        cv2.waitKey(10)

                    image_obs = (
                        obs["image"]
                        .reshape(3, FLAGS.im_size, FLAGS.im_size)
                        .transpose(1, 2, 0)
                        * 255
                    ).astype(np.uint8)
                    obs = {"image": image_obs, "proprio": obs["state"]}
                    if obs_horizon is not None:
                        if len(obs_hist) == 0:
                            obs_hist.extend([obs] * obs_horizon)
                        else:
                            obs_hist.append(obs)
                        obs = stack_obs(obs_hist)

                    last_tstep = time.time()
                    actions = get_action(obs, goal_obs)

                    if len(actions.shape) == 1:
                        actions = actions[None]
                    for i in range(FLAGS.act_exec_horizon):
                        action = actions[i]
                        action += np.random.normal(0, FIXED_STD)

                        # sticky gripper logic
                        if (action[-1] < 0.5) != is_gripper_closed:
                            num_consecutive_gripper_change_actions += 1
                        else:
                            num_consecutive_gripper_change_actions = 0

                        if (
                            num_consecutive_gripper_change_actions
                            >= STICKY_GRIPPER_NUM_STEPS
                        ):
                            is_gripper_closed = not is_gripper_closed
                            num_consecutive_gripper_change_actions = 0

                        action[-1] = 0.0 if is_gripper_closed else 1.0

                        # remove degrees of freedom
                        if NO_PITCH_ROLL:
                            action[3] = 0
                            action[4] = 0
                        if NO_YAW:
                            action[5] = 0

                        # perform environment step
                        widowx_client.step_action(action, blocking=FLAGS.blocking)

                        # save image
                        images.append(image_obs)
                        if FLAGS.goal_type == "gc":
                            image_goals.append(image_goal)

                        t += 1
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}_{policy_name}_sticky_{STICKY_GRIPPER_NUM_STEPS}.mp4",
            )
            if FLAGS.goal_type == "gc":
                video = np.concatenate([np.stack(image_goals), np.stack(images)], axis=1)
                imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)
            else:
                imageio.mimsave(save_path, images, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)
