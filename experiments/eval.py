#!/usr/bin/env python3

# TODO: remove this `eval.py`, since it is being replaced by `eval_gc` and `eval_lc`

import sys
import os
import json
import time
from datetime import datetime
import traceback

import matplotlib
from absl import app, flags, logging
matplotlib.use("Agg")

import cv2
import numpy as np
import tensorflow as tf

import jax
from pyquaternion import Quaternion
from PIL import Image

from flax.training import checkpoints
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus


##############################################################################

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("checkpoint_weights_path", None, "Path to checkpoint weights", required=True)
flags.DEFINE_multi_string("checkpoint_config_path", None, "Path to checkpoint config", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_string(
    "goal_image_path",
    None,
    "Path to a single goal image",
)
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", None, "Goal position")
flags.DEFINE_spaceseplist("initial_eep", None, "Initial position")
flags.DEFINE_bool("high_res", False, "Save high-res video and goal")
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1

FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

##############################################################################

def state_to_eep(xyz_coor, zangle: float):
    """
    Implement the state to eep function.
    Refered to `bridge_data_robot`'s `widowx_controller/widowx_controller.py`
    return a 4x4 matrix
    """
    assert len(xyz_coor) == 3
    DEFAULT_ROTATION = np.array([[0 , 0, 1.0],
                             [0, 1.0,  0],
                             [-1.0,  0, 0]])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) \
        * Quaternion(matrix=DEFAULT_ROTATION)
    new_pose[:3, :3] = new_quat.rotation_matrix
    # yaw, pitch, roll = quat.yaw_pitch_roll
    return new_pose


def mat_to_xyzrpy(mat: np.ndarray):
    """return a 6-dim vector with xyz and rpy"""
    assert mat.shape == (4, 4), "mat must be a 4x4 matrix"
    xyz = mat[:3, -1]
    quat = Quaternion(matrix=mat[:3, :3])
    yaw, pitch, roll = quat.yaw_pitch_roll
    return np.concatenate([xyz, [roll, pitch, yaw]])


def unnormalize_action(action, mean, std):
    return action * std + mean


def load_checkpoint(checkpoint_weights_path, checkpoint_config_path):
    with open(checkpoint_config_path, 'r') as f:
        config = json.load(f)    
    
    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])

    example_batch = {
        "observations": {"image": np.zeros((128, 128, 3), dtype=np.uint8)},
        "goals": {"image": np.zeros((128, 128, 3), dtype=np.uint8)},
        "actions": np.zeros(7, dtype=np.float32),
    }

    # create agent from wandb config
    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[config["agent"]].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **config["agent_kwargs"],
    )

    # load action metadata from wandb
    action_metadata = config["bridgedata_config"]["action_metadata"]
    action_mean = np.array(action_metadata["mean"])
    action_std = np.array(action_metadata["std"])

    # hydrate agent with parameters from checkpoint
    agent = checkpoints.restore_checkpoint(checkpoint_weights_path, agent)

    return agent, action_mean, action_std


def main(_):
    # policies is a dict from run_name to (agent, action_mean, action_std)
    policies = {}
    for checkpoint_weights_path, checkpoint_config_path in zip(FLAGS.checkpoint_weights_path, FLAGS.checkpoint_config_path):
        assert tf.io.gfile.exists(checkpoint_weights_path), checkpoint_weights_path
        assert tf.io.gfile.exists(checkpoint_config_path), checkpoint_config_path
        checkpoint_num = int(checkpoint_weights_path.split("_")[-1])
        agent, action_mean, action_std = load_checkpoint(checkpoint_weights_path=checkpoint_weights_path, checkpoint_config_path=checkpoint_config_path)
        policies[f"{checkpoint_num}"] = (agent, action_mean, action_std)
    
    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    # init environment
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(WidowXConfigs.DefaultEnvParams, image_size=256)

    while widowx_client.get_observation() is None:
        print("Waiting for environment to start...")
        time.sleep(1)

    # load image goal
    image_goal = None
    if FLAGS.goal_image_path is not None:
        image_goal = np.array(Image.open(FLAGS.goal_image_path))

    # goal sampling loop
    while True:
        # ask for new goal
        if image_goal is None:
            print("Taking a new goal...")
            ch = "y"
        else:
            ch = input("Taking a new goal? [y/n]")
        if ch == "y":
            if FLAGS.goal_eep is not None:
                assert isinstance(FLAGS.goal_eep, list)
                goal_eep = [float(e) for e in FLAGS.goal_eep]
            else:
                low_bound = [0.24, -0.1, 0.05, -1.57, 0]
                high_bound = [0.4, 0.20, 0.15, 1.57, 0]
                goal_eep = np.random.uniform(low_bound[:3], high_bound[:3])
            widowx_client.move_gripper(1.0) # open gripper

            # retry move action until success
            goal_eep = state_to_eep(goal_eep, 0)
            move_status = None
            while move_status != WidowXStatus.SUCCESS:
                move_status = widowx_client.move(goal_eep)

            time.sleep(1.5)
            input("Press [Enter] when ready for taking the goal image. ")

            obs = widowx_client.get_observation()
            while obs is None:
                print("WARNING retrying to get observation...")
                obs = widowx_client.get_observation()
                time.sleep(1)

            image_goal = (
                obs["image"].reshape(3, 128, 128).transpose(1, 2, 0) * 255
            ).astype(np.uint8)
            full_goal_image = obs["full_image"]

        # ask for which policy to use
        if len(policies) == 1:
            policy_idx = 0
            input("Press [Enter] to start.")
        else:
            print("policies:")
            for i, name in enumerate(policies.keys()):
                print(f"{i}) {name}")
            policy_idx = int(input("select policy: "))

        policy_name = list(policies.keys())[policy_idx]
        agent, action_mean, action_std = policies[policy_name]
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
        rng = jax.random.PRNGKey(0)
        last_tstep = time.time()
        images = []
        full_images = []
        t = 0
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
                        cv2.imshow("img_view", obs["full_image"])
                        cv2.waitKey(10)

                    image_obs = (
                        obs["image"].reshape(3, 128, 128).transpose(1, 2, 0) * 255
                    ).astype(np.uint8)
                    if FLAGS.high_res:
                        full_images.append(Image.fromarray(obs["full_image"]))
                    obs = {"image": image_obs, "proprio": obs["state"]}
                    goal_obs = {
                        "image": image_goal,
                    }

                    last_tstep = time.time()
                    rng, key = jax.random.split(rng)
                    action = np.array(
                        agent.sample_actions(obs, goal_obs, seed=key, argmax=True)
                    )
                    action = unnormalize_action(action, action_mean, action_std)
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

                    print(f"timestep {t}, calling step action {action}")
                    widowx_client.step_action(action)
                    # time.sleep(STEP_DURATION)

                    # save image
                    image_formatted = np.concatenate((image_goal, image_obs), axis=0)
                    images.append(Image.fromarray(image_formatted))

                    t += 1
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}_{policy_name}_sticky_{STICKY_GRIPPER_NUM_STEPS}.gif",
            )
            print(f"Saving Video at {save_path}")
            images[0].save(
                save_path,
                format="GIF",
                append_images=images[1:],
                save_all=True,
                duration=200,
                loop=0,
            )
        # save high-res video
        if FLAGS.high_res:
            base_path = os.path.join(FLAGS.video_save_path, "high_res")
            os.makedirs(base_path, exist_ok=True)
            print(f"Saving Video and Goal at {base_path}")
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_path = os.path.join(
                base_path,
                f"{curr_time}_{policy_name}_sticky_{STICKY_GRIPPER_NUM_STEPS}.gif",
            )
            full_images[0].save(
                video_path,
                format="GIF",
                append_images=full_images[1:],
                save_all=True,
                duration=200,
                loop=0,
            )
            goal_path = os.path.join(base_path, f"{curr_time}_{policy_name}.png")
            plt.imshow(full_goal_image)
            plt.axis("off")
            plt.savefig(goal_path, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    app.run(main)
