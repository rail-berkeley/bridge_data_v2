"""
    This script processes the full CALVIN dataset, writing it into TFRecord format. 

    This script does not process language annotations (i.e. the resulting 
    dataset can only be used for goal conditioned learning). See the sister 
    script for code that only converts the language instruction labeled portion 
    of the dataset into TFRecord format.

    Written by Pranav Atreya (pranavatreya@berkeley.edu).
"""

import numpy as np
import tensorflow as tf 
from tqdm import tqdm 
import os
from multiprocessing import Pool

########## Dataset paths ###########
raw_dataset_path = "<path_to_unzipped_raw_CALVIN_dataset>"
tfrecord_dataset_path = "<desired_destination_path_of_processed_dataset>"

########## Main logic ###########
if not os.path.exists(tfrecord_dataset_path):
    os.mkdir(tfrecord_dataset_path)
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "validation")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "validation"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/A")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/A"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/B")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/B"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/C")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/C"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/D")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/D"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "validation/D")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "validation/D"))

def make_seven_characters(id):
    id = str(id)
    while len(id) < 7:
        id = "0" + id
    return id

def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

def process_trajectory(function_data):
    global raw_dataset_path, tfrecord_dataset_path
    idx_range, letter, ctr, split = function_data
    unique_pid = split + "_" + letter + "_" + str(ctr)

    start_id, end_id = idx_range[0].item(), idx_range[1].item()

    # We will filter the keys to only include what we need
    # Namely "rel_actions", "robot_obs", and "rgb_static"
    traj_rel_actions, traj_robot_obs, traj_rgb_static = [], [], []

    for ep_id in range(start_id, end_id+1): # end_id is inclusive
        #print(unique_pid + ": iter " + str(ep_id-start_id) + " of " + str(end_id-start_id))

        ep_id = make_seven_characters(ep_id)
        timestep_data = np.load(os.path.join(raw_dataset_path, split, "episode_" + ep_id + ".npz"))
        
        rel_actions = timestep_data["rel_actions"]
        traj_rel_actions.append(rel_actions)

        robot_obs = timestep_data["robot_obs"]
        traj_robot_obs.append(robot_obs)

        rgb_static = timestep_data["rgb_static"] # not normalized, so we have to do normalization in another script
        traj_rgb_static.append(rgb_static)
    
    traj_rel_actions, traj_robot_obs, traj_rgb_static = np.array(traj_rel_actions, dtype=np.float32), np.array(traj_robot_obs, dtype=np.float32), np.array(traj_rgb_static, dtype=np.uint8)

    # Determine the output path
    write_dir = os.path.join(tfrecord_dataset_path, split, letter, "traj" + str(ctr))
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    # Split the trajectory into 1000 timestep length segments
    for traj_idx in range(0, len(traj_rel_actions), 1000):
        traj_rel_actions_segment = traj_rel_actions[traj_idx : min(traj_idx+1000, len(traj_rel_actions))]
        traj_robot_obs_segment = traj_robot_obs[traj_idx : min(traj_idx+1000, len(traj_robot_obs))]
        traj_rgb_static_segment = traj_rgb_static[traj_idx : min(traj_idx+1000, len(traj_rgb_static))]

        # Write the TFRecord
        output_tfrecord_path = os.path.join(write_dir, str(traj_idx // 1000) + ".tfrecord")
        with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "actions" : tensor_feature(traj_rel_actions_segment),
                        "proprioceptive_states" : tensor_feature(traj_robot_obs_segment),
                        "image_states" : tensor_feature(traj_rgb_static_segment)
                    }
                )
            )
            writer.write(example.SerializeToString())

# Let's prepare the inputs to the process_trajectory function and then parallelize execution
function_inputs = []

# First let's do the train data
ep_start_end_ids = np.load(os.path.join(raw_dataset_path, "training", "ep_start_end_ids.npy"))

scene_info = np.load(os.path.join(raw_dataset_path, "training", "scene_info.npy"), allow_pickle=True)
scene_info = scene_info.item()

A_ctr, B_ctr, C_ctr, D_ctr = 0, 0, 0, 0
for idx_range in ep_start_end_ids:
    start_idx = idx_range[0].item()
    if start_idx <= scene_info["calvin_scene_D"][1]:
        ctr = D_ctr
        D_ctr += 1
        letter = "D"
    elif start_idx <= scene_info["calvin_scene_B"][1]: # This is actually correct. In ascending order we have D, B, C, A
        ctr = B_ctr
        B_ctr += 1
        letter = "B"
    elif start_idx <= scene_info["calvin_scene_C"][1]:
        ctr = C_ctr
        C_ctr += 1
        letter = "C"
    else:
        ctr = A_ctr
        A_ctr += 1
        letter = "A"

    function_inputs.append((idx_range, letter, ctr, "training"))

# Next let's do the validation data
ep_start_end_ids = np.load(os.path.join(raw_dataset_path, "validation", "ep_start_end_ids.npy"))

ctr = 0
for idx_range in ep_start_end_ids:
    function_inputs.append((idx_range, "D", ctr, "validation"))
    ctr += 1

with Pool(len(function_inputs)) as p: # We have one process per input because we are io bound, not cpu bound
    p.map(process_trajectory, function_inputs)
#for function_input in tqdm(function_inputs):  # If you want to process the dataset in a serialized fashion
#    process_trajectory(function_input)
