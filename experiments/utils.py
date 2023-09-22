import jax
import numpy as np
from pyquaternion import Quaternion

def stack_obs(obs):
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return jax.tree_map(
        lambda x: np.stack(x), dict_list, is_leaf=lambda x: type(x) == list
    )

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
