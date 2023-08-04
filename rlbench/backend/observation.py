import numpy as np
from typing import List


class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 left_shoulder_rgb: np.ndarray,
                 left_shoulder_depth: np.ndarray,
                 left_shoulder_mask: np.ndarray,
                 left_shoulder_point_cloud: np.ndarray,
                 right_shoulder_rgb: np.ndarray,
                 right_shoulder_depth: np.ndarray,
                 right_shoulder_mask: np.ndarray,
                 right_shoulder_point_cloud: np.ndarray,
                 overhead_rgb: np.ndarray,
                 overhead_depth: np.ndarray,
                 overhead_mask: np.ndarray,
                 overhead_point_cloud: np.ndarray,
                 wrist_rgb: np.ndarray,
                 wrist_depth: np.ndarray,
                 wrist_mask: np.ndarray,
                 wrist_point_cloud: np.ndarray,
                 front_rgb: np.ndarray,
                 front_depth: np.ndarray,
                 front_mask: np.ndarray,
                 front_point_cloud: np.ndarray,
                 forward_rgb: np.ndarray,
                 forward_depth: np.ndarray,
                 forward_mask: np.ndarray,
                 forward_point_cloud: np.ndarray,
                 top_rgb: np.ndarray,
                 top_depth: np.ndarray,
                 top_mask: np.ndarray,
                 top_point_cloud: np.ndarray,
                 back_rgb: np.ndarray,
                 back_depth: np.ndarray,
                 back_mask: np.ndarray,
                 back_point_cloud: np.ndarray,
                 left_rgb: np.ndarray,
                 left_depth: np.ndarray,
                 left_mask: np.ndarray,
                 left_point_cloud: np.ndarray,
                 right_rgb: np.ndarray,
                 right_depth: np.ndarray,
                 right_mask: np.ndarray,
                 right_point_cloud: np.ndarray,
                 joint_velocities: np.ndarray,
                 joint_positions: np.ndarray,
                 joint_forces: np.ndarray,
                 gripper_open: float,
                 gripper_pose: np.ndarray,
                 gripper_matrix: np.ndarray,
                 gripper_joint_positions: np.ndarray,
                 gripper_touch_forces: np.ndarray,
                 task_low_dim_state: np.ndarray,
                 ignore_collisions: np.ndarray,
                 misc: dict):
        self.left_shoulder_rgb = left_shoulder_rgb
        self.left_shoulder_depth = left_shoulder_depth
        self.left_shoulder_mask = left_shoulder_mask
        self.left_shoulder_point_cloud = left_shoulder_point_cloud
        self.right_shoulder_rgb = right_shoulder_rgb
        self.right_shoulder_depth = right_shoulder_depth
        self.right_shoulder_mask = right_shoulder_mask
        self.right_shoulder_point_cloud = right_shoulder_point_cloud
        self.overhead_rgb = overhead_rgb
        self.overhead_depth = overhead_depth
        self.overhead_mask = overhead_mask
        self.overhead_point_cloud = overhead_point_cloud
        self.wrist_rgb = wrist_rgb
        self.wrist_depth = wrist_depth
        self.wrist_mask = wrist_mask
        self.wrist_point_cloud = wrist_point_cloud
        self.front_rgb = front_rgb
        self.front_depth = front_depth
        self.front_mask = front_mask
        self.front_point_cloud = front_point_cloud
        self.forward_rgb = forward_rgb
        self.forward_depth = forward_depth
        self.forward_mask = forward_mask
        self.forward_point_cloud = forward_point_cloud
        self.top_rgb = top_rgb
        self.top_depth = top_depth
        self.top_mask = top_mask
        self.top_point_cloud = top_point_cloud
        self.back_rgb = back_rgb
        self.back_depth = back_depth
        self.back_mask = back_mask
        self.back_point_cloud = back_point_cloud
        self.left_rgb = left_rgb
        self.left_depth = left_depth
        self.left_mask = left_mask
        self.left_point_cloud = left_point_cloud
        self.right_rgb = right_rgb
        self.right_depth = right_depth
        self.right_mask = right_mask
        self.right_point_cloud = right_point_cloud
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose
        self.gripper_matrix = gripper_matrix
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = gripper_touch_forces
        self.task_low_dim_state = task_low_dim_state
        self.ignore_collisions = ignore_collisions
        self.misc = misc

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [self.joint_velocities, self.joint_positions,
                     self.joint_forces,
                     self.gripper_pose, self.gripper_joint_positions,
                     self.gripper_touch_forces, self.task_low_dim_state]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
