import importlib
import pickle
from os import listdir
from os.path import join, exists
from typing import List

import numpy as np
from PIL import Image
from natsort import natsorted
from pyrep.objects import VisionSensor

from rlbench.backend.const import *
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig


class InvalidTaskName(Exception):
    pass


def name_to_task_class(task_file: str):
    name = task_file.replace('.py', '')
    class_name = ''.join([w[0].upper() + w[1:] for w in name.split('_')])
    try:
        mod = importlib.import_module("rlbench.tasks.%s" % name)
        mod = importlib.reload(mod)
    except ModuleNotFoundError as e:
        raise InvalidTaskName(
            "The task file '%s' does not exist or cannot be compiled."
            % name) from e
    try:
        task_class = getattr(mod, class_name)
    except AttributeError as e:
        raise InvalidTaskName(
            "Cannot find the class name '%s' in the file '%s'."
            % (class_name, name)) from e
    return task_class


def get_stored_demos(amount: int, image_paths: bool, dataset_root: str,
                     variation_number: int, task_name: str,
                     obs_config: ObservationConfig,
                     random_selection: bool = True,
                     from_episode_number: int = 0) -> List[Demo]:

    task_root = join(dataset_root, task_name)
    if not exists(task_root):
        raise RuntimeError("Can't find the demos for %s at: %s" % (
            task_name, task_root))

    if variation_number == -1:
        # All variations
        examples_path = join(
            task_root, VARIATIONS_ALL_FOLDER,
            EPISODES_FOLDER)
        examples = listdir(examples_path)
    else:
        # Sample an amount of examples for the variation of this task
        examples_path = join(
            task_root, VARIATIONS_FOLDER % variation_number,
            EPISODES_FOLDER)
        examples = listdir(examples_path)

    if amount == -1:
        amount = len(examples)
    if amount > len(examples):
        raise RuntimeError(
            'You asked for %d examples, but only %d were available.' % (
                amount, len(examples)))
    if random_selection:
        selected_examples = np.random.choice(examples, amount, replace=False)
    else:
        selected_examples = natsorted(
            examples)[from_episode_number:from_episode_number+amount]

    # Process these examples (e.g. loading observations)
    demos = []
    for example in selected_examples:
        example_path = join(examples_path, example)
        with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pickle.load(f)

        if variation_number == -1:
            with open(join(example_path, VARIATION_NUMBER), 'rb') as f:
                obs.variation_number = pickle.load(f)
        else:
            obs.variation_number = variation_number

        # language description
        episode_descriptions_f = join(example_path, VARIATION_DESCRIPTIONS)
        if exists(episode_descriptions_f):
            with open(episode_descriptions_f, 'rb') as f:
                descriptions = pickle.load(f)
        else:
            descriptions = ["unknown task description"]

        l_sh_rgb_f = join(example_path, LEFT_SHOULDER_RGB_FOLDER)
        l_sh_depth_f = join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
        l_sh_mask_f = join(example_path, LEFT_SHOULDER_MASK_FOLDER)
        r_sh_rgb_f = join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
        r_sh_depth_f = join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
        r_sh_mask_f = join(example_path, RIGHT_SHOULDER_MASK_FOLDER)
        oh_rgb_f = join(example_path, OVERHEAD_RGB_FOLDER)
        oh_depth_f = join(example_path, OVERHEAD_DEPTH_FOLDER)
        oh_mask_f = join(example_path, OVERHEAD_MASK_FOLDER)
        wrist_rgb_f = join(example_path, WRIST_RGB_FOLDER)
        wrist_depth_f = join(example_path, WRIST_DEPTH_FOLDER)
        wrist_mask_f = join(example_path, WRIST_MASK_FOLDER)
        front_rgb_f = join(example_path, FRONT_RGB_FOLDER)
        front_depth_f = join(example_path, FRONT_DEPTH_FOLDER)
        front_mask_f = join(example_path, FRONT_MASK_FOLDER)
        forward_rgb_f = join(example_path, FORWARD_RGB_FOLDER)
        forward_depth_f = join(example_path, FORWARD_DEPTH_FOLDER)
        forward_mask_f = join(example_path, FORWARD_MASK_FOLDER)
        top_rgb_f = join(example_path, TOP_RGB_FOLDER)
        top_depth_f = join(example_path, TOP_DEPTH_FOLDER)
        top_mask_f = join(example_path, TOP_MASK_FOLDER)
        back_rgb_f = join(example_path, BACK_RGB_FOLDER)
        back_depth_f = join(example_path, BACK_DEPTH_FOLDER)
        back_mask_f = join(example_path, BACK_MASK_FOLDER)
        left_rgb_f = join(example_path, LEFT_RGB_FOLDER)
        left_depth_f = join(example_path, LEFT_DEPTH_FOLDER)
        left_mask_f = join(example_path, LEFT_MASK_FOLDER)
        right_rgb_f = join(example_path, RIGHT_RGB_FOLDER)
        right_depth_f = join(example_path, RIGHT_DEPTH_FOLDER)
        right_mask_f = join(example_path, RIGHT_MASK_FOLDER)

        num_steps = len(obs)

        if not (num_steps == len(listdir(l_sh_rgb_f)) == len(
                listdir(l_sh_depth_f)) == len(listdir(r_sh_rgb_f)) == len(
                listdir(r_sh_depth_f)) == len(listdir(oh_rgb_f)) == len(
                listdir(oh_depth_f)) == len(listdir(wrist_rgb_f)) == len(
                listdir(wrist_depth_f)) == len(listdir(front_rgb_f)) == len(
                listdir(front_depth_f))):
            raise RuntimeError('Broken dataset assumption')

        for i in range(num_steps):
            # descriptions
            obs[i].misc['descriptions'] = descriptions

            si = IMAGE_FORMAT % i
            if obs_config.left_shoulder_camera.rgb:
                obs[i].left_shoulder_rgb = join(l_sh_rgb_f, si)
            if obs_config.left_shoulder_camera.depth or obs_config.left_shoulder_camera.point_cloud:
                obs[i].left_shoulder_depth = join(l_sh_depth_f, si)
            if obs_config.left_shoulder_camera.mask:
                obs[i].left_shoulder_mask = join(l_sh_mask_f, si)
            if obs_config.right_shoulder_camera.rgb:
                obs[i].right_shoulder_rgb = join(r_sh_rgb_f, si)
            if obs_config.right_shoulder_camera.depth or obs_config.right_shoulder_camera.point_cloud:
                obs[i].right_shoulder_depth = join(r_sh_depth_f, si)
            if obs_config.right_shoulder_camera.mask:
                obs[i].right_shoulder_mask = join(r_sh_mask_f, si)
            if obs_config.overhead_camera.rgb:
                obs[i].overhead_rgb = join(oh_rgb_f, si)
            if obs_config.overhead_camera.depth or obs_config.overhead_camera.point_cloud:
                obs[i].overhead_depth = join(oh_depth_f, si)
            if obs_config.overhead_camera.mask:
                obs[i].overhead_mask = join(oh_mask_f, si)
            if obs_config.wrist_camera.rgb:
                obs[i].wrist_rgb = join(wrist_rgb_f, si)
            if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                obs[i].wrist_depth = join(wrist_depth_f, si)
            if obs_config.wrist_camera.mask:
                obs[i].wrist_mask = join(wrist_mask_f, si)
            if obs_config.front_camera.rgb:
                obs[i].front_rgb = join(front_rgb_f, si)
            if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                obs[i].front_depth = join(front_depth_f, si)
            if obs_config.front_camera.mask:
                obs[i].front_mask = join(front_mask_f, si)
            if obs_config.forward_camera.rgb:
                obs[i].forward_rgb = join(forward_rgb_f, si)
            if obs_config.forward_camera.depth or obs_config.forward_camera.point_cloud:
                obs[i].forward_depth = join(forward_depth_f, si)
            if obs_config.forward_camera.mask:
                obs[i].forward_mask = join(forward_mask_f, si)
            if obs_config.top_camera.rgb:
                obs[i].top_rgb = join(top_rgb_f, si)
            if obs_config.top_camera.depth or obs_config.top_camera.point_cloud:
                obs[i].top_depth = join(top_depth_f, si)
            if obs_config.top_camera.mask:
                obs[i].top_mask = join(top_mask_f, si)
            if obs_config.back_camera.rgb:
                obs[i].back_rgb = join(back_rgb_f, si)
            if obs_config.back_camera.depth or obs_config.back_camera.point_cloud:
                obs[i].back_depth = join(back_depth_f, si)
            if obs_config.back_camera.mask:
                obs[i].back_mask = join(back_mask_f, si)
            if obs_config.left_camera.rgb:
                obs[i].left_rgb = join(left_rgb_f, si)
            if obs_config.left_camera.depth or obs_config.left_camera.point_cloud:
                obs[i].left_depth = join(left_depth_f, si)
            if obs_config.left_camera.mask:
                obs[i].left_mask = join(left_mask_f, si)
            if obs_config.right_camera.rgb:
                obs[i].right_rgb = join(right_rgb_f, si)
            if obs_config.right_camera.depth or obs_config.right_camera.point_cloud:
                obs[i].right_depth = join(right_depth_f, si)
            if obs_config.right_camera.mask:
                obs[i].right_mask = join(right_mask_f, si)

            # Remove low dim info if necessary
            if not obs_config.joint_velocities:
                obs[i].joint_velocities = None
            if not obs_config.joint_positions:
                obs[i].joint_positions = None
            if not obs_config.joint_forces:
                obs[i].joint_forces = None
            if not obs_config.gripper_open:
                obs[i].gripper_open = None
            if not obs_config.gripper_pose:
                obs[i].gripper_pose = None
            if not obs_config.gripper_joint_positions:
                obs[i].gripper_joint_positions = None
            if not obs_config.gripper_touch_forces:
                obs[i].gripper_touch_forces = None
            if not obs_config.task_low_dim_state:
                obs[i].task_low_dim_state = None

        if not image_paths:
            for i in range(num_steps):
                if obs_config.left_shoulder_camera.rgb:
                    obs[i].left_shoulder_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].left_shoulder_rgb),
                            obs_config.left_shoulder_camera.image_size))
                if obs_config.right_shoulder_camera.rgb:
                    obs[i].right_shoulder_rgb = np.array(
                        _resize_if_needed(Image.open(
                        obs[i].right_shoulder_rgb),
                            obs_config.right_shoulder_camera.image_size))
                if obs_config.overhead_camera.rgb:
                    obs[i].overhead_rgb = np.array(
                        _resize_if_needed(Image.open(
                        obs[i].overhead_rgb),
                            obs_config.overhead_camera.image_size))
                if obs_config.wrist_camera.rgb:
                    obs[i].wrist_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist_rgb),
                            obs_config.wrist_camera.image_size))
                if obs_config.front_camera.rgb:
                    obs[i].front_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].front_rgb),
                            obs_config.front_camera.image_size))
                if obs_config.forward_camera.rgb:
                    obs[i].forward_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].forward_rgb),
                            obs_config.forward_camera.image_size))
                if obs_config.top_camera.rgb:
                    obs[i].top_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].top_rgb),
                            obs_config.top_camera.image_size))
                if obs_config.back_camera.rgb:
                    obs[i].back_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].back_rgb),
                            obs_config.back_camera.image_size))
                if obs_config.left_camera.rgb:
                    obs[i].left_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].left_rgb),
                            obs_config.left_camera.image_size))
                if obs_config.right_camera.rgb:
                    obs[i].right_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].right_rgb),
                            obs_config.right_camera.image_size))

                if obs_config.left_shoulder_camera.depth or obs_config.left_shoulder_camera.point_cloud:
                    l_sh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].left_shoulder_depth),
                            obs_config.left_shoulder_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['left_shoulder_camera_near']
                    far = obs[i].misc['left_shoulder_camera_far']
                    l_sh_depth_m = near + l_sh_depth * (far - near)
                    if obs_config.left_shoulder_camera.depth:
                        d = l_sh_depth_m if obs_config.left_shoulder_camera.depth_in_meters else l_sh_depth
                        obs[i].left_shoulder_depth = obs_config.left_shoulder_camera.depth_noise.apply(d)
                    else:
                        obs[i].left_shoulder_depth = None

                if obs_config.right_shoulder_camera.depth or obs_config.right_shoulder_camera.point_cloud:
                    r_sh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].right_shoulder_depth),
                            obs_config.right_shoulder_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['right_shoulder_camera_near']
                    far = obs[i].misc['right_shoulder_camera_far']
                    r_sh_depth_m = near + r_sh_depth * (far - near)
                    if obs_config.right_shoulder_camera.depth:
                        d = r_sh_depth_m if obs_config.right_shoulder_camera.depth_in_meters else r_sh_depth
                        obs[i].right_shoulder_depth = obs_config.right_shoulder_camera.depth_noise.apply(d)
                    else:
                        obs[i].right_shoulder_depth = None

                if obs_config.overhead_camera.depth or obs_config.overhead_camera.point_cloud:
                    oh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].overhead_depth),
                            obs_config.overhead_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['overhead_camera_near']
                    far = obs[i].misc['overhead_camera_far']
                    oh_depth_m = near + oh_depth * (far - near)
                    if obs_config.overhead_camera.depth:
                        d = oh_depth_m if obs_config.overhead_camera.depth_in_meters else oh_depth
                        obs[i].overhead_depth = obs_config.overhead_camera.depth_noise.apply(d)
                    else:
                        obs[i].overhead_depth = None

                if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                    wrist_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist_depth),
                            obs_config.wrist_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['wrist_camera_near']
                    far = obs[i].misc['wrist_camera_far']
                    wrist_depth_m = near + wrist_depth * (far - near)
                    if obs_config.wrist_camera.depth:
                        d = wrist_depth_m if obs_config.wrist_camera.depth_in_meters else wrist_depth
                        obs[i].wrist_depth = obs_config.wrist_camera.depth_noise.apply(d)
                    else:
                        obs[i].wrist_depth = None

                if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                    front_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].front_depth),
                            obs_config.front_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['front_camera_near']
                    far = obs[i].misc['front_camera_far']
                    front_depth_m = near + front_depth * (far - near)
                    if obs_config.front_camera.depth:
                        d = front_depth_m if obs_config.front_camera.depth_in_meters else front_depth
                        obs[i].front_depth = obs_config.front_camera.depth_noise.apply(d)
                    else:
                        obs[i].front_depth = None
                
                if obs_config.forward_camera.depth or obs_config.forward_camera.point_cloud:
                    forward_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].forward_depth),
                            obs_config.forward_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['forward_camera_near']
                    far = obs[i].misc['forward_camera_far']
                    forward_depth_m = near + forward_depth * (far - near)
                    if obs_config.forward_camera.depth:
                        d = forward_depth_m if obs_config.forward_camera.depth_in_meters else forward_depth
                        obs[i].forward_depth = obs_config.forward_camera.depth_noise.apply(d)
                    else:
                        obs[i].forward_depth = None
                
                if obs_config.top_camera.depth or obs_config.top_camera.point_cloud:
                    top_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].top_depth),
                            obs_config.top_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['top_camera_near']
                    far = obs[i].misc['top_camera_far']
                    top_depth_m = near + top_depth * (far - near)
                    if obs_config.top_camera.depth:
                        d = top_depth_m if obs_config.top_camera.depth_in_meters else top_depth
                        obs[i].top_depth = obs_config.top_camera.depth_noise.apply(d)
                    else:
                        obs[i].top_depth = None
                if obs_config.back_camera.depth or obs_config.back_camera.point_cloud:
                    back_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].back_depth),
                            obs_config.back_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['back_camera_near']
                    far = obs[i].misc['back_camera_far']
                    back_depth_m = near + back_depth * (far - near)
                    if obs_config.back_camera.depth:
                        d = back_depth_m if obs_config.back_camera.depth_in_meters else back_depth
                        obs[i].back_depth = obs_config.back_camera.depth_noise.apply(d)
                    else:
                        obs[i].back_depth = None
                
                if obs_config.left_camera.depth or obs_config.left_camera.point_cloud:
                    left_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].left_depth),
                            obs_config.left_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['left_camera_near']
                    far = obs[i].misc['left_camera_far']
                    left_depth_m = near + left_depth * (far - near)
                    if obs_config.left_camera.depth:
                        d = left_depth_m if obs_config.left_camera.depth_in_meters else left_depth
                        obs[i].left_depth = obs_config.left_camera.depth_noise.apply(d)
                    else:
                        obs[i].left_depth = None
                
                if obs_config.right_camera.depth or obs_config.right_camera.point_cloud:
                    right_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].right_depth),
                            obs_config.right_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['right_camera_near']
                    far = obs[i].misc['right_camera_far']
                    right_depth_m = near + right_depth * (far - near)
                    if obs_config.right_camera.depth:
                        d = right_depth_m if obs_config.right_camera.depth_in_meters else right_depth
                        obs[i].right_depth = obs_config.right_camera.depth_noise.apply(d)
                    else:
                        obs[i].right_depth = None

                if obs_config.left_shoulder_camera.point_cloud:
                    obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        l_sh_depth_m,
                        obs[i].misc['left_shoulder_camera_extrinsics'],
                        obs[i].misc['left_shoulder_camera_intrinsics'])
                if obs_config.right_shoulder_camera.point_cloud:
                    obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        r_sh_depth_m,
                        obs[i].misc['right_shoulder_camera_extrinsics'],
                        obs[i].misc['right_shoulder_camera_intrinsics'])
                if obs_config.overhead_camera.point_cloud:
                    obs[i].overhead_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        oh_depth_m,
                        obs[i].misc['overhead_camera_extrinsics'],
                        obs[i].misc['overhead_camera_intrinsics'])
                if obs_config.wrist_camera.point_cloud:
                    obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        wrist_depth_m,
                        obs[i].misc['wrist_camera_extrinsics'],
                        obs[i].misc['wrist_camera_intrinsics'])
                if obs_config.front_camera.point_cloud:
                    obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        front_depth_m,
                        obs[i].misc['front_camera_extrinsics'],
                        obs[i].misc['front_camera_intrinsics'])
                if obs_config.forward_camera.point_cloud:
                    obs[i].forward_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        forward_depth_m,
                        obs[i].misc['forward_camera_extrinsics'],
                        obs[i].misc['forward_camera_intrinsics'])
                if obs_config.top_camera.point_cloud:
                    obs[i].top_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        top_depth_m,
                        obs[i].misc['top_camera_extrinsics'],
                        obs[i].misc['top_camera_intrinsics'])
                if obs_config.back_camera.point_cloud:
                    obs[i].back_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        back_depth_m,
                        obs[i].misc['back_camera_extrinsics'],
                        obs[i].misc['back_camera_intrinsics'])
                if obs_config.left_camera.point_cloud:
                    obs[i].left_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        left_depth_m,
                        obs[i].misc['left_camera_extrinsics'],
                        obs[i].misc['left_camera_intrinsics'])
                if obs_config.right_camera.point_cloud:
                    obs[i].right_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        right_depth_m,
                        obs[i].misc['right_camera_extrinsics'],
                        obs[i].misc['right_camera_intrinsics'])





                # Masks are stored as coded RGB images.
                # Here we transform them into 1 channel handles.
                if obs_config.left_shoulder_camera.mask:
                    obs[i].left_shoulder_mask = rgb_handles_to_mask(
                        np.array(_resize_if_needed(Image.open(
                            obs[i].left_shoulder_mask),
                            obs_config.left_shoulder_camera.image_size)))
                if obs_config.right_shoulder_camera.mask:
                    obs[i].right_shoulder_mask = rgb_handles_to_mask(
                        np.array(_resize_if_needed(Image.open(
                            obs[i].right_shoulder_mask),
                            obs_config.right_shoulder_camera.image_size)))
                if obs_config.overhead_camera.mask:
                    obs[i].overhead_mask = rgb_handles_to_mask(
                        np.array(_resize_if_needed(Image.open(
                            obs[i].overhead_mask),
                            obs_config.overhead_camera.image_size)))
                if obs_config.wrist_camera.mask:
                    obs[i].wrist_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].wrist_mask),
                            obs_config.wrist_camera.image_size)))
                if obs_config.front_camera.mask:
                    obs[i].front_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].front_mask),
                            obs_config.front_camera.image_size)))
                if obs_config.forward_camera.mask:
                    obs[i].forward_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].forward_mask),
                            obs_config.forward_camera.image_size)))
                if obs_config.top_camera.mask:
                    obs[i].top_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].top_mask),
                            obs_config.top_camera.image_size)))
                if obs_config.back_camera.mask:
                    obs[i].back_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].back_mask),
                            obs_config.back_camera.image_size)))
                if obs_config.left_camera.mask:
                    obs[i].left_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].left_mask),
                            obs_config.left_camera.image_size)))
                if obs_config.right_camera.mask:
                    obs[i].right_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].right_mask),
                            obs_config.right_camera.image_size)))

        demos.append(obs)
    return demos


def _resize_if_needed(image, size):
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
    return image
