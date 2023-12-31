import pickle
import random

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import time
import pandas as pd
import sys
import json
import warnings
import nltk
import torch
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import mmdet
import mmcv
from mmdet.apis import inference_detector, init_detector
from skimage.draw import polygon2mask

random_seed = 42
np.random.seed(random_seed)

from utils.utils.import (
    compute_ssim,
    compute_mse,
    dice_coefficient,
    weighted_dice_coefficient,
    new_weighted_dice_coefficient,
    jaccard_similarity,
    apply_gaussian_blur,
    weighted_jaccard_index,
    get_tasks,
    new_labels)


class SBP:
    def __init__(self, input_folder, main_save_folder):
        self.input_folder = input_folder
        self.lang_ann = np.load(f'{self.input_folder}/lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy',
                                allow_pickle=True)
        self.lang_ann = self.lang_ann[()]

        self.envs_to_episode_index = {
            'A':[1191339, 1795044],
            'B':[0, 598909],
            'C': [598910, 1191338]
        }

        tasks =  self.lang_ann['language']['task']
        indices =  self.lang_ann['info']['indx']

        self.env_to_indices = {env: [] for env in self.envs_to_episode_index}
        target_percentage = 0.2  # or whatever you desire
        num_samples = int(target_percentage * len(indices))

        samples_per_env = num_samples // len(self.envs_to_episode_index)
        samples_per_task_per_env = samples_per_env // len(set(tasks))

        # Map each episode to a task and to an environment
        task_to_env_indices = {task: {env: [] for env in self.envs_to_episode_index} for task in set(tasks)}

        for idx, (start, end) in enumerate(indices):
            task = tasks[idx]
            for env, (env_start, env_end) in self.envs_to_episode_index.items():
                if env_start <= start <= env_end:
                    task_to_env_indices[task][env].append(idx)

        for task, env_indices_map in task_to_env_indices.items():
            for env, task_indices in env_indices_map.items():
                sampled_indices = random.sample(task_indices, min(samples_per_task_per_env, len(task_indices)))
                self.env_to_indices[env].extend(sampled_indices)

        self.episode_index_to_envs = {value: key for key, values in self.envs_to_episode_index.items() for value in values}
        self.main_folder = main_save_folder
        try:
            with open(f'./utils/object_segmentation_dict.txt', 'r') as f:
                self.object_dict = json.load(f)
            print("Successfully loaded object_dict")
        except FileNotFoundError:
            print("Error: File not found.")
        try:
            with open('./utils/task_object_mapping.txt', 'r') as f_mapping:
                self.task_object_mapping = json.load(f_mapping)
            print("Successfully loaded task_object_mapping_dict")
        except FileNotFoundError:
            print("Error: File2 not found.")

        self.eval_step = 0
        self.eval_counter = 0
        self.shape = (84, 84)
        self.shape_static = (200, 200)

        self.only_static = False

        self.reference_rgb_temp_gripper = None
        self.reference_mask_temp_gripper = None
        self.reference_vector_temp_gripper = None

        self.reference_rgb_temp_static = None
        self.reference_mask_temp_static = None
        self.reference_vector_temp_static = None

        self.best_results = []
        self.new_search = True
        self.shuffle_objects_list = []

    def decide_label_handles(self, candidates):

        result = {}
        iteration_count = 0
        while candidates:

            iteration_count += 1

            if iteration_count > len(candidates):
                break

            highest_aspect_ratio = max(candidates.values(), key=lambda x: x['aspect_ratio'])

            selected_candidate = highest_aspect_ratio

            key = selected_candidate['key']
            value = selected_candidate['aspect_ratio']

            if key:
                avg_color = candidates[key]['avg_color']
                white_pixel_count = candidates[key]['white_pixel_count']
                mask_size = candidates[key]['mask_size']

                if value > 1.1 and white_pixel_count < 100:
                    if 'lightbulb' not in result:
                        result['lightbulb'] = candidates[key]['object_mask']

                    del candidates[key]
                elif value <= 1.1 and white_pixel_count < 100:
                    if 'base__slide' not in result:
                        result['base__slide'] = candidates[key]['object_mask']
                    del candidates[key]

                elif white_pixel_count > 100 or value > 2.0:
                    if 'base__drawer' not in result:
                        result['base__drawer'] = candidates[key]['object_mask']
                    del candidates[key]

        return result

    def handels_new(self, image, task, object, object_info):

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        hsv_lower = np.array(object_info['hsv_lower'], dtype=np.uint8)
        hsv_upper = np.array(object_info['hsv_upper'], dtype=np.uint8)
        region = object_info['region']

        if self.eval_step >= 60:
            if task == 'turn_on_lightbulb':
                region = 'upper'
            elif task == 'turn_off_lightbulb':
                region = 'lower'

        object_mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
        object_mask = self.filter_mask_by_region(object_mask, region, image.shape)

        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            object_contour = max(contours, key=cv2.contourArea)

            # Extract the bounding box
            x, y, w, h = cv2.boundingRect(object_contour)
            center_x = x + w // 2
            center_y = y + h // 2

            target_w = object_info['target_w']
            target_h = object_info['target_h']

            bbox_x = center_x - target_w // 2
            bbox_y = center_y - target_h // 2

            bbox_x = max(bbox_x, 0)
            bbox_y = max(bbox_y, 0)
            bbox_x = min(bbox_x, image.shape[1] - target_w)
            bbox_y = min(bbox_y, image.shape[0] - target_h)

            x = bbox_x
            y = bbox_y
            w = target_w
            h = target_h

            object_mask = object_mask[y:y + h, x:x + w]
            object_mask_cropped = cv2.resize(object_mask, self.shape)
            object_mask_blurred = apply_gaussian_blur(object_mask_cropped)

            return {object: object_mask_blurred}
        else:
            return None



    def get_info_handles(self, image, task, object_info):
        candidates = {}

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        hsv_lower = np.array(object_info['hsv_lower'], dtype=np.uint8)
        hsv_upper = np.array(object_info['hsv_upper'], dtype=np.uint8)
        region = object_info['region']

        object_mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
        object_mask = self.filter_mask_by_region(object_mask, region, image.shape)

        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 20 #70
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
        for i, contour in enumerate(filtered_contours):

            obj_mask = np.zeros_like(image, dtype=np.uint8)
            obj_mask = cv2.drawContours(obj_mask, [contour], 0, (255, 255, 255), -1)

            lower_bound_y = min(contour[:, 0, 1])

            y_region = lower_bound_y
            region_mask = np.zeros_like(object_mask)
            region_mask[y_region:, :] = 255

            object_mask_in_region = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_mask = cv2.dilate(object_mask_in_region, kernel, iterations=1)
            eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)
            object_mask_in_region = cv2.bitwise_and(eroded_mask, region_mask)

            contours_in_region, _ = cv2.findContours(object_mask_in_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            object_contour = max(contours_in_region, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(object_contour)

            center_x = x + w // 2
            center_y = y + h // 2

            target_w = int(1.3 * w)
            target_h = int(1.3 * h)

            bbox_x = center_x - target_w // 2
            bbox_y = center_y - target_h // 2

            bbox_x = max(bbox_x, 0)
            bbox_y = max(bbox_y, 0)
            bbox_x = min(bbox_x, image.shape[1] - target_w)
            bbox_y = min(bbox_y, image.shape[0] - target_h)

            aspect_ratio = w / h

            white_lower = np.array([0, 0, 120])
            white_upper = np.array([180, 30, 255])
            white = cv2.inRange(hsv_image, white_lower, white_upper)

            white_cropped = white[y:y + h, x:x + w]
            mask_size = white_cropped.size

            num_nonzero_pixels = np.count_nonzero(white_cropped)
            avg_color = np.mean(white_cropped, axis=(0, 1))

            x = bbox_x
            y = bbox_y
            w = target_w
            h = target_h

            candidates[f'object_{i}'] = {
                'aspect_ratio': aspect_ratio,
                'avg_color': avg_color,
                'white_pixel_count': num_nonzero_pixels,
                'mask_size': mask_size,
                'bbox': (x, y, w, h),
                'object_mask':   object_mask_in_region.astype(np.uint8) / 255,
                'key': f'object_{i}'
            }

        return candidates



    def get_env_from_index(self, index):
        for env, index_range in self.envs_to_episode_index.items():
            if index_range[0] <= index <= index_range[1]:
                return env
        return None





    def create_similarity_grid(self, frames, similarities, envs, counter, task, traj, use_frames=False):

        num_frames = 9
        num_cols = 3
        num_rows = (num_frames + num_cols - 1) // num_cols


        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 12))

        # Set the first image as the reference mask
        if use_frames:
            resized_gripper_image = cv2.resize(self.reference_rgb_temp_gripper, self.shape_static)
            resized_static_image = cv2.resize(self.reference_rgb_temp_static, self.shape_static)
            axs[0, 0].imshow(np.hstack((resized_static_image, resized_gripper_image)))
            axs[0, 0].set_title('Reference')

        elif not use_frames:
            resized_gripper_image = cv2.resize(self.reference_mask_temp_gripper, self.shape_static)
            resized_static_image = cv2.resize(self.reference_mask_temp_static, self.shape_static)
            axs[0, 0].imshow(np.hstack((resized_static_image, resized_gripper_image)))
            axs[0, 0].set_title('Reference')

        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col - 1

                if 0 <= index < len(frames):
                    frame_static, frame_gripper = frames[index]
                    frame_gripper = cv2.resize(frame_gripper, (frame_static.shape[0], frame_static.shape[0]))
                    frame_static = cv2.resize(frame_static, (frame_static.shape[0], frame_static.shape[0]))
                    combined_frame = np.hstack((frame_static, frame_gripper))
                    axs[row, col].imshow(combined_frame)
                    axs[row, col].set_title(f'{similarities[index]:.3f}_{traj[index]}_{envs[index]}')
                elif row == 0 and col == 0:
                    continue
                else:
                    axs[row, col].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        if not use_frames:
            save_path = f'{self.main_folder}/{task}/masking/combined'
            img_path = os.path.join(save_path, f'similarity_grid_{counter}_{self.eval_step}.png')
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(img_path)
            plt.close()
        else:
            save_path = f'{self.main_folder}/{task}/masking_original/combined'
            img_path = os.path.join(save_path, f'similarity_grid_{counter}_{self.eval_step}.png')
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(img_path)
            plt.close()

    def get_frames_and_actions(self, task, step):

        if self.new_search:
            object = self.task_object_mapping[task]
            tasks_to_search = get_tasks(task, object) #all tasks from all env with same object shape
            task_indices = [
                indx for env, task_name in tasks_to_search
                if env in self.env_to_indices
                for indx in self.env_to_indices[env]
                if self.lang_ann['language']['task'][indx] == task_name
            ]

        else:
            task_indices = self.best_results

        data = []


        for index in task_indices:
            start, end = self.lang_ann['info']['indx'][index]
            if start is None or end is None:
                continue
            if end < start:
                start, end = end, start
            traj_length = end - start
            try:
                for i in range(start, end + 1):
                    if i == start + step:
                        episode_index = str(i).zfill(7)
                        episode_path = os.path.join(self.input_folder, f"episode_{episode_index}.npz")
                        frame_static, frame_gripper, depth_gripper, action, rel_actions = self.load_episode(episode_path)
                        env = self.get_env_from_index(i)
                        data.append((frame_static, frame_gripper, depth_gripper, action, rel_actions, traj_length, index, env))
            except:
                continue

        frames = [(frame_static, frame_gripper, depth_gripper) for frame_static, frame_gripper, depth_gripper, _, _, _, _,_ in data]
        abs_actions_list = [action for _, _, _, action, _, _, _, _ in data]
        rel_actions_list = [rel_actions for _, _, _, _, rel_actions, _, _, _ in data]
        indices = [index for _, _, _, _, _, _, index, _ in data]
        trajectory_length = [traj_length for _, _, _, _, _, traj_length, _, _ in data]
        all_envs = [env for _, _, _, _, _, _, _, env in data ]

        return frames, abs_actions_list, rel_actions_list, indices, trajectory_length, all_envs

    def load_episode(self, episode_path):
        episode = np.load(episode_path)
        frame_static = episode['rgb_static']
        frame_gripper = episode['rgb_gripper']
        depth_gripper = episode['depth_gripper']
        action = episode['actions']
        rel_actions = episode['rel_actions']
        episode.close()
        return frame_static, frame_gripper, depth_gripper, action, rel_actions

    def load_all_frames(self, top_similar_indices, step):

        data = []

        for index in top_similar_indices:
            start, end = self.lang_ann['info']['indx'][index]
            if end < start:
                start, end = end, start
            for i in range(start + step, end + 1):
                episode_index = str(i).zfill(7)
                episode_path = os.path.join(self.input_folder, f"episode_{episode_index}.npz")
                frame_static, frame_gripper, depth_gripper, abs_actions, rel_actions = self.load_episode(episode_path)
                data.append((frame_static, frame_gripper, depth_gripper, abs_actions, rel_actions))

        frames_static = [frame_static for frame_static, _, _, _, _ in data]
        frames_gripper = [frame_gripper for _, frame_gripper, _, _, _ in data]
        depths_gripper = [depth_gripper for _, _, depth_gripper, _, _ in data]
        abs_actions_list = [abs_action for _, _, _, abs_action, _ in data]
        rel_actions_list = [rel_actions for _, _, _, _, rel_actions in data]

        return frames_static, frames_gripper, depths_gripper, abs_actions_list, rel_actions_list

    def create_latent_vector_cropped(self, image, object_info, task=None):

        object_mask, _ = self.get_bbox(image, object_info, task)
        if object_mask is None:
            object_mask = np.zeros(self.shape, dtype=np.uint8)
            return object_mask, object_mask.flatten()

        object_mask_blurred = apply_gaussian_blur(object_mask)
        flattened_vector = object_mask_blurred.flatten()
        flattened_vector = flattened_vector / 255
        return object_mask_blurred, flattened_vector

    def filter_mask_by_region(self, mask, region, image_shape):
        width, height, _ = image_shape
        if region == 'lower':
            lower_half_mask = np.zeros_like(mask)
            lower_half_mask[int(height / 2.5):, :] = 255
            mask = cv2.bitwise_and(mask, lower_half_mask)

        elif region == 'upper':
            upper_half_mask = np.zeros_like(mask)
            upper_half_mask[:int(height / 2), :] = 255
            mask = cv2.bitwise_and(mask, upper_half_mask)

        elif region == 'mid_left':
            mid_left_mask = np.zeros_like(mask)
            mid_left_mask[:int(height / 2.2), :int(width / 1.45)] = 255
            mask = cv2.bitwise_and(mask, mid_left_mask)

        elif region == 'mid_right':
            mid_right_mask = np.zeros_like(mask)
            mid_right_mask[:int(height / 2.2), int(width / 2.8):] = 255
            mask = cv2.bitwise_and(mask, mid_right_mask)

        elif region == 'right':
            right_mask = np.zeros_like(mask)
            right_mask[:int(height / 2.2), int(width / 1.40):] = 255
            mask = cv2.bitwise_and(mask, right_mask)
        elif region == 'left':
            left_mask = np.zeros_like(mask)
            left_mask[:int(height / 2.2), :int(width / 3.0):] = 255
            mask = cv2.bitwise_and(mask, left_mask)
        elif region == 'mid_upper':
            mid_upper = np.zeros_like(mask)
            mid_upper[:int(height / 2.0), :int(width / 1)] = 255
            mask = cv2.bitwise_and(mask, mid_upper)
        return mask

    def get_bbox(self, image, object_info, task=None):

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        hsv_lower = np.array(object_info['hsv_lower'], dtype=np.uint8)
        hsv_upper = np.array(object_info['hsv_upper'], dtype=np.uint8)
        region = object_info['region']

        if task in ['push_into_drawer', 'unstack_block']:
            region = 'mid_upper'

        object_mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
        object_mask = self.filter_mask_by_region(object_mask, region, image.shape)

        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            object_contour = max(contours, key=cv2.contourArea)
            if object_contour is not None:
                largest_contour_area = cv2.contourArea(object_contour)
            else:
                largest_contour_area = 0.0
            return object_mask,  largest_contour_area
        else:
            return None, 0.0

    def add_switch_text(self, frame, text, gripper):
        if gripper:
            frame = cv2.putText(frame, text, (frame.shape[1] - 80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            frame = cv2.putText(frame, text, (frame.shape[1] - 180, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        return frame


    def find_best_time_step(self, task, counter, alpha, metric, all_trajectories, start_step, mse=False):

        highest_similarity = 0
        lowest_similarity = np.inf
        save_folder = f'{self.main_folder}/{task}'
        start_step = start_step
        maxbound = 65
        s = 4
        highest_trajectory_similarities = []
        highest_trajectory_masks = []
        highest_trajectory_frames = []
        highest_trajectory_frames_cropped = []

        for step in range(start_step, maxbound, s):
            try:
                frames, abs_actions, _, task_indices, trajectory_lengths, all_envs = self.get_frames_and_actions(task, step)
            except:
                continue

            similarities = []
            all_masks = []
            all_frames = []
            all_cropped_frames = []

            # iterate through all trajectories
            for i, f in enumerate(frames):

                frame_static = f[0]
                frame_gripper = f[1]
                env = all_envs[i]


                similarity_static, object_mask_static = self.calculate_similarity(frame_static, task, env, metric, False)

                if not self.only_static:
                    similarity_gripper, object_mask_gripper = self.calculate_similarity(frame_gripper, task,  env, metric, True)
                else:
                    #print("no gripper because only static")
                    similarity_gripper = 0.0
                    object_mask_gripper = np.zeros(self.shape, dtype=np.uint8)


                weighted_similarity = np.round(alpha * similarity_gripper + (1 - alpha) * similarity_static, 3)



                trajectory_index = task_indices[i]
                similarities.append((weighted_similarity, trajectory_index, i, step, env))
                all_masks.append((i, trajectory_index, (object_mask_static, object_mask_gripper)))
                all_frames.append((i, trajectory_index, (frame_static, frame_gripper)))
                all_cropped_frames.append((i, trajectory_index, (frame_static, frame_gripper)))

            if similarities:

                for similarity in similarities:

                    similarity_score = similarity[0]
                    trajectory_index = similarity[1]
                    step_index = similarity[2]

                    index_found = False
                    for i, (score, index, _, _, _) in enumerate(highest_trajectory_similarities):
                        if index == trajectory_index:
                            index_found = True
                            if not mse:
                                if similarity_score > score:
                                    highest_trajectory_similarities[i] = similarity
                                    highest_trajectory_masks[i] = [m for m in all_masks if m[0] == step_index][0]
                                    highest_trajectory_frames[i] = [f for f in all_frames if f[0] == step_index][0]
                                    highest_trajectory_frames_cropped[i] = [f for f in all_cropped_frames if f[0] == step_index][0]

                                break
                            else:
                                if similarity_score < score:
                                    highest_trajectory_similarities[i] = similarity
                                    highest_trajectory_masks[i] = [m for m in all_masks if m[0] == step_index][0]
                                    highest_trajectory_frames[i] = [f for f in all_frames if f[0] == step_index][0]
                                    highest_trajectory_frames_cropped[i] = [f for f in all_cropped_frames if f[0] == step_index][0]

                                break

                    if not index_found:
                        highest_trajectory_similarities.append(similarity)
                        highest_trajectory_masks.append([m for m in all_masks if m[0] == step_index][0])
                        highest_trajectory_frames.append([f for f in all_frames if f[0] == step_index][0])
                        highest_trajectory_frames_cropped.append([f for f in all_cropped_frames if f[0] == step_index][0])

        if not mse:
            sorted_highest_trajectory_similarities = sorted(highest_trajectory_similarities, key=lambda x: x[0],
                                                            reverse=True)
        else:
            sorted_highest_trajectory_similarities = sorted(highest_trajectory_similarities, key=lambda x: x[0])

        index_to_position = {item[1]: idx for idx, item in enumerate(sorted_highest_trajectory_similarities)}

        sorted_highest_trajectory_cropped_frames = sorted(highest_trajectory_frames_cropped,
                                                          key=lambda x: index_to_position[x[1]])
        sorted_highest_trajectory_masks = sorted(highest_trajectory_masks, key=lambda x: index_to_position[x[1]])

        top_res = sorted_highest_trajectory_similarities[:15]

        b_similarities = [similarity[0] for similarity in top_res]

        b_traj = [similarity[1] for similarity in top_res]
        b_envs = [similarity[4] for similarity in top_res]

        b_frames_cropped = [frame[2] for frame in sorted_highest_trajectory_cropped_frames if any(frame[0] == tuple_item[2] for tuple_item in top_res)]

        b_masks = [frame[2] for frame in sorted_highest_trajectory_masks if any(frame[0] == tuple_item[2] for tuple_item in top_res)]

        self.create_similarity_grid(frames=b_masks, similarities=b_similarities, counter=counter, task=task,
                                    traj=b_traj,envs=b_envs)
        self.create_similarity_grid(frames=b_frames_cropped, similarities=b_similarities, counter=counter,
                                    task=task, traj=b_traj, envs=b_envs, use_frames=True)

        if b_masks:
            best_mask = b_masks[0]
            best_rgb = b_frames_cropped[0]
            best_mask = self.prep_best_img(best_mask, b_traj[0], b_similarities[0], b_envs[0], mask=True)
            best_rgb = self.prep_best_img(best_rgb, b_traj[0], b_similarities[0], b_envs[0], mask=False)
        else:
            best_mask = None
            best_rgb = None

        return sorted_highest_trajectory_similarities[:3], best_mask, best_rgb


    def save_best_img_new(self, images, task, mask=False):

        save_folder = f'{self.main_folder}/{task}'

        if mask:
            save_folder = os.path.join(save_folder, 'best_match_masked')
        else:
            save_folder = os.path.join(save_folder, 'best_match_rgb')
        os.makedirs(save_folder, exist_ok=True)

        image_path_static = os.path.join(save_folder,
                                         f'best_mask_{task}_gripper_False_{self.eval_counter}_{self.eval_step}.png')
        image_path_gripper = os.path.join(save_folder,
                                          f'best_mask_{task}_gripper_True_{self.eval_counter}_{self.eval_step}.png')

        for i, (image_static, image_gripper) in enumerate(images):
            image_path_static = os.path.join(save_folder,
                                             f'best_mask_{task}_gripper_False_{self.eval_counter}_{i}.png')
            image_path_gripper = os.path.join(save_folder,
                                              f'best_mask_{task}_gripper_True_{self.eval_counter}_{i}.png')

            cv2.imwrite(image_path_static, image_static)
            cv2.imwrite(image_path_gripper, image_gripper)

    def prep_best_img(self, best, traj, sim, env, mask=False):

        best_static = best[0]
        best_gripper = best[1]

        if not mask:
            best_static = cv2.cvtColor(best_static, cv2.COLOR_BGR2RGB)
            best_gripper = cv2.cvtColor(best_gripper, cv2.COLOR_BGR2RGB)
            text_color = (0, 0, 0)  # Black color
        else:
            text_color = (255, 255, 255)  # White color
            #best_gripper = (best_gripper * 255).astype(np.uint8)

        text_position_static = (best_static.shape[1] - 150, 140)
        text_position_gripper = (best_gripper.shape[1] - 84, 74)

        #best_static = cv2.putText(best_static, f'{self.eval_step}_{traj}_{sim:.3f}_{env}', text_position_static,
        #                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        #best_gripper = cv2.putText(best_gripper, f'{self.eval_step}_{traj}_{sim:.3f}_{env}', text_position_gripper,
        #                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)

        return best_static, best_gripper

    def save_reference_img(self, static, gripper, task, traj, step, mask):


        save_folder = f'{self.main_folder}/{task}'

        if mask:
            save_folder = os.path.join(save_folder, 'reference_masked')
        else:
            save_folder = os.path.join(save_folder, 'reference_rgb')
        os.makedirs(save_folder, exist_ok=True)

        image_path_static = os.path.join(save_folder, f'ref_mask_{task}_gripper_False_{self.eval_counter}_{step}.png')
        image_path_gripper = os.path.join(save_folder, f'ref_mask_{task}_gripper_True_{self.eval_counter}_{step}.png')


        if not mask:
            static = cv2.cvtColor(static, cv2.COLOR_BGR2RGB)
            gripper = cv2.cvtColor(gripper, cv2.COLOR_BGR2RGB)
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)
            #gripper = (gripper * 255).astype(np.uint8)


        #text_position_static = (static.shape[1] - 200, 190)
        #text_position_gripper = (gripper.shape[1] - 84,74)

        static_with_text = static.copy()
        gripper_with_text = gripper.copy()

        #best_static = cv2.putText(static_with_text, f'{traj}', text_position_static, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        #best_gripper = cv2.putText(gripper_with_text, f'{traj}', text_position_gripper, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color,1)


        cv2.imwrite(image_path_static, static)
        cv2.imwrite(image_path_gripper, gripper)


    def save_trajectory_img(self, static, gripper, task, traj, step, current_traj_sim, env, mask):
        save_folder = f'{self.main_folder}/{task}'

        if mask:
            save_folder = os.path.join(save_folder, 'trajectory_images_mask')
        else:
            save_folder = os.path.join(save_folder, 'trajectory_images_rgb')
        os.makedirs(save_folder, exist_ok=True)

        image_path_static = os.path.join(save_folder, f'{self.eval_counter}_step_{step}_static.png')
        image_path_gripper = os.path.join(save_folder, f'{self.eval_counter}_step_{step}_gripper.png')

        if not mask:
            static = cv2.cvtColor(static, cv2.COLOR_BGR2RGB)
            gripper = cv2.cvtColor(gripper, cv2.COLOR_BGR2RGB)
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)
            #gripper = (gripper * 255).astype(np.uint8)

        text_position_static = (static.shape[1] - 200, 190)
        text_position_gripper = (gripper.shape[1] - 84, 74)

        static = cv2.putText(static, f'{step}_{traj}_{current_traj_sim}_{env}', text_position_static,
                             cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                             text_color, 1)
        gripper = cv2.putText(gripper, f'{step}_{traj}_{current_traj_sim}_{env}', text_position_gripper,
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.2, text_color, 1)

        cv2.imwrite(image_path_static, static)
        cv2.imwrite(image_path_gripper, gripper)

    def calculate_similarity(self, frame, task, env, similarity_metric, gripper):

        object = self.task_object_mapping[task]
        new_object_label = new_labels(env, object)
        object = object if new_object_label is None else new_object_label
        if object == 'all':
            object = self.shuffle_objects_list[-1]
            new_object_label = new_labels(env, object)
            object = object if new_object_label is None else new_object_label

        object_detected, res = self.detect_object_in_scene(object, frame, env, gripper, task = task)
        if not object_detected:
            object_mask = np.zeros(self.shape, dtype=np.uint8)
            return 0.0, object_mask


        else:
            camera = 'gripper' if gripper else 'static'
            object_info = self.object_dict[env][camera][object]
            if gripper:
                task = None
            object_mask, object_vector = self.create_latent_vector_cropped(frame, object_info, task)

        if gripper:
            reference_vector = self.reference_vector_temp_gripper
            reference_mask = self.reference_mask_temp_gripper
        elif not gripper:
            reference_vector = self.reference_vector_temp_static
            reference_mask = self.reference_mask_temp_static

        if similarity_metric == "weighted_dice":
            similarity, dice_coef, size_coef = weighted_dice_coefficient(reference_vector, object_vector)
        elif similarity_metric == "iou":
            similarity = jaccard_similarity(reference_vector, object_vector)
        elif similarity_metric == 'weighted_iou':
            similarity = weighted_jaccard_index(reference_vector, object_vector)
        elif similarity_metric == "mse":
            similarity = compute_mse(reference_vector, object_vector)
        elif similarity_metric == "ssim":
            similarity = compute_ssim(reference_vector, object_vector)
        elif similarity_metric == "dice":
            similarity = dice_coefficient(reference_vector, object_vector)
        elif similarity_metric == "new_weighted_dice":
            similarity = new_weighted_dice_coefficient(reference_vector, object_vector)
        else:
            print("WRONG SIMILARITY")

        return similarity, object_mask

    def detect_object_in_scene(self, objects, image, env, gripper, task=None):
        camera = 'gripper' if gripper else 'static'
        object_dict = self.object_dict[env][camera]

        if isinstance(objects, list):
            largest_obj = None
            largest_obj_size = 0.0

            for obj in objects:
                if obj in object_dict:
                    object_info = object_dict[obj]
                    if gripper:
                        task = None
                    object_mask, size = self.get_bbox(image, object_info, task)

                    if largest_obj is None or size > largest_obj_size:
                        largest_obj = obj
                        largest_obj_size = size

            return largest_obj is not None, largest_obj

        elif objects in object_dict:
            object_info = object_dict[objects]
            if gripper:
                task = None
            object_mask, _ = self.get_bbox(image, object_info, task)
            return object_mask is not None, None

        return False, None

    def get_mask(self, image, task, env, gripper):


        object = self.task_object_mapping[task]
        new_object_label = new_labels(env, object)
        object = object if new_object_label is None else new_object_label
        if object == 'all':
            object = self.shuffle_objects_list[-1]
        object_detected, res = self.detect_object_in_scene(object, image, env, gripper, task)
        if not object_detected:
            return np.zeros(self.shape, dtype=np.uint8)

        else:
            camera = 'gripper' if gripper else 'static'
            object_info = self.object_dict[env][camera][object]
            object_mask, _ = self.create_latent_vector_cropped(image, object_info, task)

        return object_mask

    def compare_trajectory_with_live(self, task, img_static, img_gripper, trajectory_static, trajectory_gripper,
                                     trajectory_number, step, alpha, similarity_metric, env):

        ref_object = self.task_object_mapping[task] #in env D
        if ref_object == 'all':
            ref_object = self.shuffle_objects_list[-1]

        new_object_label = new_labels(env, ref_object)

        object = ref_object if new_object_label is None else new_object_label


        object_detected_gripper, _ = self.detect_object_in_scene(ref_object, img_gripper, 'D',True)
        if not object_detected_gripper:
            object_mask_gripper = np.zeros(self.shape, dtype=np.uint8)
            object_vector_gripper = object_mask_gripper.flatten()
        else:
            object_info = self.object_dict['D']['gripper'][ref_object]
            object_mask_gripper, object_vector_gripper = self.create_latent_vector_cropped(
                img_gripper, object_info)

        object_detected_static, _ = self.detect_object_in_scene(ref_object, img_static, 'D', False, task)
        if not object_detected_static:
            object_mask_static = np.zeros(self.shape, dtype=np.uint8)
            object_vector_static = object_mask_static.flatten()

        else:
            object_info = self.object_dict['D']['static'][ref_object]
            object_mask_static, object_vector_static = self.create_latent_vector_cropped(
                img_static, object_info, task)

        object_detected_traj_gripper, res = self.detect_object_in_scene(object, trajectory_gripper, env,True)
        if not object_detected_traj_gripper:
            #print("trajectory gripper empty")

            object_mask_trajectory_gripper = np.zeros(self.shape, dtype=np.uint8)
            object_vector_trajectory_gripper = object_mask_trajectory_gripper.flatten()
        else:
            object_info = self.object_dict[env]['gripper'][object]
            object_mask_trajectory_gripper, object_vector_trajectory_gripper = self.create_latent_vector_cropped(
                trajectory_gripper, object_info)


        object_detected_traj_static, _ = self.detect_object_in_scene(object, trajectory_static, env,False, task)
        if not object_detected_traj_static :
            object_mask_trajectory_static = np.zeros(self.shape, dtype=np.uint8)
            object_vector_trajectory_static = object_mask_trajectory_static.flatten()
        else:
            object_info = self.object_dict[env]['static'][object]
            object_mask_trajectory_static, object_vector_trajectory_static = self.create_latent_vector_cropped(
                trajectory_static, object_info, task)

        if similarity_metric == "weighted_dice":
            similarity_static, dice_static, size_coef_static = weighted_dice_coefficient(object_vector_static, object_vector_trajectory_static)
            print(f'static: {similarity_static}')
            similarity_gripper, dice_gripper, size_coef_gripper = weighted_dice_coefficient(object_vector_gripper, object_vector_trajectory_gripper)
            print(f'gripper: {similarity_gripper}')
        elif similarity_metric == "new_weighted_dice":
            similarity_static = new_weighted_dice_coefficient(object_vector_static, object_vector_trajectory_static)
            similarity_gripper = new_weighted_dice_coefficient(object_vector_gripper, object_vector_trajectory_gripper)
        elif similarity_metric == 'weighted_iou':
            similarity_static = weighted_jaccard_index(object_vector_static, object_vector_trajectory_static)
            similarity_gripper = weighted_jaccard_index(object_vector_gripper, object_vector_trajectory_gripper)

        current_traj_sim = np.round(alpha * similarity_gripper + (1 - alpha) * similarity_static, 3)
        print("current traj sim", current_traj_sim)

        self.save_trajectory_img(trajectory_static, trajectory_gripper, task, trajectory_number,
                                 step, current_traj_sim, env, False)
        self.save_trajectory_img(object_mask_trajectory_static, object_mask_trajectory_gripper, task, trajectory_number,
                                 step, current_traj_sim, env, True)

        self.save_reference_img(img_static, img_gripper, task, trajectory_number, step, mask=False)
        self.save_reference_img(object_mask_static, object_mask_gripper, task, trajectory_number, step, mask=True)

        return current_traj_sim



    def run_find_best(self, task, img_static, img_gripper, eval_counter, step, alpha, metric, all_trajectories, new_search, start_step, new_shuffle = False):

        print("run find best")
        print("new search", new_search)

        self.new_search = new_search
        self.eval_step = step
        self.eval_counter = eval_counter


        env = 'D'
        save_folder = f'{self.main_folder}/{task}'

        object = self.task_object_mapping[task]
        if object == 'all' and new_shuffle:
            object_detected, object = self.detect_object_in_scene(['block_blue', 'block_pink', 'block_red'], img_gripper, env, gripper=True, task=task)
            if object_detected:
                self.shuffle_objects_list.append(object)
            else:
                obj_detected = []
                for obj in ['block_blue', 'block_pink', 'block_red']:
                    if obj not in self.shuffle_objects_list:
                        object_detected, _ = self.detect_object_in_scene(obj, img_static, env, gripper=False, task=task)
                        if object_detected:
                            obj_detected.append(obj)
                if obj_detected:
                    object = np.random.choice(obj_detected)
                    self.shuffle_objects_list.append(object)

        elif object == 'all' and not new_shuffle:
            object = self.shuffle_objects_list[-1]


        object_detected, _ = self.detect_object_in_scene(object, img_static, env, gripper=False, task=task)

        if not object_detected:
            self.reference_rgb_temp_static = cv2.resize(img_gripper, self.shape_static)
            self.reference_mask_temp_static = np.zeros(self.shape_static, dtype=np.uint8)
            self.reference_vector_temp_static = self.reference_mask_temp_static.flatten()
        else:
            camera = 'static'
            object_info = self.object_dict[env][camera][object]
            object_mask, object_vector = self.create_latent_vector_cropped(img_static, object_info, task)
            self.reference_rgb_temp_static = img_static
            self.reference_mask_temp_static = object_mask
            self.reference_vector_temp_static = object_vector

        object_detected, res = self.detect_object_in_scene(object, img_gripper, env, True)
        if not object_detected:
            self.only_static = True
            self.reference_rgb_temp_gripper = cv2.resize(img_gripper, (84, 84))
            self.reference_mask_temp_gripper = np.zeros(self.shape, dtype=np.uint8)
            self.reference_vector_temp_gripper = self.reference_mask_temp_gripper.flatten()

        else:
            self.only_static = False
            camera = 'gripper'
            object_info = self.object_dict[env][camera][object]
            object_mask, object_vector = self.create_latent_vector_cropped(img_gripper, object_info)

            self.reference_rgb_temp_gripper = img_gripper
            self.reference_mask_temp_gripper = object_mask
            self.reference_vector_temp_gripper = object_vector


        find_best_time_step_results, best_mask, best_rgb = self.find_best_time_step(
            task, eval_counter, alpha, metric, all_trajectories, start_step)

        return find_best_time_step_results, best_mask, best_rgb

    def mask_best_match_trajectory(self, images, task, env, gripper):

        masked_images = []
        for image in images:
            image_masked = self.get_mask(image, task, env, gripper)
            masked_images.append(image_masked)

        return masked_images

    def run_pipeline(self, task, counter, eval_step, trajectory, img_step, env):

        self.eval_step = eval_step
        self.eval_counter = counter
        trajectory = [trajectory]

        if eval_step > 0:
            top_step = img_step
        else:
            top_step = 0

        top_frames_static, top_frames_gripper, top_depth_frames, top_abs_actions, top_rel_actions = self.load_all_frames(
            trajectory, step=top_step)

        top_frames_static_masked = self.mask_best_match_trajectory(top_frames_static, task, env, False)

        top_frames_gripper_masked = self.mask_best_match_trajectory(top_frames_gripper, task, env, True)

        return top_abs_actions, top_rel_actions, top_frames_static, top_frames_gripper, top_frames_static_masked, top_frames_gripper_masked, self.only_static

