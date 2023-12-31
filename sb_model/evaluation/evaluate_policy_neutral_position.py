import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import random
import cv2
from calvin_agent.evaluation.multistep_sequences import get_sequences

from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import pandas as pd
from calvin_env.envs.play_table_env import get_env
from PIL import Image
import pickle

from search_policy import SBP
from utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    generate_sorted_multi_task,
    generate_MTLC_eval,
    load_data,
    join_vis_lang_old,
    transform_data_classifier
)

from clasification_model_new import GraspClassifier
from dataset_new import GraspDataModule

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

save_results_path = 'xxx'
log_path = f'{save_results_path}/hulc_eval.log'

if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)

logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


logger = logging.getLogger('hulc_evaluation_logger')

logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)


ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

classification_checkpoint_path = './utils'
ckpt_files = [file for file in os.listdir(classification_checkpoint_path) if file.endswith('.ckpt')]
model_save_path = os.path.join(classification_checkpoint_path, ckpt_files[0])
classification_model = GraspClassifier.load_from_checkpoint(model_save_path, num_classes=2, class_weights=[11.5709,  1.0946])
classification_model.eval()


model = SBP()
EP_LEN = 180
NUM_SEQUENCES = 1000


# Initialize an empty dataset
data = {
    'task': [],
    'eval_counter': [],
    'steps': [],
    'success': []
}

dataset = pd.DataFrame(data)
def add_data(task, eval_counter, steps, success):
    global dataset
    new_data = pd.DataFrame({
        'task': [task],
        'eval_counter': [eval_counter],
        'steps': [steps],
        'success': [success]
    })
    dataset = pd.concat([dataset, new_data], ignore_index=True)

def evaluate_policy(env, epoch, eval_log_dir=None, debug=False):


    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    #logger.debug('tasks distribution in eval_sequeances: %s', task_counts)

    results = []

    eval_counter = 0
    score = 0
    for initial_state, eval_sequence in eval_sequences:
        print(f"EVAL_COUNTER:{eval_counter}/{len(eval_sequences)}")
        print("INIT STATE", initial_state)
        print("EVALUATE", eval_sequence)
        result = evaluate_sequence(env, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_counter)
        print(result)
        eval_counter += 1
        results.append(result)
        print(f"SCORE: {score / eval_counter:.3f}")
    dataset.to_csv(f"{save_results_path}/results.csv", index=False)
    return results


def evaluate_sequence(env, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_counter, do_hack = False):

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0

    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")

    for subtask in eval_sequence:

        success = rollout(env, task_checker, subtask, val_annotations, plans, debug, eval_counter)
        if success:
            success_counter += 1
        else:
            return success_counter
        return success_counter


def rollout(env, task_oracle, subtask, val_annotations, plans, debug, eval_counter, create_video=True):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    obs = env.get_obs()
    # get lang annotation for subtask
    logger.info(f'Get language annotation for subtask: {subtask}')
    print("NEW EVAL with subtask", subtask)
    #val_annotation is a predefined yaml with the subtask.
    lang_annotation = val_annotations[subtask][0]
    logger.debug('language_annotation: %s', lang_annotation)

    device = torch.device('cuda')
    torch.cuda.empty_cache()

    start_info = env.get_info()


    output_folder = f'{save_results_path}/evaluation'
    subtask_folder = os.path.join(output_folder, subtask)
    success_folder = os.path.join(subtask_folder, "success")
    fail_folder = os.path.join(subtask_folder, "fail")
    os.makedirs(success_folder, exist_ok=True)
    os.makedirs(fail_folder, exist_ok=True)


    video_frames_static_rgb = []
    video_frames_gripper_rgb = []
    video_frames_static_mask = []
    video_frames_gripper_mask = []

    best_masks = []
    best_rgbs = []

    trajectory_similarities = []
    action_index = 0
    actions = []
    only_closed_list = []

    alpha = 0.9
    current_trajectory_sim = 0.0
    current_traj = None
    current_step = None
    current_env = None
    amount_switched = 0
    change = 0
    ooa = False
    metric = 'weighted_dice'
    only_closed = False
    switch = True
    switch_counter = 0
    new_search = False
    shuffle = True

    for step in range(EP_LEN):
        print("STEP", step)

        #TODO: Build logic to detect closes block to robot arm instead of random block
        if subtask in ['place_in_slider', 'place_in_drawer', 'push_into_drawer', 'unstack_block', 'stack_block']:
            if step == 0:
                shuffle = True
            else:
                shuffle = False

        img, img_gripper = env.render(mode="rgb_array")

        do_switch = False

        if len(trajectory_similarities) >= 2:
            std_curr_traj_sim = np.round(np.std(trajectory_similarities[-2:]), 3)

            if trajectory_similarities[-1] < 0.2 and trajectory_similarities[-2] < 0.2:
                threshold = 0.003
            else:
                threshold = 0.03

        else:
            std_curr_traj_sim = 0
            threshold = 0.003

        #if there is contact/grip with object, dont open gripper when switching to a trajectory where gripper would be open
        if len(only_closed_list) >= 3 and all(only_closed_list[-3:]):
            switch = False

        if std_curr_traj_sim >= threshold or action_index >= len(actions) or new_search:

            if current_traj is None:

                best_trajectories, new_best_mask, new_best_rgb = model.run_find_best(
                    subtask, img, img_gripper, eval_counter, step, alpha, metric, True, new_search=True, start_step=0,
                    new_shuffle=shuffle)

                best_result = best_trajectories[0]
                new_sim = best_result[0]
                new_traj = best_result[1]
                new_step = best_result[3]
                new_env = best_result[4]

                abs_actions, rel_actions, top_frames_static, top_frames_gripper, top_frames_static_masked, top_frames_gripper_masked, only_static = model.run_pipeline(
                    subtask, eval_counter, step, new_traj, new_step, new_env)

                only_static = only_static or (new_sim < 0.3 and not only_static and step <= 40)

                current_traj = new_traj
                current_step = new_step
                current_env = new_env

                if only_static:
                    relative_action = False
                    abs_actions = np.array(abs_actions, dtype=np.float32)
                    actions = torch.tensor(abs_actions).to(device)
                else:
                    relative_action = True
                    rel_actions = np.array(rel_actions, dtype=np.float32)
                    actions = torch.tensor(rel_actions).to(device)

                action_index = 0

            elif current_traj is not None or action_index >= len(actions):

                start_step = max(0, new_step - 16)
                best_trajectories, new_best_mask, new_best_rgb = model.run_find_best(
                    subtask, img, img_gripper, eval_counter, step, alpha, metric, True, new_search=True,
                    start_step=start_step, new_shuffle=shuffle)

                best_result = best_trajectories[0]
                new_sim = best_result[0]
                new_traj = best_result[1]
                new_step = best_result[3]
                new_env = best_result[4]

                if action_index >= len(actions) and switch_counter < 2:
                    #out of actions in prusued trajectory
                    ooa = True

                    if not switch:
                        switch_counter += 1

                    trajectory_similarities = []
                    amount_switched += 1
                    do_switch = True

                    current_sim = new_sim
                    current_traj = new_traj
                    new_step = new_step if switch_counter == 0 else 0
                    current_step = new_step
                    current_env = new_env

                    abs_actions, rel_actions, top_frames_static, top_frames_gripper, top_frames_static_masked, top_frames_gripper_masked, only_static = model.run_pipeline(
                        subtask, eval_counter, step, new_traj, new_step, new_env)
                    only_static = only_static or (new_sim < 0.3 and not only_static)
                    if only_static:
                        relative_action = False
                        abs_actions = np.array(abs_actions, dtype=np.float32)
                        actions = torch.tensor(abs_actions).to(device)
                    else:
                        relative_action = True
                        rel_actions = np.array(rel_actions, dtype=np.float32)
                        actions = torch.tensor(rel_actions).to(device)

                    if only_closed:
                        actions[:, -1] = 0.0

                    action_index = 0

                elif new_sim > current_trajectory_sim and switch:
                    ooa = False

                    trajectory_similarities = []
                    amount_switched += 1
                    do_switch = True

                    current_traj = new_traj
                    current_step = new_step
                    current_env = new_env

                    abs_actions, rel_actions, top_frames_static, top_frames_gripper, top_frames_static_masked, top_frames_gripper_masked, only_static = model.run_pipeline(
                        subtask, eval_counter, step, new_traj, new_step, new_env)
                    only_static = only_static or (new_sim < 0.3 and not only_static)
                    if only_static:
                        relative_action = False
                        abs_actions = np.array(abs_actions, dtype=np.float32)
                        actions = torch.tensor(abs_actions).to(device)
                    else:
                        relative_action = True
                        rel_actions = np.array(rel_actions, dtype=np.float32)
                        actions = torch.tensor(rel_actions).to(device)

                    if only_closed:
                        actions[:, -1] = 0.0

                    action_index = 0

        if switch_counter > 1 and action_index >= len(actions):
            return False

        best_masks.append(new_best_mask)
        best_rgbs.append(new_best_rgb)

        current_traj_static_rgb = top_frames_static[action_index]
        current_traj_gripper_rgb = top_frames_gripper[action_index]

        current_trajectory_sim = model.compare_trajectory_with_live(subtask, img, img_gripper,
                                                                    current_traj_static_rgb,
                                                                    current_traj_gripper_rgb,
                                                                    current_traj, step, alpha, metric, current_env)

        trajectory_similarities.append(current_trajectory_sim)

        action = actions[action_index]

        obs, _, _, current_info, applied_action = env.step(action, relative_action)

        if debug:
            img, _ = env.render(mode="rgb_array")
            join_vis_lang_old(img, lang_annotation, step)

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        _, img_gripper_class = env.render('rgb_array')
        pil_image = Image.fromarray(np.uint8(img_gripper_class))
        input_tensor = transform_data_classifier(pil_image)
        predicted_label, predicted_probabilities = classification_model.predict(input_tensor)
        if subtask not in ['place_in_slider', 'place_in_drawer', 'unstack_block', 'stack_block']:
            if predicted_label == 1:
                only_closed = True
                only_closed_list.append(False)
                save_path = '/media/jannik/media/thesis/GitHub/model_jannik/utils/classification/closed'
                os.makedirs(save_path, exist_ok=True)
                img_save_path = os.path.join(save_path, f'{eval_counter}_{step}.png')
                gripper_scene = cv2.cvtColor(img_gripper_class, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_save_path, gripper_scene)
            else:
                only_closed = False
                only_closed_list.append(False)
                save_path = '/media/jannik/media/thesis/GitHub/model_jannik/utils/classification/open'
                os.makedirs(save_path, exist_ok=True)
                img_save_path = os.path.join(save_path, f'{eval_counter}_{step}.png')
                gripper_scene = cv2.cvtColor(img_gripper_class, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_save_path, gripper_scene)

        if len(current_task_info) > 0:
            logger.info(f'Task succeeded after {step} steps')

            if debug:
                print(colored("success", "green"), end=" ")

            if create_video:
                # img, img_gripper = env.render(mode="rgb_array")

                image, image_gripper, img, img_gripper = do_annot(img, img_gripper, lang_annotation, subtask, step,
                                                                  do_switch, current_traj, current_step,
                                                                  amount_switched, step, current_trajectory_sim,
                                                                  std_curr_traj_sim, relative_action, only_closed,
                                                                  ooa,
                                                                  'D')

                video_frames_static_rgb.append(img)
                video_frames_gripper_rgb.append(img_gripper)

                video_frames_static_mask.append(image)
                video_frames_gripper_mask.append(image_gripper)

            if create_video:
                create_video_frames_rgb(video_frames_static_rgb, video_frames_gripper_rgb, success_folder, subtask,
                                        eval_counter, change)
                create_video_frames_mask(video_frames_static_mask, video_frames_gripper_mask, success_folder,
                                         subtask,
                                         eval_counter, change)

                model.save_best_img_new(best_masks, subtask, mask=True)
                model.save_best_img_new(best_rgbs, subtask, mask=False)
            add_data(subtask, eval_counter, step, True)
            return True
        else:
            logger.info('Task failed')
            if create_video:
                # img, img_gripper = env.render(mode="rgb_array")

                image, image_gripper, img, img_gripper = do_annot(img, img_gripper, lang_annotation, subtask, step,
                                                                  do_switch, current_traj, current_step,
                                                                  amount_switched, step, current_trajectory_sim,
                                                                  std_curr_traj_sim, relative_action, only_closed,
                                                                  ooa,
                                                                  'D')
                video_frames_static_rgb.append(img)
                video_frames_gripper_rgb.append(img_gripper)

                video_frames_static_mask.append(image)
                video_frames_gripper_mask.append(image_gripper)

            action_index += 1
    if debug:
        print(colored("fail", "red"), end=" ")

    if create_video:
        create_video_frames_rgb(video_frames_static_rgb, video_frames_gripper_rgb, fail_folder, subtask,
                                eval_counter, change)
        create_video_frames_mask(video_frames_static_mask, video_frames_gripper_mask, fail_folder, subtask,
                                 eval_counter, change)

        model.save_best_img_new(best_masks, subtask, mask=True)
        model.save_best_img_new(best_rgbs, subtask, mask=False)

    add_data(subtask, eval_counter, step, False)

    return False

def add_text_to_best(img, step, trajectory, trajectory_sim, std, mask):
    image = img.copy()
    if not mask:
        text_color = (0, 0, 0)  # Black color
    else:
        text_color = (255, 255, 255)  # White color

    text_x = int(img.shape[1] * 0.05)
    text_y = int(img.shape[0] * 0.95)
    text_position = (text_x, text_y)


    image = cv2.putText(img, f'{step}_{trajectory}_{trajectory_sim}_{std}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                   text_color, 1)
    return image


def do_annot(img, img_gripper, lang_annotation, subtask, step, do_switch, trajectory, best_step, amount_switched, real_switch, curr_traj_sim, current_std, relative_action, only_closed, ooa, env):
    lang_annotation = ''
    image = model.get_mask(img, subtask, env, gripper=False)
    image_gripper = model.get_mask(img_gripper, subtask, env, gripper=True)

    img = join_vis_lang(img, lang_annotation, step=step, do_switch=do_switch, trajectory=trajectory, top_step=best_step, amount_switched = amount_switched, real_switch = real_switch, curr_traj_sim = curr_traj_sim, current_std=current_std, relative_action=relative_action, only_closed=only_closed, ooa=ooa)
    img_gripper = join_vis_lang(img_gripper, lang_annotation, step=step, do_switch=do_switch, trajectory=trajectory,top_step=best_step, amount_switched = amount_switched, real_switch=real_switch, curr_traj_sim = curr_traj_sim, current_std=current_std, relative_action=relative_action, only_closed=only_closed, ooa=ooa)

    return image, image_gripper, img, img_gripper



def create_combined_video(static_rgb_frames, gripper_rgb_frames, static_mask_frames, gripper_mask_frames,
                          success_fail_folder, subtask, eval_counter):
    if not os.path.exists(success_fail_folder):
        os.makedirs(success_fail_folder)
    video_path = os.path.join(success_fail_folder, f"{subtask}_{eval_counter:.0f}_combined.mp4")
    static_height, static_width, _ = static_rgb_frames[0].shape
    combined_width = static_width * 2
    combined_height = static_height * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, (combined_width, combined_height), isColor=True)
    for i in range(len(static_rgb_frames)):

        static_rgb_frame = static_rgb_frames[i]

        gripper_rgb_frame = cv2.resize(gripper_rgb_frames[i], (static_width, static_height))
        static_mask_frame = cv2.resize(static_mask_frames[i], (static_width, static_height))
        static_mask_frame = cv2.cvtColor(static_mask_frame, cv2.COLOR_GRAY2BGR)
        gripper_mask_frame = cv2.resize(gripper_mask_frames[i], (static_width, static_height))
        gripper_mask_frame = cv2.cvtColor(gripper_mask_frame, cv2.COLOR_GRAY2BGR)


        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_frame[:static_height, :static_width] = static_rgb_frame
        combined_frame[:static_height, static_width:] = static_mask_frame
        combined_frame[static_height:, :static_width] = gripper_rgb_frame
        combined_frame[static_height:, static_width:] = gripper_mask_frame

        video_writer.write(combined_frame)
    video_writer.release()

def create_video_frames_mask(static_video_frames, gripper_video_frames, success_fail_folder, subtask, eval_counter, change):
    if not os.path.exists(success_fail_folder):
        os.makedirs(success_fail_folder)
    video_path = os.path.join(success_fail_folder, f"{subtask}_{eval_counter:.0f}_mask_{change}.mp4")
    static_height, static_width = 200, 200
    combined_width = static_width * 2


    fourcc = cv2.VideoWriter_fourcc(*"mp4v") #  *'DIVX'    *"XVID"    *"mp4v"   "H264"
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, (combined_width, static_height), isColor=False)
    for i in range(len(static_video_frames)):
        static_frame = cv2.cvtColor(static_video_frames[i], cv2.COLOR_GRAY2BGR)
        static_frame = cv2.resize(static_frame, (static_width, static_height))
        gripper_frame = cv2.resize(gripper_video_frames[i], (static_width, static_height))
        #print(gripper_frame.shape)
        #plt.imshow(gripper_frame)
        #plt.show()
        gripper_frame = cv2.cvtColor(gripper_frame, cv2.COLOR_GRAY2BGR)

        combined_frame = np.concatenate((static_frame, gripper_frame), axis=1)
        combined_frame_gray = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2GRAY)

        video_writer.write(combined_frame_gray)
    video_writer.release()

def create_video_frames_rgb(static_video_frames, gripper_video_frames, success_fail_folder, subtask, eval_counter, change):
    print("CREATE VUDEI RGB")

    if not os.path.exists(success_fail_folder):
        os.makedirs(success_fail_folder)
    video_path = os.path.join(success_fail_folder, f"{subtask}_{eval_counter:.0f}_rgb_{change}.mp4")
    static_height, static_width = static_video_frames[0].shape[:2]
    combined_width = static_width * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") #  *'DIVX'    *"XVID"    *"mp4v"   "H264"
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, (combined_width, static_height),  isColor=True)
    for i in range(len(static_video_frames)):
        static_frame = static_video_frames[i]
        static_frame = cv2.resize(static_frame, (static_width, static_height))
        gripper_frame = cv2.resize(gripper_video_frames[i], (static_width, static_height))
        combined_frame = np.concatenate((static_frame, gripper_frame), axis=1)
        #combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        video_writer.write(combined_frame)
    video_writer.release()


def create_video_frames_mask_best_match(static_video_frames, gripper_video_frames, task, counter, best_match=False):
    print("CREATE VUDEI MASK")
    save_folder = f'{save_results_path}/{task}/trajectory_videos_masked'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    video_path = os.path.join(save_folder, f"{task}_{ counter:.0f}.mp4")
    static_height, static_width = static_video_frames[0].shape[:2]
    combined_width = static_width * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # *'DIVX'    *"XVID"    *"mp4v"   "H264"
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, (combined_width, static_height), isColor=False)
    for i in range(len(static_video_frames)):
        static_frame = cv2.cvtColor(static_video_frames[i], cv2.COLOR_GRAY2BGR)
        gripper_frame = cv2.resize(gripper_video_frames[i], (static_width, static_height))
        gripper_frame = cv2.cvtColor(gripper_frame, cv2.COLOR_GRAY2BGR)

        combined_frame = np.concatenate((static_frame, gripper_frame), axis=1)
        combined_frame_gray = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2GRAY)

        video_writer.write(combined_frame_gray)
    video_writer.release()

def create_video_frames_rgb_best_match(static_video_frames, gripper_video_frames, task, counter, best_match = False):

    save_folder = f'{save_results_path}/{task}/trajectory_videos_rgb'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    video_path = os.path.join(save_folder, f"{task}_{counter:.0f}.mp4")
    static_height, static_width, _ = static_video_frames[0].shape
    combined_width = static_width * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # *'DIVX'    *"XVID"    *"mp4v"   "H264"
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, (combined_width, static_height), isColor=True)
    for i in range(len(static_video_frames)):
        static_frame = static_video_frames[i]
        gripper_frame = cv2.resize(gripper_video_frames[i], (static_width, static_height))
        combined_frame = np.concatenate((static_frame, gripper_frame), axis=1)
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        video_writer.write(combined_frame)
    video_writer.release()




