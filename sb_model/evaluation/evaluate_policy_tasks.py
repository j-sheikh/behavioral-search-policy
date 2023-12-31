import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import random
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import cv2

from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
from calvin_agent.evaluation.utils import get_default_model_and_env
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
    join_vis_lang_old
)


from grasp_classifier.Classifer import GraspClassifier

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

classification_checkpoint_path = './utils'
ckpt_files = [file for file in os.listdir(classification_checkpoint_path) if file.endswith('.ckpt')]
model_save_path = os.path.join(classification_checkpoint_path, ckpt_files[0])
classification_model = GraspClassifier.load_from_checkpoint(model_save_path, num_classes=2, class_weights=class_weights)
classification_model.eval()




save_results_path = 'xxx'

model = SBP()

def evaluate_policy_singlestep(model, env, datamodule, args, checkpoint):

    print("IN EVALUATE POLICY SINGLESTEP")
    model = model

    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    task_to_id_dict = torch.load(checkpoint)["task_to_id_dict"]
    dataset = datamodule.val_dataloader().dataset.datasets["vis"]

    results = Counter()

    for task, ids in task_to_id_dict.items():
        for i in ids:
            episode = dataset[int(i)]
            results[task] += rollout(env, episode, i, task_oracle, args, task, val_annotations)
        print(f"{task}: {results[task]} / {len(ids)}")

    print(f"SR: {sum(results.values()) / sum(len(x) for x in task_to_id_dict.values()) * 100:.1f}%")


def rollout(env, episode, eval_counter, task_oracle, args, subtask, val_annotations, create_video = True):
    # state_obs, rgb_obs, depth_obs = episode["robot_obs"], episode["rgb_obs"], episode["depth_obs"]
    reset_info = episode["state_info"]
    # idx = episode["idx"]
    obs = env.reset(robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"][0])
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]

    start_info = env.get_info()



    print("EVAL with subtask", subtask)
    device = torch.device('cuda')

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
    fail = False
    alpha = 0.99  # 90% with 0,01 -> 100%
    current_trajectory_sim = 0.0
    current_traj = None
    current_step = None
    amount_switched = 0
    change = 0
    ooa = False
    metric = 'weighted_dice'
    threshold = 0.01
    only_closed = False
    only_static = False


    for step in range(args.ep_len):
        print("STEP", step)
        do_switch = False

        print(trajectory_similarities)
        if len(trajectory_similarities) >= 2:
            std_curr_traj_sim = np.round(np.std(trajectory_similarities[-2:]), 3)

            if trajectory_similarities[-1] < 0.01 and trajectory_similarities[-2] < 0.01:
                threshold = 0.002
            elif trajectory_similarities[-1] < 0.1 and trajectory_similarities[-2] < 0.1:
                threshold = 0.01
            else:
                threshold = 0.015
        else:
            std_curr_traj_sim = 0
            print("Not enough values to calculate std.")

        print("STD_CURR_TRAJ_SIM", std_curr_traj_sim)
        img, img_gripper = env.render(mode="rgb_array")
        if std_curr_traj_sim >= threshold or action_index >= len(actions):

            best_trajectories, new_best_mask, new_best_rgb = model.run_find_best(
                subtask, img, img_gripper, eval_counter, step, alpha, metric, True)

            print("BEST TRAJECTORIES", best_trajectories)

            if best_trajectories is None:
                return False

            elif current_traj is None:

                best_result = best_trajectories[0]

                new_sim = best_result[0]
                new_traj = best_result[1]
                new_step = best_result[3]

                print("NEW SIMILARITY", new_sim)
                print("NEW TRAJECTORY", new_traj)

                abs_actions, rel_actions, top_frames_static, top_frames_gripper, top_frames_static_masked, top_frames_gripper_masked, only_static = model.run_pipeline(
                    subtask, eval_counter, step, new_traj, new_step)
                only_static = only_static
                if only_static or current_traj is None:
                    print("START-absolute")


                    current_traj = new_traj
                    current_step = new_step

                    relative_action = False
                    abs_actions = np.array(abs_actions, dtype=np.float32)
                    actions = torch.tensor(abs_actions).to(device)
                else:
                    print("START-relative")

                    relative_action = True
                    rel_actions = np.array(rel_actions, dtype=np.float32)
                    actions = torch.tensor(rel_actions).to(device)

                action_index = 0

            elif current_traj is not None or action_index >= len(actions):

                print(f"check for new actions for {subtask}")

                print("CURRENT TRAJ SIM", current_trajectory_sim)
                print("CURRENT TRAJ", current_traj)

                best_result = best_trajectories[0]

                new_sim = best_result[0]
                new_traj = best_result[1]
                new_step = best_result[3]

                print("NEW SIMILARITY", new_sim)
                print("NEW TRAJECTORY", new_traj)

                if action_index >= len(actions):
                    print("OOA-SWITCH")
                    ooa = True

                    trajectory_similarities = []
                    amount_switched += 1
                    do_switch = True

                    current_traj = new_traj
                    current_step = new_step

                    abs_actions, rel_actions, top_frames_static, top_frames_gripper, top_frames_static_masked, top_frames_gripper_masked, only_static = model.run_pipeline(
                        subtask, eval_counter, step, new_traj, new_step)
                    only_static = only_static
                    if only_static:
                        relative_action = False
                        abs_actions = np.array(abs_actions, dtype=np.float32)
                        actions = torch.tensor(abs_actions).to(device)
                    else:
                        relative_action = True
                        rel_actions = np.array(rel_actions, dtype=np.float32)
                        actions = torch.tensor(rel_actions).to(device)

                    if only_closed:
                        actions[:, -1] = -1.0

                    action_index = 0

                elif new_sim > current_trajectory_sim:
                    ooa = False

                    trajectory_similarities = []
                    amount_switched += 1
                    do_switch = True

                    current_traj = new_traj
                    current_step = new_step

                    abs_actions, rel_actions, top_frames_static, top_frames_gripper, top_frames_static_masked, top_frames_gripper_masked, only_static = model.run_pipeline(
                        subtask, eval_counter, step, new_traj, new_step)
                    only_static = only_static
                    if only_static:
                        relative_action = False
                        abs_actions = np.array(abs_actions, dtype=np.float32)
                        actions = torch.tensor(abs_actions).to(device)
                    else:
                        relative_action = True
                        rel_actions = np.array(rel_actions, dtype=np.float32)
                        actions = torch.tensor(rel_actions).to(device)

                    if only_closed:
                        actions[:, -1] = -1.0

                    action_index = 0

        action = actions[action_index]

        if len(only_closed_list) >= 2:
         if only_closed_list[-2] and not only_closed_list[-1]:
                fail = True

        obs, _, _, current_info, applied_action = env.step(action, relative_action)

        current_traj_static_rgb = top_frames_static[action_index]
        current_traj_gripper_rgb = top_frames_gripper[action_index]

        current_trajectory_sim = model.compare_trajectory_with_live(subtask, img, img_gripper, current_traj_static_rgb,
                                                                    current_traj_gripper_rgb,
                                                                    current_traj, step, alpha, metric)

        trajectory_similarities.append(current_trajectory_sim)

        best_masks.append(new_best_mask)
        best_rgbs.append(new_best_rgb)

        if args.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})

        _, img_gripper = env.render('rgb_array')
        pil_image = Image.fromarray(np.uint8(img_gripper))
        input_tensor = transform_data_classifier(pil_image)
        predicted_label, predicted_probabilities = classification_model.predict(input_tensor)
        if predicted_label == 1:
            only_closed = True
            only_closed_list.append(True)
        else:
            only_closed = False
            only_closed_list.append(True)

        # check if current step solves a task
        if len(current_task_info) > 0:
            if args.debug:
                print(colored("S", "green"), end=" ")

            if create_video:
                img, img_gripper = env.render(mode="rgb_array")

                image, image_gripper, img, img_gripper = do_annot(img, img_gripper, lang_annotation, subtask, step,
                                                                  do_switch, current_traj, current_step,
                                                                  amount_switched, step, current_trajectory_sim,
                                                                  std_curr_traj_sim, relative_action, only_closed, ooa)

                video_frames_static_rgb.append(img)
                video_frames_gripper_rgb.append(img_gripper)

                video_frames_static_mask.append(image)
                video_frames_gripper_mask.append(image_gripper)

            if create_video:
                create_video_frames_rgb(video_frames_static_rgb, video_frames_gripper_rgb, success_folder, subtask,
                                        eval_counter, change)
                create_video_frames_mask(video_frames_static_mask, video_frames_gripper_mask, success_folder, subtask,
                                         eval_counter, change)

                model.save_best_img_new(best_masks, subtask, mask=True)
                model.save_best_img_new(best_rgbs, subtask, mask=False)

            return True
        else:

            if create_video:
                img, img_gripper = env.render(mode="rgb_array")

                image, image_gripper, img, img_gripper = do_annot(img, img_gripper, lang_annotation, subtask, step,
                                                                  do_switch, current_traj, current_step,
                                                                  amount_switched, step, current_trajectory_sim,
                                                                  std_curr_traj_sim, relative_action, only_closed, ooa)
                video_frames_static_rgb.append(img)
                video_frames_gripper_rgb.append(img_gripper)

                video_frames_static_mask.append(image)
                video_frames_gripper_mask.append(image_gripper)

        action_index += 1

    if create_video:
        create_video_frames_rgb(video_frames_static_rgb, video_frames_gripper_rgb, fail_folder, subtask,
                                eval_counter, change)
        create_video_frames_mask(video_frames_static_mask, video_frames_gripper_mask, fail_folder, subtask,
                                 eval_counter, change)

        model.save_best_img_new(best_masks, subtask, mask=True)
        model.save_best_img_new(best_rgbs, subtask, mask=False)



    if args.debug:
        print(colored("F", "red"), end=" ")
    return False


def do_annot(img, img_gripper, lang_annotation, subtask, step, do_switch, trajectory, best_step, amount_switched,
             real_switch, curr_traj_sim, current_std, relative_action, only_closed, ooa):
    lang_annotation = ''
    image = model.get_mask(img, subtask, gripper=False)
    image_gripper = model.get_mask(img_gripper, subtask, gripper=True)

    img = join_vis_lang(img, lang_annotation, step=step, do_switch=do_switch, trajectory=trajectory, top_step=best_step,
                        amount_switched=amount_switched, real_switch=real_switch, curr_traj_sim=curr_traj_sim,
                        current_std=current_std, relative_action=relative_action, only_closed=only_closed, ooa=ooa)
    img_gripper = join_vis_lang(img_gripper, lang_annotation, step=step, do_switch=do_switch, trajectory=trajectory,
                                top_step=best_step, amount_switched=amount_switched, real_switch=real_switch,
                                curr_traj_sim=curr_traj_sim, current_std=current_std, relative_action=relative_action,
                                only_closed=only_closed, ooa=ooa)

    return image, image_gripper, img, img_gripper


def create_video_frames_mask(static_video_frames, gripper_video_frames, success_fail_folder, subtask, eval_counter,
                             change):
    if not os.path.exists(success_fail_folder):
        os.makedirs(success_fail_folder)
    video_path = os.path.join(success_fail_folder, f"{subtask}_{eval_counter:.0f}_mask_{change}.mp4")
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


def create_video_frames_rgb(static_video_frames, gripper_video_frames, success_fail_folder, subtask, eval_counter,
                            change):
    if not os.path.exists(success_fail_folder):
        os.makedirs(success_fail_folder)
    video_path = os.path.join(success_fail_folder, f"{subtask}_{eval_counter:.0f}_rgb_{change}.mp4")
    static_height, static_width, _ = static_video_frames[0].shape
    combined_width = static_width * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # *'DIVX'    *"XVID"    *"mp4v"   "H264"
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, (combined_width, static_height), isColor=True)
    for i in range(len(static_video_frames)):
        static_frame = static_video_frames[i]
        gripper_frame = cv2.resize(gripper_video_frames[i], (static_width, static_height))
        combined_frame = np.concatenate((static_frame, gripper_frame), axis=1)
        # combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        video_writer.write(combined_frame)
    video_writer.release()


def create_video_frames_mask_best_match(static_video_frames, gripper_video_frames, task, counter):
    save_folder = f'{save_results_path}/{task}/trajectory_videos_masked'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    video_path = os.path.join(save_folder, f"{task}_{counter:.0f}.mp4")
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


def create_video_frames_rgb_best_match(static_video_frames, gripper_video_frames, task, counter):
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


def add_text_to_best(img, step, trajectory, trajectory_sim, std, mask):
    image = img.copy()
    if not mask:
        text_color = (0, 0, 0)  # Black color
    else:
        text_color = (255, 255, 255)  # White color

    text_x = int(img.shape[1] * 0.05)
    text_y = int(img.shape[0] * 0.95)
    text_position = (text_x, text_y)

    image = cv2.putText(img, f'{step}_{trajectory}_{trajectory_sim}_{std}', text_position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color, 1)
    return image



if __name__ == "__main__":
    seed_everything(0, workers=True)
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Manually specify checkpoint path (default is latest). Only used for calvin_agent.",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    args = parser.parse_args()

    # Do not change
    args.ep_len = 240

    checkpoints = []
    if args.checkpoint is None and args.last_k_checkpoints is None:
        print("Evaluating model with last checkpoint.")
        checkpoints = [get_last_checkpoint(Path(args.train_folder))]
    elif args.checkpoint is not None:
        print(f"Evaluating model with checkpoint {args.checkpoint}.")
        checkpoints = [args.checkpoint]
    elif args.checkpoint is None and args.last_k_checkpoints is not None:
        print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
        checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]

    env = None
    for checkpoint in checkpoints:
        model, env, datamodule = get_default_model_and_env(args.train_folder, args.dataset_path, checkpoint, env=env)
        evaluate_policy_singlestep(model, env, datamodule, args, checkpoint)

