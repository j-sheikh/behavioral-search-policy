import numpy as np
import cv2
import math
import pandas as pd
import os
import pickle
from skimage.metrics import mean_squared_error, structural_similarity
import yaml
import shutil
import random
from collections import Counter
from torchvision import transforms

random_seed = 42
np.random.seed(random_seed)

def compute_mse(image1, image2):
    """Compute Mean Squared Error (MSE) between two images."""
    return mean_squared_error(image1, image2)

def compute_ssim(image1, image2, data_range=1.0):
    """Compute Structural Similarity Index (SSIM) between two images."""
    return structural_similarity(image1, image2, multichannel=False, data_range=data_range)


def dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    true_positives = np.count_nonzero(intersection)
    false_positives = np.count_nonzero(mask2) - true_positives
    false_negatives = np.count_nonzero(mask1) - true_positives
    dice_coeff = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
    return dice_coeff


def weighted_dice_coefficient(mask1, mask2):

    total_pixels_mask1 = np.count_nonzero(mask1)
    total_pixels_mask2 = np.count_nonzero(mask2)

    if total_pixels_mask1 == 0:
        #print("total pixel mask 1 == 0")
        return 0.0, 0.0, 0.0
    elif total_pixels_mask2 == 0:
        #print("total pixel mask 2 == 0")
        return 0.0, 0.0, 0.0

    dice_coef = dice_coefficient(mask1, mask2)
    size_coef = total_pixels_mask1 / total_pixels_mask2
    size_coef = min(size_coef, 1 / size_coef)
    #size_coef = min(size_coef, 1.0)
    if np.isnan(size_coef):
        #print("size_coef nan")
        return 0.0, 0.0, 0.0
    elif np.isinf(size_coef):
        #print("size_coef inf")
        return 0.0, 0.0, 0.0

    weighted_dice_coef = dice_coef * size_coef


    return weighted_dice_coef, dice_coef, size_coef


def new_weighted_dice_coefficient(mask1, mask2):
    #mask1 == reference mask
    sum_of_pixels = np.sum(mask1)
    total_pixels = mask1.shape[0]
    ratio = sum_of_pixels / total_pixels
    dice_coef = dice_coefficient(mask1, mask2)
    weighted_dice_coef = np.round(dice_coef * ratio * 10,3)
    return weighted_dice_coef

def jaccard_similarity(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    score = np.sum(intersection) / np.sum(union)
    return score

def weighted_jaccard_index(mask1, mask2):
    jaccard = jaccard_similarity(mask1, mask2)
    size_coef = np.sum(mask1) / mask1.size
    weighted_jaccard = jaccard * size_coef
    return weighted_jaccard


def new_jaccard_similarity(mask1, mask2):
    weight_object = 1.5
    weight_background = 1.0
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    tp = np.sum(intersection) * weight_object
    fp = np.sum(mask2) - tp * weight_background
    fn = np.sum(mask1) - tp * weight_object

    weighted_jaccard = tp / (tp + fp + fn)
    return weighted_jaccard


def apply_gaussian_blur(mask, kernel_size=3, sigma=0):
    blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    return blurred_mask
def calculate_mean(data):
    return round(sum(data) / len(data),3)
def calculate_standard_deviation(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_deviation = round(math.sqrt(variance),3)
    return std_deviation

def calculate_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        median = round((sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2,3)
    else:
        median = round(sorted_data[n // 2],3)
    return median

def calculate_range(data):
    return round(max(data) - min(data),3)

def calculate_iqr(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    q1_index = n // 4
    q3_index = n - q1_index - 1
    q1 = sorted_data[q1_index]
    q3 = sorted_data[q3_index]
    iqr = round(q3 - q1,3)
    return iqr



def creat_segmentation_dict():
    static_shape = 150
    gripper_shape = 84

    object_dict = {
        'static': {
            'base__drawer': {
                'color': 'gray',
                'hsv_lower': (0, 0, 0),
                'hsv_upper': (250, 0, 100),
                'region': 'lower',
                'target_w': static_shape,
                'target_h': static_shape
            },
            'base__slide': {
                'color': 'gray',
                'hsv_lower': (0, 0, 0),
                'hsv_upper': (250, 0, 100),
                'region': 'mid_left',
                'target_w': static_shape,
                'target_h': static_shape
            },
            'led': {
                'color': 'black',
                'hsv_lower': (0, 0, 0),
                'hsv_upper': (180, 255, 30),
                'region': 'unknown',
                'target_w': static_shape,
                'target_h':static_shape
            },
            'lightbulb': {
                'color': 'gray',
                'hsv_lower': (0, 0, 0),
                'hsv_upper': (250, 0, 100),
                'region': 'mid_right',
                'target_w': static_shape,
                'target_h': static_shape
            },
            'block_blue': {
                'color': 'blue',
                'hsv_lower': (100, 90, 90),
                'hsv_upper': (120, 255, 255),
                'region': 'unknown',
                'target_w': static_shape,
                'target_h': static_shape
            },
            'block_red': {
                'color': 'red',
                'hsv_lower': (0, 100, 100),
                'hsv_upper': (10, 255, 255),
                'region': 'unknown',
                'target_w': static_shape,
                'target_h': static_shape
            },
            'block_pink': {
                'color': 'pink',
                'hsv_lower': (140, 100, 100),
                'hsv_upper': (170, 255, 255),
                'region': 'unknown',
                'target_w': static_shape,
                'target_h': static_shape
            }
        },
        'gripper': {
            'base__drawer': {
                'color': 'gray',
                'hsv_lower': (0, 0, 0),
                'hsv_upper': (250, 0, 100),
                'region': 'lower',
                'target_w': gripper_shape,
                'target_h': gripper_shape
            },
            'base__slide': {
                'color': 'gray',
                'hsv_lower': (0, 0, 0),
                'hsv_upper': (250, 0, 100),
                'region': 'mid_left',
                'target_w': gripper_shape,
                'target_h': gripper_shape
            },
            'led': {
                'color': 'black',
                'hsv_lower': (0, 0, 0),
                'hsv_upper': (180, 255, 30),
                'region': 'unknown',
                'target_w': gripper_shape,
                'target_h': gripper_shape
            },
            'lightbulb': {
                'color': 'gray',
                'hsv_lower': (0, 0, 0),
                'hsv_upper': (250, 0, 100),
                'region': 'mid_right',
                'target_w': gripper_shape,
                'target_h': gripper_shape
            },
            'block_blue': {
                'color': 'blue',
                'hsv_lower': (100, 90, 90),
                'hsv_upper': (120, 255, 255),
                'region': 'unknown',
                'target_w': gripper_shape,
                'target_h': gripper_shape
            },
            'block_red': {
                'color': 'red',
                'hsv_lower': (0, 100, 100),
                'hsv_upper': (10, 255, 255),
                'region': 'unknown',
                'target_w': gripper_shape,
                'target_h': gripper_shape
            },
            'block_pink': {
                'color': 'pink',
                'hsv_lower': (140, 100, 100),
                'hsv_upper': (170, 255, 255),
                'region': 'unknown',
                'target_w': gripper_shape,
                'target_h': gripper_shape
            }
        }
    }

    path = './utils'
    file_name = 'object_segmentation_settings.pkl'
    full_path = os.path.join(path, file_name)
    with open(full_path, 'wb') as file:
        pickle.dump(object_dict, file)

#creat_segmentation_dict()



def creat_task_object_mapping():
    with open('./hulc/calvin_env/conf/tasks/new_playtable_tasks.yaml', 'r') as file:
        # Load the YAML content
        yaml_data = yaml.safe_load(file)

    task_object_mapping = {}

    for task, info in yaml_data['tasks'].items():
        print(task, info)
        if len(info) > 1:
            object_name = info[1]
            task_object_mapping[task] = object_name

    file_name = 'task_object_mapping.pkl'
    path = './utils'
    full_path = os.path.join(path, file_name)
    with open(full_path, 'wb') as file:
        pickle.dump(task_object_mapping, file)
#creat_task_object_mapping()



def new_labels(env, object):

    new_labels_dict = {
        'A': {'block_pink': 'block_blue', 'block_red': 'block_red',
              'block_blue': 'block_pink'},
        'B': {'block_pink': 'block_blue', 'block_red': 'block_pink',
              'block_blue': 'block_red'},
        'C': {'block_pink': 'block_red', 'block_red': 'block_pink',
              'block_blue': 'block_blue'},
        'D': {'block_pink': 'block_pink', 'block_red': 'block_red', 'block_blue': 'block_blue'}
    }

    if env in new_labels_dict and object in new_labels_dict[env]:
        return new_labels_dict[env][object]
    else:
        return None

def new_classes(env, object):

    new_labels_dict = {
        'A': {'block_pink': 'block_small', 'block_red': 'block_medium',
              'block_blue': 'block_large'},
        'B': {'block_pink': 'block_small', 'block_red': 'block_large',
              'block_blue': 'block_medium'},
        'C': {'block_pink': 'block_medium', 'block_red': 'block_large',
              'block_blue': 'block_small'},
        'D': {'block_pink': 'block_large', 'block_red': 'block_medium', 'block_blue': 'block_small'}
    }

    if env in new_labels_dict and object in new_labels_dict[env]:
        return new_labels_dict[env][object]
    else:
        return None


def transform_data_classifier(pil_image):
    data_transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply transformations to the PIL image
    input_tensor = data_transform(pil_image)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor



def modify_task_name(task_name, old_object_name, new_object_name):
    old_color = get_color(old_object_name)
    new_color = get_color(new_object_name)
    if old_color  in task_name:
        modified_task_name = task_name.replace(old_color, new_color)
        return modified_task_name
    else:
        return None
def get_color(object):
    return object.split('_')[1]


def get_tasks(task, object):
    tasks = []
    if not task in ['turn_on_led', 'turn_off_led', 'place_in_slider', 'place_in_drawer', 'push_into_drawer', 'unstack_block', 'stack_block']:
        for env in ['A', 'B', 'C']:
            new_object = new_labels(env, object)
            if new_object is None:
                tasks.append((env, task))
            else:
                tasks.append((env, modify_task_name(task, object, new_object)))
    elif task == 'place_in_slider':
        for env in ['A']:
            tasks.append((env, task))

    elif task in ['turn_on_led', 'turn_off_led']:
        for env in ['C']:
            tasks.append((env, 'turn_on_led'))


    else:
        for env in ['A', 'B', 'C']:
            tasks.append((env, task))
    return tasks


def generate_sorted_multi_task(eval_sequences, tasks):
    multi_task = []

    for tup in eval_sequences:
        subtasks = tup[1]

        selected_task = None
        for task in tasks:
            if subtasks and subtasks[0] == task:
                selected_task = task
                break

        if selected_task is None:
            for task in tasks:
                if task in subtasks:
                    selected_task = task
                    break

        if selected_task is not None:
            new_tup = (tup[0], [selected_task])
            multi_task.append(new_tup)

    task_counts = {task: 0 for task in tasks}
    selected_tasks = []
    for task in tasks:
        if task_counts[task] < 10:
            all_tasks = [task_tuple for task_tuple in multi_task if task_tuple[1][0] == task]
            selected_sample = random.sample(all_tasks, min(10 - task_counts[task], len(all_tasks)))
            selected_tasks.extend(selected_sample)
            task_counts[task] += len(selected_sample)

    task_names = [task_tuple[1][0] for task_tuple in selected_tasks]
    task_counts = Counter(task_names)
    max_task_count = max(task_counts.values())
    task_indices = {task: [] for task in task_counts.keys()}

    for idx, task_name in enumerate(task_names):
        task_indices[task_name].append(idx)

    sorted_multi_task = []
    for _ in range(max_task_count):
        for task_name, indices in task_indices.items():
            if indices:
                idx = indices.pop(0)
                sorted_multi_task.append(multi_task[idx])


    tasks = [tup[1] for tup in sorted_multi_task]
    task_counts = {}
    for task_list in tasks:
        for task in task_list:
            if task in task_counts:
                task_counts[task] += 1
            else:
                task_counts[task] = 1
    print(task_counts)

    return sorted_multi_task


def generate_MTLC_eval(tasks, amount=10):
    input_folder = './hulc/dataset/task_ABC_D/validation'
    lang_ann = np.load(f'{input_folder}/lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy',
                       allow_pickle=True)
    lang_ann = lang_ann[()]

    dataset = {}
    task_counts = {task: 0 for task in tasks}
    for task in tasks:
        task_indices = [
            indx for indx, task_name in enumerate(lang_ann['language']['task'])
            if task_name == task
        ]
        if task_counts[task] < amount:
            selected_sample = random.sample(task_indices, min(amount - task_counts[task], len(task_indices)))
            dataset[task] = selected_sample
            task_counts[task] += len(selected_sample)

    return dataset


def load_data(episode_index):
    input_folder = './hulc/dataset/task_ABC_D/validation'
    lang_ann = np.load(f'{input_folder}/lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy',
                       allow_pickle=True)
    lang_ann = lang_ann[()]

    start, end = lang_ann['info']['indx'][episode_index]

    if end < start:
        start, end = end, start
    episode_index = str(start).zfill(7)
    episode_path = os.path.join(input_folder, f"episode_{episode_index}.npz")
    episode = np.load(episode_path)

    return episode
