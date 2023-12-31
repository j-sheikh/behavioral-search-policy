import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class GraspDataset(Dataset):
    def __init__(self, image_info, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_info = image_info

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_name, label = self.image_info[idx][0], self.image_info[idx][1]
        subfolder = "closed" if label == 1 else "open"
        file_path = os.path.join(self.root_dir, subfolder)
        image_path = os.path.join(file_path, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class GraspDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=32):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None

    def prepare_data(self):
        closed_gripper_images = [(image, 1)
                                 for image in os.listdir(os.path.join(self.dataset_path, 'closed')) if
                                 image.endswith('.png')]
        open_gripper_images = [(image, 0)
                               for image in os.listdir(os.path.join(self.dataset_path, 'open')) if
                               image.endswith('.png')]

        closed_gripper_train = closed_gripper_images[:int(0.7 * len(closed_gripper_images))]
        open_gripper_train = open_gripper_images[:int(0.7 * len(open_gripper_images))]

        closed_gripper_val = closed_gripper_images[
                             int(0.7 * len(closed_gripper_images)):int(0.85 * len(closed_gripper_images))]
        open_gripper_val = open_gripper_images[int(0.7 * len(open_gripper_images)):int(0.85 * len(open_gripper_images))]

        closed_gripper_test = closed_gripper_images[int(0.85 * len(closed_gripper_images)):]
        open_gripper_test = open_gripper_images[int(0.85 * len(open_gripper_images)):]

        self.train_dataset = GraspDataset(closed_gripper_train + open_gripper_train,
                                          root_dir=self.dataset_path, transform=self.data_transform)
        self.val_dataset = GraspDataset(closed_gripper_val + open_gripper_val,
                                        root_dir=self.dataset_path, transform=self.data_transform)
        self.test_dataset = GraspDataset(closed_gripper_test + open_gripper_test,
                                         root_dir=self.dataset_path, transform=self.data_transform)

        # Calculate class weights based on the new train_dataset
        class_counts = [len(self.train_dataset) - sum(label == 0 for _, label in self.train_dataset),
                        sum(label == 0 for _, label in self.train_dataset)]
        total_samples = sum(class_counts)
        class_weights = torch.tensor([total_samples / count for count in class_counts], dtype=torch.float32)
        self.class_weights = class_weights.to(self.device)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
