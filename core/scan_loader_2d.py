import os
import cv2
import pandas as pd
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast, Compose
)

def get_transforms():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        RandomBrightnessContrast(p=0.5)
    ])

transforms = get_transforms()

def augment_scan(slices):
    augmented_slices = []
    for slice in slices:
        augmented = transforms(image=slice)
        augmented_slices.append(augmented['image'])
    return augmented_slices

class ScanLoader2D(Dataset):
    def __init__(self, cfg, data_csv):
        if cfg.MODEL.NUM_CLASSES == 3:
            labels = [0, 1, 2]
        elif cfg.MODEL.NUM_CLASSES == 4:
            labels = [0, 1, 2, 3]
        self.data_csv = data_csv[data_csv.label.isin(labels)].reset_index(drop=True)
        self.slice_dir = self.data_csv["slice_dir"].values
        self.scan_id = np.unique(self.data_csv["scan_id"].values)
        self.cfg = cfg

        # count the number of samples in each scan
        class_counts = data_csv.groupby('label')['scan_id'].nunique()
        max_count = max(class_counts.values)

        # create a list of scan_ids that need to be augmented
        self.augmented_data = []
        for label, count in class_counts.items():
            if count < max_count:
                deficit = max_count - count
                scan_ids = self.data_csv[self.data_csv['label'] == label]['scan_id'].values
                for _ in range(deficit):
                    selected_scan_id = random.choice(scan_ids)
                    self.augmented_data.append((selected_scan_id, label))  # save the scan_id and label

    def __len__(self):
        return len(self.scan_id) + len(self.augmented_data)
    
    def __getitem__(self, idx):
        if idx < len(self.scan_id):
            # process original data
            slice_dir = self.data_csv[self.data_csv.scan_id == self.scan_id[idx]].slice_dir.values[0]
            scan_dir = "/".join(slice_dir.split("/")[:-1])
            label = self.data_csv[self.data_csv.slice_dir.str.contains(scan_dir)].label.values[0]
        else:
            # process augmented data
            scan_id, label = self.augmented_data[idx - len(self.scan_id)]
            slice_dir = self.data_csv[self.data_csv.scan_id == scan_id].slice_dir.values[0]
            scan_dir = "/".join(slice_dir.split("/")[:-1])

        slice_paths = [os.path.join(scan_dir, f) for f in sorted(os.listdir(scan_dir))]
        
        if "minor_channels" in self.cfg.NAME:
            slices = [cv2.imread(path) for path in slice_paths]
        elif "major_channels" in self.cfg.NAME:
            slices = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in slice_paths]

        np.random.seed(42)
        rng = np.random.default_rng()
        indices = np.arange(len(slices))
        replace = len(indices) < 32
        chosen_indices = rng.choice(indices, size=32, replace=replace)
        slices = [slices[i] for i in chosen_indices]

        if idx >= len(self.scan_id):
            slices = augment_scan(slices)
            
        slices = [cv2.resize(slice, self.cfg.DATA.SIZE, interpolation=cv2.INTER_AREA) for slice in slices]

        if "minor_channels" in self.cfg.NAME:
            slices = np.stack(slices, axis=0)
            slices = slices.transpose(0, 3, 1, 2)  # CHW format for PyTorch
        elif "major_channels" in self.cfg.NAME:
            slices = np.array(slices)  # Convert list to numpy array

        return slices, label
    
def build_dataloader_2d(cfg, mode="train"):
    """Build dataloader

    Returns:
        dataloader: Dataloader object
    """
    if mode == "train":
        data_csv = pd.read_csv(cfg.DATA.CSV.TRAIN)
    elif mode == "valid":
        data_csv = pd.read_csv(cfg.DATA.CSV.VALID)

    dataset = ScanLoader2D(cfg, data_csv)

    # DEBUG: Only take a subset of dataloader to run script
    if cfg.DATA.DEBUG:
        dataset = Subset(dataset, np.random.choice(np.arange(len(dataset)), 64))

    dataloader = DataLoader(
        dataset, cfg.TRAIN.SCAN_LEVEL_BATCH_SIZE, pin_memory=False, shuffle=True, drop_last=False, num_workers=8
    )
    return dataloader

def linear_interpolation(slice_paths, target_num_slices=512):
    """Ensure all scans have the same number of slices"""
    num_slices = len(slice_paths)
    if num_slices == target_num_slices:
        return [cv2.imread(path) for path in slice_paths]
    interpolated_slices = []
    # Calculate interpolation step
    scale_factor = (num_slices - 1) / float(target_num_slices - 1)
    for i in range(target_num_slices):
        index = i * scale_factor
        lower_index = int(np.floor(index))
        upper_index = int(np.ceil(index))
        t = index - lower_index
        if lower_index == upper_index:
            img = cv2.imread(slice_paths[lower_index])
        else:
            img1 = cv2.imread(slice_paths[lower_index])
            img2 = cv2.imread(slice_paths[upper_index])
            img = cv2.addWeighted(img1, 1 - t, img2, t, 0)
        interpolated_slices.append(img)
    return interpolated_slices