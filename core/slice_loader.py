import cv2
import random
import numpy as np
import pandas as pd

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

class Data(Dataset):
    def __init__(self, cfg, data_csv, mode):
        """A Dataset object that load all data for running

        Args:
            cfg (CfgNode): Config object containing running configuration
            mode (str): Model running mode
        """
        self.data_csv = data_csv
        self.study_id = data_csv["study_id"].values
        self.scan_id = data_csv["scan_id"].values
        self.slice_dir = data_csv["slice_dir"].values
        self.label = data_csv["label"].values
        self.cfg = cfg

        # count the number of samples in each class
        class_counts = data_csv['label'].value_counts().to_dict()
        max_count = max(class_counts.values())

        # create a list of slice_dirs that need to be augmented
        self.augmented_data = []
        for label, count in class_counts.items():
            if count < max_count:
                deficit = max_count - count
                slice_dirs = self.data_csv[self.data_csv['label'] == label]['slice_dir'].values
                for _ in range(deficit):
                    selected_slice_dir = random.choice(slice_dirs)
                    self.augmented_data.append((selected_slice_dir, label))  # save the slice_dir and label

    def __len__(self):
        return len(self.slice_dir) + len(self.augmented_data)

    def __getitem__(self, idx):
        if idx < len(self.slice_dir):
            slice_dir = self.slice_dir[idx]
            study_id = self.study_id[idx]
            scan_id = self.scan_id[idx]
            label = self.label[idx]
        else:
            slice_dir, label = self.augmented_data[idx - len(self.slice_dir)]
            study_id = "/".join(slice_dir.split("/")[-3:-2])
            scan_id = "/".join(slice_dir.split("/")[-2:-1])

        slice = cv2.imread(slice_dir)
        slice = cv2.resize(slice, self.cfg.DATA.SIZE, interpolation=cv2.INTER_AREA)

        slice = transforms(image=slice)['image']
        slice = slice / 255.0
        slice = slice.transpose(2, 0, 1)

        return slice_dir, study_id, scan_id, slice, label

def build_dataloader(cfg, mode="train"):
    """Build dataloader

    Returns:
        dataloader: Dataloader object
    """
    if mode == "train":
        data_csv = pd.read_csv(cfg.DATA.CSV.TRAIN)
    elif mode == "valid":
        data_csv = pd.read_csv(cfg.DATA.CSV.VALID)
        # data_csv = data_csv.head(200)
    elif mode == "test":
        data_csv = pd.read_csv(cfg.DATA.CSV.TEST)
    
    dataset = Data(cfg, data_csv, mode)
    # DEBUG: Only take a subset of dataloader to run script
    if cfg.DATA.DEBUG:
        dataset = Subset(dataset, np.random.choice(np.arange(len(dataset)), 904))

    shuffle = True
    drop_last = True
    if mode != "train":
        shuffle = False
        drop_last = False
    dataloader = DataLoader(
        dataset,
        cfg.TRAIN.SLICE_LEVEL_BATCH_SIZE,
        pin_memory=False,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=8,
    )
    return dataloader
