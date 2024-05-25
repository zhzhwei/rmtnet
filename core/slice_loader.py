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
        self.imgs = data_csv["image"].values
        self.labels = data_csv["label"].values
        self.study_IDs = data_csv["study"].values
        self.series = data_csv["series"].values
        self.cfg = cfg

        # count the number of samples in each class
        class_counts = data_csv['label'].value_counts().to_dict()
        max_count = max(class_counts.values())

        # create a list of series that need to be augmented
        self.augmented_series = []
        for label, count in class_counts.items():
            if count < max_count:
                deficit = max_count - count
                label_series = self.data_csv[self.data_csv['label'] == label]['series'].values
                for _ in range(deficit):
                    selected_series = random.choice(label_series)
                    self.augmented_series.append((selected_series, label))  # save the series and label

    def __len__(self):
        return len(self.imgs) + len(self.augmented_series)

    def __getitem__(self, idx):
        if idx < len(self.imgs):
            filename = self.imgs[idx]
            study_ID = self.study_IDs[idx]
            series = self.series[idx]
            label = self.labels[idx]
        else:
            series, label = self.augmented_series[idx - len(self.imgs)]
            filename = self.data_csv[self.data_csv.series == series].image.values[0]
            study_ID = self.data_csv[self.data_csv.series == series].study.values[0]

        img = cv2.imread(filename)
        img = cv2.resize(img, self.cfg.DATA.SIZE, interpolation=cv2.INTER_AREA)

        img = transforms(image=img)['image']
        img = img / 255.0
        img = img.transpose(2, 0, 1)

        return filename, study_ID, series, img, label

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
