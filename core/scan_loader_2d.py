import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A

from torch.utils.data import Dataset, DataLoader, Subset

class ScanLoader2D(Dataset):
    def __init__(self, cfg, data_csv):
        if cfg.MODEL.NUM_CLASSES == 3:
            labels = [0, 1, 2]
        elif cfg.MODEL.NUM_CLASSES == 4:
            labels = [0, 1, 2, 3]
        self.data_csv = data_csv[data_csv.label.isin(labels)].reset_index(drop=True)
        self.imgs = self.data_csv["image"].values
        self.series = np.unique(self.data_csv["series"].values)
        self.cfg = cfg
        
    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        slice_dir = self.data_csv[self.data_csv.series == self.series[idx]].image.values[0]
        scan_dir = "/".join(slice_dir.split("/")[:-1])
        slice_paths = [os.path.join(scan_dir, f) for f in sorted(os.listdir(scan_dir))]

        if "minor_channels" in self.cfg.NAME:
            if self.cfg.DATA.INTERPOLATION:
                slices = linear_interpolation(slice_paths, 512)
            else:
                slices = [cv2.imread(path) for path in slice_paths]
        elif "major_channels" in self.cfg.NAME:
            if self.cfg.DATA.INTERPOLATION:
                slices = linear_interpolation(slice_paths, 512)
            else:
                slices = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in slice_paths]

        np.random.seed(42)
        rng = np.random.default_rng()
        indices = np.arange(len(slices))
        replace = len(indices) < 16
        chosen_indices = rng.choice(indices, size=32, replace=replace)  
        slices = [slices[i] for i in chosen_indices]

        slices = [cv2.resize(slice, self.cfg.DATA.SIZE, interpolation=cv2.INTER_AREA) for slice in slices]
        slices = [slice_augmentation(slice) for slice in slices]
        
        if "minor_channels" in self.cfg.NAME:
            slices = np.stack(slices, axis=0)
            slices = slices.transpose(0, 3, 1, 2)  # CHW format for PyTorch
        elif "major_channels" in self.cfg.NAME:
            slices = np.array(slices)  # Convert list to numpy array
        
        target = self.data_csv[self.data_csv.image.str.contains(scan_dir)].label.values[0]

        return slices, target

def slice_augmentation(img):
    transform = A.Compose(
        [
            # A.RandomCrop(width=128, height=128),
            # A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.5)
        ]
    )

    transformed = transform(image=img)
    transformed_image = transformed["image"]

    processed_imgae = transformed_image / 255.0  # Normalization
    return processed_imgae

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
        dataset, cfg.TRAIN.SCAN_LEVEL_BATCH_SIZE, pin_memory=False, shuffle=True, drop_last=False, num_workers=12
    )
    return dataloader


