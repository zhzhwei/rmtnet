import numpy as np
import pandas as pd
import cv2
import albumentations as A

from torch.utils.data import Dataset, DataLoader, Subset


class Data(Dataset):
    def __init__(self, cfg, data_csv, mode):
        """A Dataset object that load all data for running

        Args:
            cfg (CfgNode): Config object containing running configuration
            mode (str): Model running mode
        """
        self.imgs = data_csv["image"].values
        self.labels = data_csv["label"].values
        self.study_IDs = data_csv["study"].values
        self.series = data_csv["series"].values
        self.cfg = cfg

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_dir = self.cfg.DIRS.DATA
        # filename = f"{data_dir}/{self.imgs[idx]}"
        filename = self.imgs[idx]
        study_ID = self.study_IDs[idx]
        series = self.series[idx]

        img = cv2.imread(filename)   
        # print(filename)
        # print(img.shape)
        # RESIZE IMAGE
        img = cv2.resize(img, self.cfg.DATA.SIZE, interpolation=cv2.INTER_AREA)

        # ADD AUGEMTATION
        img = augment_image(img)
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        label = self.labels[idx]

        return filename, study_ID, series, img, label


def augment_image(img):
    transform = A.Compose(
        [
            # A.RandomCrop(width=256, height=256),
            # A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.5),
        ]
    )

    transformed = transform(image=img)
    transformed_image = transformed["image"]

    return transformed_image


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
