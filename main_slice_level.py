import os
import sys
import warnings

from utils import parse_args, setup_determinism
from utils import build_loss_func, build_optim
from utils import build_scheduler, load_checkpoint
from utils import build_model

from config.cfg_defaults import get_cfg_defaults
from core.slice_loader import build_dataloader
from core.train_model import train_model
from core.valid_model import valid_model
from model.RMTNET import RMTNET

import torch.multiprocessing
from torch.cuda.amp import GradScaler
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")
scaler = GradScaler()

sys.stdout = open('./logs/training.out','a')

def main(cfg, args):

    # Declare variables
    best_metric = 0
    start_epoch = 0
    mode = args.mode

    trainloader = build_dataloader(cfg, mode="train")
    validloader = build_dataloader(cfg, mode="valid")

    # Define model/loss/optimizer/Scheduler
    model = RMTNET(num_classes=cfg.MODEL.NUM_CLASSES)

    if cfg.MODEL.NUM_CLASSES == 3:
        weight=torch.Tensor([0.1, 0.1, 0.1])
    elif cfg.MODEL.NUM_CLASSES == 4:
        weight=torch.Tensor([0.1, 0.1, 0.1, 1.5])

    weight = weight.to("cuda")
    loss = nn.CrossEntropyLoss(weight=weight)
    optimizer = build_optim(cfg, model)
    scheduler = build_scheduler(cfg, len(trainloader), optimizer)
    # Load model checkpoint
    model, start_epoch, best_metric = load_checkpoint(args, model)
    start_epoch, best_metric = 0, 0

    # Run Script
    if mode == "train":
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHES):
            train_loss = train_model(
                cfg,
                epoch,
                model,
                trainloader,
                loss,
                scheduler,
                optimizer,
                scaler
            )
            best_metric = valid_model(
                cfg,
                mode,
                epoch,
                model,
                validloader,
                loss,
                best_metric=best_metric,
                save_prediction=False,
                visual=False
            )

    elif mode == "valid":
        valid_model(
            cfg, mode, 0, model, validloader, loss, best_metric=best_metric, save_prediction=True, visual=True
        )

if __name__ == "__main__":
    # Set up Variable
    seed = 10

    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    # print(cfg)

    # Set seed for reproducible result
    setup_determinism(seed)

    main(cfg, args)
