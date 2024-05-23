from tqdm import tqdm

import torch
from torch.cuda.amp import autocast

from utils import AverageMeter


def train_model(
    cfg, epoch, model, dataloader, criterion, scheduler, optimizer, scaler, writer=None
):
    """Run 1 training epoch

    Args:
        cfg (CfgNode): Config object containing training configuration
        epoch (int): Number of epoch to train
        model (nn.Module): Model object to be trained
        dataloader (data.DataLoader): Dataloader object to load training data batch-wise
        criterion: Loss function
        scheduler (_LRScheduler): Learning rate scheduler
        optimizer: Optimizer to optimize our model
        scaler (GradScaler): GradScaler for mixed precision training
        writer (SummaryWriter): Writer to keep log and show training loss on Tensorboard

    Returns:
        average_loss (float): Averaged loss over all iteration in one training epoch
    """

    # Declare variables
    print(f"\nEpoch: {epoch + 1}")
    gpu = cfg.SYSTEM.GPU
    losses = AverageMeter()
    model.train()
    model.cuda()

    tbar = tqdm(dataloader)
    for i, (filename, study_ID, seriesNumber, image, target) in enumerate(tbar):
        image = image.float()
        # print(torch.min(image), torch.max(image))
        # print(image.shape)
        if gpu:
            with autocast():
                # print(type(image))
                # print(type(target), target)
                image, target = image.cuda(), target.cuda()
                output = model(image)
                loss = criterion(output, target)

        # Optimizer Step
        scaler.scale(loss).backward()

        scheduler(optimizer, i, epoch)
        scaler.step(optimizer)
        # optimizer.step()
        optimizer.zero_grad()
        scaler.update()

        # Record loss
        losses.update(loss.item() * cfg.SOLVER.GD_STEPS, target.size(0))
        tbar.set_description(
            "TRAIN: epoch: %d, train loss: %.9f, learning rate: %.9f"
            % (epoch + 1, losses.avg, optimizer.param_groups[-1]["lr"])
        )
    # print("Train loss: %.9f, learning rate: %.6f" %
    #        (losses.avg, optimizer.param_groups[-1]['lr']))

    average_loss = losses.avg
    return average_loss
