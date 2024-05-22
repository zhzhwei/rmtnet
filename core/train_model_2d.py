import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

from scipy import stats
# from statistics import mode
from torch.cuda.amp import autocast, GradScaler
from utils import AverageMeter
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

# writer = SummaryWriter(log_dir='./logs/example_2/tensorboard/')

def train_model_2d(cfg, epoch, model, trainloader, loss, scheduler, optimizer, scaler, num_epochs, accumulation_steps=2):
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # Declare variables
    print(f"\nEpoch: {epoch + 1}/{num_epochs}")
    losses = AverageMeter()
    model.train()
    model.cuda()

    scan_labels, scan_preds = [], []

    tbar = tqdm(trainloader)
    for i, (scan, label) in enumerate(tbar):
        optimizer.zero_grad()
        scan = scan.float().cuda()
        scan = scan.view(-1, cfg.DATA.INP_CHANNEL, cfg.DATA.SIZE[0], cfg.DATA.SIZE[1])
        with autocast():
            scan_output = model(scan)
            sigmoid = nn.Sigmoid()
            scan_probs = sigmoid(scan_output)
            scan_pred = torch.argmax(scan_probs, 1)
            label_tensor = torch.full((32,), fill_value=label[0], dtype=torch.long).cuda()
            for j in range(1, len(label)):
                label_tensor = torch.cat([
                    label_tensor,
                    torch.full((32,), fill_value=label[j], dtype=torch.long).cuda()
                ])
            scan_loss = loss(scan_output, label_tensor)
        
        # Normalize the loss to account for gradient accumulation
        scan_loss = scan_loss / accumulation_steps
        scaler.scale(scan_loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scheduler(optimizer, i, epoch)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scan_pred_list = []
        for j in range(len(label)):
            scan_pred_j = np.array([stats.mode(scan_pred[j*32:(j+1)*32].detach().cpu().numpy())[0][0]])
            scan_pred_list.append(scan_pred_j)
        scan_pred = np.concatenate(scan_pred_list)

        scan_preds.extend(scan_pred.tolist())
        scan_labels.extend(label.detach().cpu().numpy().tolist())
        losses.update(scan_loss.item() * accumulation_steps, label_tensor.size(0))
        
        tbar.set_description(
            "TRAIN: Epoch: %d, Train loss: %.9f, learning rate: %.9f"
            % (epoch + 1, losses.avg, optimizer.param_groups[-1]["lr"])
        )

        # Free up memory
        del scan, label, scan_output, scan_probs, scan_pred, label_tensor, scan_loss
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate Metrics
    accuracy = accuracy_score(scan_labels, scan_preds)
    recall = recall_score(scan_labels, scan_preds, average="weighted")
    precision = precision_score(scan_labels, scan_preds, average="weighted")
    f1 = f1_score(scan_labels, scan_preds, average="weighted")
    print(
        "TRAIN: ACCURACY: %.9f, RECALL: %.9f, PRECISION: %.9f, F1: %.9f"
        % (accuracy, recall, precision, f1)
    )

    if cfg.MODEL.NUM_CLASSES == 3:
        label_names=["Non", "Venous", "Aterial"]
    elif cfg.MODEL.NUM_CLASSES == 4:
        label_names=["Non", "Venous", "Aterial", "Others"]

    report = classification_report(
        scan_labels,
        scan_preds,
        target_names=label_names,
        digits=4
    )
    print(report)

    # writer.add_scalar('loss/train', losses.avg, epoch+1)

    average_loss = losses.avg
    return average_loss