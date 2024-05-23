import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from scipy import stats
# from statistics import mode
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import classification_report

from utils import save_checkpoint
from utils import AverageMeter

def valid_model_2d(cfg, epoch, model, validloader, criterion, voting, best_metric=None):
    losses = AverageMeter()
    model.eval()
    model.cuda()

    scan_labels, scan_preds = [], []

    tbar = tqdm(validloader)
    for scan, label in tbar:
        with torch.no_grad():
            scan = scan.float().cuda()
            scan = scan.view(-1, cfg.DATA.INP_CHANNEL, cfg.DATA.SIZE[0], cfg.DATA.SIZE[1])
            scan_output = model(scan)
            sigmoid = nn.Sigmoid()
            scan_probs = sigmoid(scan_output)
            scan_pred = torch.argmax(scan_probs, 1)
            if voting == "weighted":
                scan_score = scan_probs.max(dim=1)[0].detach().cpu().numpy()
            label_tensor = torch.full((32,), fill_value=label[0], dtype=torch.long).cuda()
            for j in range(1, len(label)):
                label_tensor = torch.cat([
                    label_tensor,
                    torch.full((32,), fill_value=label[j], dtype=torch.long).cuda()
                ])
            scan_loss = criterion(scan_output, label_tensor)

        if voting == "majority":
            scan_pred_list = []
            for j in range(len(label)):
                scan_pred_j = np.array([stats.mode(scan_pred[j*32:(j+1)*32].detach().cpu().numpy())[0][0]])
                scan_pred_list.append(scan_pred_j)
            scan_pred = np.concatenate(scan_pred_list)
        elif voting == "weighted":
            scan_pred_list = []
            for j in range(len(label)):
                scan_pred_j = np.array([round(np.average(scan_pred[j*32:(j+1)*32].detach().cpu().numpy(), weights=scan_score[j*32:(j+1)*32]))])
                scan_pred_list.append(scan_pred_j)
            scan_pred = np.concatenate(scan_pred_list)
        elif voting == "prob_sum":
            scan_pred_list = []
            for j in range(len(label)):
                scan_pred_j = np.array([np.argmax(np.sum(scan_probs[j*32:(j+1)*32].detach().cpu().numpy(), axis=0))])
                scan_pred_list.append(scan_pred_j)
            scan_pred = np.concatenate(scan_pred_list)
        elif voting == "average":
            scan_pred_list = []
            for j in range(len(label)):
                scan_pred_j = np.array([np.argmax(np.mean(scan_probs[j*32:(j+1)*32].detach().cpu().numpy(), axis=0))])
                scan_pred_list.append(scan_pred_j)
            scan_pred = np.concatenate(scan_pred_list)
        elif voting == "median":
            scan_pred_list = []
            for j in range(len(label)):
                scan_pred_j = np.array([np.argmax(np.median(scan_probs[j*32:(j+1)*32].detach().cpu().numpy(), axis=0))])
                scan_pred_list.append(scan_pred_j)
            scan_pred = np.concatenate(scan_pred_list)

        scan_preds.extend(scan_pred.tolist())
        scan_labels.extend(label.detach().cpu().numpy().tolist())
        losses.update(scan_loss.item(), label_tensor.size(0))
        
        tbar.set_description("VALID: %.9f" % (losses.avg))

    # Calculate Metrics
    accuracy = accuracy_score(scan_labels, scan_preds)
    recall = recall_score(scan_labels, scan_preds, average="weighted")
    precision = precision_score(scan_labels, scan_preds, average="weighted")
    f1 = f1_score(scan_labels, scan_preds, average="weighted")
    print(
        "VALID: ACCURACY: %.9f, RECALL: %.9f, PRECISION: %.9f, F1: %.9f"
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

    save_dict = {
            "epoch": epoch + 1,
            "arch": cfg.NAME,
            "state_dict": model.state_dict(),
            "best_metric": best_metric,
        }
    save_filename = f"{cfg.NAME}_{str(f1)}.pth"
    
    # save_checkpoint(save_dict, root=cfg.DIRS.WEIGHTS, filename=save_filename)

    best_metric = max(f1, best_metric)
        
    return best_metric


