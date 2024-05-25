from sklearn import preprocessing
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import classification_report

from utils import AverageMeter
from itertools import cycle
from utils import save_checkpoint
from sklearn.metrics import roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt

target_names_dict = {"Non": 0, "Venous": 1, "Arterial": 2, "Others": 3}
map_id_name = {0: "Non Contrast", 1: "Venous", 2: "Arterial", 3: "Others"}

def valid_model(
    cfg,
    mode,
    epoch,
    model,
    dataloader,
    criterion,
    writer=None,
    save_prediction=True,
    best_metric=None,
    visual=False
):
    """Evaluate model performance on Validating dataset

    Args:
        cfg (CfgNode): Config object containing running configuration
        mode (str): Model running mode (valid/test)
        model (nn.Module): Model that need to have performance evaluated
        dataloader (data.DataLoader): Dataloader object to load data batch-wise
        criterion: Loss function
        writer (Summarywriter): Logger that log validation loss and plot it on Tensorboard
        save_prediction (Boolean): Whether to save prediction output or not (for bootstraping)
        best_metric (float, optional): Best performance result of loaded model. Defaults to None.
    """

    # Declare variables
    gpu = cfg.SYSTEM.GPU
    output_log_dir = cfg.DIRS.OUTPUTS
    model.eval()
    model.cuda()
    losses = AverageMeter()
    tbar = tqdm(dataloader)
    targets, preds, slice_dirs, study_ids, scan_ids = (
        list(),
        list(),
        list(),
        list(),
        list(),
    )
    data = dict()

    total_time = 0
    all_probs = []
    for i, (slice_dir, study_id, scan_id, image, target) in enumerate(tbar):
        with torch.no_grad():
            image = image.float()
            if gpu:
                image, target = image.cuda(), target.cuda()
                start = time.time()
                output = model(image)
                end = time.time()

            sigmoid = nn.Sigmoid()
            probs = sigmoid(output)
            pred = torch.argmax(probs, 1)
            probs = probs.cpu().numpy()
            all_probs.append(probs)
            total_time += end - start
 
            loss = criterion(output, target)

            losses.update(loss.item() * cfg.SOLVER.GD_STEPS, target.size(0))
            tbar.set_description("VALID: epoch: %d, valid loss: %.9f" % (epoch + 1, losses.avg))

            target = list(target.detach().cpu().numpy())
            pred = list(pred.detach().cpu().numpy())
            slice_dir = list(slice_dir)
            targets += target
            preds += pred
            slice_dirs += slice_dir
            study_ids += study_id
            scan_ids += list(np.array(scan_id))
    all_targets = []
    for idx in range(len(targets)):
        cur = [0] * 4
        cur[targets[idx]] = 1
        all_targets.append([cur])
    all_probs = np.concatenate(all_probs, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    
    if visual == True:
        plot_roc_curves(all_target, all_probs, map_id_name)

    # Calculate Metrics
    data, f1, f1_series = calculate_metrics(cfg, data, targets, preds, study_ids, scan_ids, slice_dirs)
    
    # save_dict = {
    #         "epoch": epoch + 1,
    #         "arch": cfg.NAME,
    #         "state_dict": model.state_dict(),
    #         "best_metric": best_metric,
    #     }
    # save_slice_dir = f"{cfg.NAME}_{str(f1)}_{str(f1_series)}.pth"
    
    # save_checkpoint(save_dict, root=cfg.DIRS.WEIGHTS, slice_dir=save_slice_dir)

    if mode == "train":
        # writer.add_scalars(
        #     f"Metrics",
        #     {
        #         "F1_SCORE": f1,
        #         "ACCURACY": accuracy,
        #         "RECALL": recall,
        #         "PRECISION": precision,
        #     },
        #     epoch,
        # )

        is_best = f1 > best_metric
        best_metric = max(f1, best_metric)
    
    if save_prediction:
        data.to_csv(f"eval_{mode}.csv", index=False)
    
    return best_metric

def plot_roc_curves(all_target, all_probs, map_id_name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(all_target[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red"])
    lw = 2
    plt.figure()
    for i, color in zip(range(4), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label=f"ROC curve of class {map_id_name[i]} (area = {roc_auc[i]})"
        )
        
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()

def calculate_metrics(cfg, data, labels, preds, study_ids, scan_ids, slice_dirs):
    # Calculate Metrics
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")
    print(
        "ACCURACY: %.9f, RECALL: %.9f, PRECISION: %.9f, F1: %.9f"
        % (accuracy, recall, precision, f1)
    )

    report = classification_report(
        labels,
        preds,
        target_names=["Non", "Venous", "Arterial", "Others"],
        digits=4,
    )
    print(report)

    data["study_ids"] = study_ids
    data["slice_dir"] = slice_dirs
    data["scan_ids"] = scan_ids
    data["preds"] = preds
    data["labels"] = labels
    data = pd.DataFrame(data)
    all_series = []
    for (studyuid, seriesuid), tmp_df in data.groupby(['study_id', 'scan_id']):
        preds = tmp_df['preds'].tolist()
        labels = tmp_df['labels'].tolist()
        f1_series = f1_score(labels, preds, average='weighted')
        all_series.append(f1_series)
    all_series = np.array(all_series)
    f1_series = np.mean(all_series)
    # print("series", f1_series)
    
    return data, f1, f1_series

