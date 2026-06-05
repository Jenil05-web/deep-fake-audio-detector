"""
Evaluation Metrics
===================
AUC, EER, F1, accuracy — everything needed to measure model quality.

Key metric for deepfake detection: EER (Equal Error Rate)
  - EER = point where False Acceptance Rate == False Rejection Rate
  - Lower is better. Random = 50%, good model < 10%, great model < 3%
"""

import numpy as np


def compute_auc(probs: list, labels: list) -> float:
    """
    Area Under ROC Curve via trapezoidal rule.
    probs:  list of fake probabilities [0, 1]
    labels: list of ground truth (0=real, 1=fake)
    """
    paired   = sorted(zip(probs, labels), reverse=True)
    n_pos    = sum(labels)
    n_neg    = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tp, fp   = 0, 0
    auc      = 0.0
    prev_fp  = 0

    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp

    return auc / (n_pos * n_neg)


def compute_eer(probs: list, labels: list) -> float:
    """
    Equal Error Rate — threshold where FAR == FRR.
    Lower = better detector.
    """
    thresholds = np.linspace(0, 1, 200)
    probs_arr  = np.array(probs)
    labels_arr = np.array(labels)

    n_pos = labels_arr.sum()
    n_neg = len(labels_arr) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    best_eer = 1.0
    for t in thresholds:
        preds = (probs_arr >= t).astype(int)
        fp    = ((preds == 1) & (labels_arr == 0)).sum()
        fn    = ((preds == 0) & (labels_arr == 1)).sum()
        far   = fp / max(n_neg, 1)
        frr   = fn / max(n_pos, 1)
        eer   = (far + frr) / 2
        if eer < best_eer:
            best_eer = eer

    return float(best_eer)


def compute_f1(probs: list, labels: list, threshold: float = 0.5) -> float:
    probs_arr  = np.array(probs)
    labels_arr = np.array(labels)
    preds      = (probs_arr >= threshold).astype(int)

    tp = ((preds == 1) & (labels_arr == 1)).sum()
    fp = ((preds == 1) & (labels_arr == 0)).sum()
    fn = ((preds == 0) & (labels_arr == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return float(f1)


def compute_all_metrics(probs: list, labels: list, threshold: float = 0.5) -> dict:
    return {
        "auc":      round(compute_auc(probs, labels),  4),
        "eer":      round(compute_eer(probs, labels),  4),
        "f1":       round(compute_f1(probs, labels, threshold), 4),
        "accuracy": round(sum((p >= threshold) == l for p, l in zip(probs, labels)) / max(len(labels), 1), 4),
    }