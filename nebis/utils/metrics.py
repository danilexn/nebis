# coding: utf-8

import numpy as np

from sklearn.metrics import precision_recall_fscore_support
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score


def classification_metrics(y, y_pred):
    calc_metrics = {}

    (
        calc_metrics["precision"],
        calc_metrics["recall"],
        calc_metrics["f-score"],
        _,
    ) = precision_recall_fscore_support(y, y_pred, average="macro")

    if len(np.unique(y)) == 2:
        fpr, tpr, _ = metrics.roc_curve(y, y_pred)
        calc_metrics["auc"] = metrics.auc(fpr, tpr)

    calc_metrics["acc"] = metrics.accuracy_score(y, y_pred)

    for metric in calc_metrics:
        print("{}={}".format(metric, calc_metrics[metric]))

    return calc_metrics


def classification_roc_auc(y, y_probs):
    n_classes = np.unique(y).shape[0]
    y_test = label_binarize(y, classes=list(range(n_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print("micro-ROC-AUC={}".format(roc_auc["micro"]))

    return fpr, tpr, roc_auc


def survival_metrics(y_true_T, y_true_E, y_pred_risk, y_pred_survival, time_points):
    try:
        _c_index = c_index(y_true_T, y_true_E, y_pred_risk)
    except:
        _c_index = -1
        raise ValueError(
            "NaNs detected in input when calculating integrated brier score."
        )

    try:
        _ibs = ibs(y_true_T, y_true_E, y_pred_survival, time_points)
    except:
        _ibs = -1
        raise ValueError(
            "NaNs detected in input when calculating integrated brier score."
        )

    metrics = {"c-index": _c_index, "ibs": _ibs}

    for m in metrics:
        print("{}={}".format(m, metrics[m]))

    return metrics


def c_index(true_T, true_E, pred_risk, include_ties=True):
    """
    Calculate c-index for survival prediction downstream task
    """
    # Ordering true_T, true_E and pred_score in descending order according to true_T
    order = np.argsort(-true_T)

    true_T = true_T[order]
    true_E = true_E[order]
    pred_risk = pred_risk[order]

    # Calculating the c-index
    # result = concordance_index(true_T, -pred_risk, true_E)
    # result = _concordance_index(pred_risk, true_T, true_E, include_ties)[0]
    result = concordance_index_censored(true_E.astype(bool), true_T, pred_risk)[0]

    return result


def ibs(true_T, true_E, pred_survival, time_points):
    """
    Calculate integrated brier score for survival prediction downstream task
    """
    true_E_bool = true_E.astype(bool)
    true = np.array(
        [(true_E_bool[i], true_T[i]) for i in range(len(true_E))],
        dtype=[("event", np.bool_), ("time", np.float32)],
    )

    # time points must be within the range of T
    min_T = true_T.min()
    max_T = true_T.max()
    valid_index = []
    for i in range(len(time_points)):
        if min_T <= time_points[i] <= max_T:
            valid_index.append(i)
    time_points = time_points[valid_index]
    pred_survival = pred_survival[:, valid_index]

    result = integrated_brier_score(true, true, pred_survival, time_points)

    return result
