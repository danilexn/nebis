# coding: utf-8

import numpy as np

from sklearn.metrics import precision_recall_fscore_support
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


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
    
    return fpr, tpr, roc_auc