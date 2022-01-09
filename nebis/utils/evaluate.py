import numpy as np

from nebis.utils.metrics import (
    classification_metrics,
    classification_roc_auc,
    survival_metrics,
)


class ClassificationEvaluator:
    def __init__(self, Ys, Ps) -> None:
        self.Y = np.concatenate(Ys)
        self.Y_preds = np.concatenate([P[1]["label"] for P in Ps])
        self.Y_probs = np.concatenate([P[1]["predicted"] for P in Ps])

    def evaluate(self):
        metrics = classification_metrics(self.Y_preds, self.Y)
        fpr, tpr, roc_auc = classification_roc_auc(self.Y, self.Y_probs)
        return metrics


class SurvivalEvaluator:
    def __init__(self, Ys, Ps) -> None:
        self.Y_true_T = np.concatenate([Y[1] for Y in Ys])
        self.Y_true_E = np.concatenate([Y[0] for Y in Ys])
        self.Y_pred_r = [P[1]["risk"] for P in Ps]
        self.Y_pred_s = [P[1]["survival"] for P in Ps]
        self.time_points = Ys[0][3]

    def evaluate(self):
        return survival_metrics(
            self.Y_true_T, self.Y_true_E, self.Y_pred_r, self.Y_pred_s, self.time_points
        )


_evaluator_dict = {
    "survival": SurvivalEvaluator,
    "classification": ClassificationEvaluator,
}


def list_evaluators():
    return list(_evaluator_dict.keys())


def get_evaluator(name):
    try:
        if type(name) is str:
            return _evaluator_dict[name]
        else:
            return name
    except:
        raise ValueError("Could not retrieve evaluator for '{}'".format(name))
