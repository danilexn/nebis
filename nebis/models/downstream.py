import torch
import torch.nn as nn

from nebis.models.losses import MTLR_survival_loss, cross_entropy_loss


class DownTaskSurvival(nn.Module):
    def __init__(self, config):
        super(DownTaskSurvival, self).__init__()

        self.config = config
        self.DropoutEmbed = nn.Dropout(self.config.p_dropout)
        self.LinearEmbed = nn.Linear(
            self.config.embedding_size, self.config.hidden_size
        )
        self.LinearHidden = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        if self.config.activation == "sigmoid":
            self.ActivationDownTaskSurvival = nn.Sigmoid()
        elif self.config.activation == "relu":
            self.ActivationDownTaskSurvival = nn.ReLU()
        elif self.config.activation == "tanh":
            self.ActivationDownTaskSurvival = nn.Tanh()

        self.LinearSurvival = nn.Linear(self.config.hidden_size, self.config.num_times)

        # TODO: include the option for more types of survival
        self.survival_loss = "MTLR"
        if self.survival_loss == "MTLR":
            self.tri_matrix_1 = self.get_tri_matrix(dimension_type=1)
            self.tri_matrix_2 = self.get_tri_matrix(dimension_type=2)

        torch.nn.init.xavier_uniform_(self.LinearSurvival.weight)
        torch.nn.init.xavier_uniform_(self.LinearHidden.weight)

        self.loss_fct = MTLR_survival_loss

    def forward(self, x):
        x_a = self.DropoutEmbed(x)
        x_a = self.LinearEmbed(x_a)
        x_a = self.ActivationDownTaskSurvival(x_a)
        survival = self.LinearSurvival(x_a)

        return survival

    # TODO: uncouple from the downstream, include inside loss function
    def get_tri_matrix(self, dimension_type=1):
        """
        Get tensor of the triangular matrix
        """
        if dimension_type == 1:
            ones_matrix = torch.ones(
                self.config.num_times,
                self.config.num_times + 1,
                device=self.config.device,
            )
        else:
            ones_matrix = torch.ones(
                self.config.num_times + 1,
                self.config.num_times + 1,
                device=self.config.device,
            )
        tri_matrix = torch.tril(ones_matrix)
        return tri_matrix

    def loss(self, pred, target, weight=None, reduction="mean"):
        E, y_time, y_onehot_time, time_num = target
        pred = pred.view(-1, self.config.num_times)
        return self.loss_fct(pred, y_onehot_time, E, self.tri_matrix_1, reduction)

    def prediction(self, y=None):
        """
        Predict the density, survival and hazard function, as well as the risk score
        """
        y = y.view(-1, self.config.num_times)
        if self.survival_loss == "MTLR":
            phi = torch.exp(torch.mm(y, self.tri_matrix_1))
            div = torch.repeat_interleave(
                torch.sum(phi, 1).reshape(-1, 1), phi.shape[1], dim=1
            )

        density = phi / div
        survival = torch.mm(density, self.tri_matrix_2)
        hazard = density[:, :-1] / survival[:, 1:]

        cumulative_hazard = torch.cumsum(hazard, dim=1)
        risk = torch.sum(cumulative_hazard, 1)

        return {
            "predicted": y.cpu().numpy(),
            "density": density.cpu().numpy(),
            "survival": survival.cpu().numpy(),
            "hazard": hazard.cpu().numpy(),
            "risk": risk.cpu().numpy(),
        }


class DownTaskClassification(nn.Module):
    def __init__(self, config):
        super(DownTaskClassification, self).__init__()
        self.config = config

        self.DropoutDownTaskClassification = nn.Dropout(self.config.p_dropout)
        self.LinearDownTaskClassification = nn.Linear(
            self.config.embedding_size, self.config.num_classes, bias=False
        )
        self.SoftmaxDownTaskClassification = nn.Softmax(dim=1)
        if self.config.activation == "sigmoid":
            self.ActivationDownTaskClassification = nn.Sigmoid()
        elif self.config.activation == "relu":
            self.ActivationDownTaskClassification = nn.ReLU()

    def forward(self, x):
        x_embed = self.DropoutDownTaskClassification(x[:, 0, :])
        x_embed = self.ActivationDownTaskClassification(x_embed)
        logits = self.LinearDownTaskClassification(x_embed)

        return logits

    def prediction(self, y=None):
        return {
            "predicted": y.detach().cpu().numpy(),
            "label": self.SoftmaxDownTaskClassification(y).detach().cpu().numpy(),
        }

    def loss(self, pred, target, weight=None):
        target = target[0] if type(target) is list else target
        target = target.view(-1)
        pred = pred.view(-1, self.config.num_classes)
        return cross_entropy_loss(pred, target, weight)


_downstream_dict = {
    "survival": DownTaskSurvival,
    "classification": DownTaskClassification,
}


def list_downstream():
    return list(_downstream_dict.keys())


def get_downstream(name):
    try:
        if type(name) is str:
            return _downstream_dict[name]
        else:
            return name
    except:
        raise ValueError("Could not retrieve downstream method '{}'".format(name))
