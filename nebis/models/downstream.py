import torch
import torch.nn as nn

from nebis.models.losses import MTLR_survival_loss, cross_entropy_loss

class DownTaskSurvival(nn.Module):
    def __init__(
        self, time_num=256, embedding_size=512, p_dropout=0.05, hidden_size=256, device="CUDA"
    ):
        super(DownTaskSurvival, self).__init__()

        self.device = device
        self.embedding_size = embedding_size
        self.time_num = time_num
        self.p_dropout = p_dropout
        self.survival_loss = "MTLR"
        self.DropoutEmbed = nn.Dropout(self.p_dropout)
        self.LinearEmbed = nn.Linear(self.embedding_size, hidden_size)
        self.LinearHidden = nn.Linear(hidden_size, hidden_size)
        self.ActivationHidden = nn.Tanh()
        self.LinearSurvival = nn.Linear(hidden_size, self.time_num)

        if self.survival_loss == "MTLR":
            self.tri_matrix_1 = self.get_tri_matrix(dimension_type=1)
            self.tri_matrix_2 = self.get_tri_matrix(dimension_type=2)

        torch.nn.init.xavier_uniform_(self.LinearSurvival.weight)
        torch.nn.init.xavier_uniform_(self.LinearHidden.weight)

        self.loss_fct = MTLR_survival_loss

    def forward(self, x):
        x_a = self.DropoutEmbed(x)
        x_a = self.LinearEmbed(x_a)
        x_a = self.ActivationHidden(x)
        # x_a = self.LinearHidden(x_a)
        survival = self.LinearSurvival(x_a)

        return survival

    def get_tri_matrix(self, dimension_type=1):
        """
        Get tensor of the triangular matrix
        """
        if dimension_type == 1:
            ones_matrix = torch.ones(self.time_num, self.time_num + 1, device=self.device)
        else:
            ones_matrix = torch.ones(
                self.time_num + 1, self.time_num + 1, device=self.device
            )
        tri_matrix = torch.tril(ones_matrix)
        return tri_matrix

    def loss(self, y_pred, y_true, E, weight=None, reduction="mean"):
        return self.loss_fct(y_pred, y_true, E, self.tri_matrix_1, reduction)

    def predict_risk(self, y_out):
        """
        Predict the density, survival and hazard function, as well as the risk score
        """
        if self.survival_loss == "MTLR":
            phi = torch.exp(torch.mm(y_out, self.tri_matrix_1))
            div = torch.repeat_interleave(
                torch.sum(phi, 1).reshape(-1, 1), phi.shape[1], dim=1
            )

        density = phi / div
        survival = torch.mm(density, self.tri_matrix_2)
        hazard = density[:, :-1] / survival[:, 1:]

        cumulative_hazard = torch.cumsum(hazard, dim=1)
        risk = torch.sum(cumulative_hazard, 1)

        return {
            "density": density,
            "survival": survival,
            "hazard": hazard,
            "risk": risk,
        }


class DownTaskClassification(nn.Module):
    def __init__(
        self, num_labels=33, embedding_size=768, p_dropout=0.05, activation="relu", device="CUDA"
    ):
        super(DownTaskSurvival, self).__init__()
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.device = device

        self.DropoutDownTaskClassification = nn.Dropout(self.p_dropout)
        self.LinearDownTaskClassification = nn.Linear(self.embedding_size, self.num_labels, bias=False)
        self.SoftmaxDownTaskClassification = nn.Softmax(dim=1)
        if self.activation == "sigmoid":
            self.ActivationDownTaskClassification = nn.Sigmoid()
        elif self.activation == "relu":
            self.ActivationDownTaskClassification = nn.ReLU()

    def forward(self, x):
        x_embed = self.Dropout(x[:, 0, :])
        x_embed = self.ActivateFC(x_embed)
        logits = self.FC_classifier(x_embed)

        return logits

    def loss(self, logits, y, weight=None):
        return self.cross_entropy_loss(logits, y, weight)


_downstream_dict = {"survival": DownTaskSurvival, "classification": DownTaskClassification}

def get_downstream(name):
    try:
        if type(name) is str:
            return _downstream_dict[name]
        else:
            return name
    except:
        raise ValueError("Could not retrieve downstream method '{}'".format(name))