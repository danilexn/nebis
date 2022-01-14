import torch
import torch.nn as nn

from nebis.models.attention import MAB


class BasePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


class PoolingMax(BasePooler):
    def forward(self, X):
        batch_size = X.shape[0]
        return torch.max(X, dim=1)[0].view(batch_size, -1, self.config.embedding_size)


class PoolingMin(BasePooler):
    def forward(self, X):
        batch_size = X.shape[0]
        return torch.min(X, dim=1)[0].view(batch_size, -1, self.config.embedding_size)


class PoolingMean(BasePooler):
    def forward(self, X):
        batch_size = X.shape[0]
        return torch.mean(X, dim=1)[0].view(batch_size, -1, self.config.embedding_size)


class PoolingPMA(BasePooler):
    def __init__(self, config):
        super().__init__(config)

        self.S = nn.Parameter(
            torch.Tensor(1, self.config.k_seeds, self.config.embedding_size)
        )

        self.rFF_S_mutome = nn.Linear(
            self.config.embedding_size, self.config.embedding_size, bias=False
        )

        self.MutomePMA = MAB(
            self.config.embedding_size,
            self.config.mutome_heads,
            batch_first=True,
            dropout=self.config.p_dropout,
        )

        nn.init.xavier_uniform_(self.S)

    def forward(self, X):
        batch_size = X.shape[0]
        S = self.S.repeat(batch_size, 1, 1)
        Y = self.MutomePMA(self.rFF_S_mutome(S), X)
        return Y


class PoolingISAB(BasePooler):
    def __init__(self, config):
        super().__init__(config)

        self.S = nn.Parameter(
            torch.Tensor(1, self.config.k_seeds, self.config.embedding_size)
        )
        self.I = nn.Parameter(
            torch.Tensor(1, self.config.k_identity, self.config.embedding_size)
        )

        self.rFF_S_mutome = nn.Linear(
            self.config.embedding_size, self.config.embedding_size, bias=False
        )
        self.rFF_I_mutome = nn.Linear(
            self.config.embedding_size, self.config.embedding_size, bias=False
        )

        self.InducingMAB = MAB(
            self.config.embedding_size,
            self.config.mutome_heads,
            batch_first=True,
            dropout=self.config.p_dropout,
        )

        self.MutomeMAB = MAB(
            self.config.embedding_size,
            self.config.mutome_heads,
            batch_first=True,
            dropout=self.config.p_dropout,
        )
        self.MutomePMA = MAB(
            self.config.embedding_size,
            self.config.mutome_heads,
            batch_first=True,
            dropout=self.config.p_dropout,
        )

        nn.init.xavier_uniform_(self.S)
        nn.init.xavier_uniform_(self.I)

    def forward(self, X):
        batch_size = X.shape[0]
        I = self.I.repeat(batch_size, 1, 1)

        H = self.InducingMAB(self.rFF_I_mutome(I), X,)

        # Calculate the self-attention in
        # sub-quadratic time
        Z = self.MutomeMAB(X, H,)

        # Performs the multi-head dot-product
        # attention pooling
        S = self.S.repeat(batch_size, 1, 1)
        Y = self.MutomePMA(self.rFF_S_mutome(S), Z)
        return Y


class PoolingMAB(BasePooler):
    def __init__(self, config):
        super().__init__(config)

        self.S = nn.Parameter(
            torch.Tensor(1, self.config.k_seeds, self.config.embedding_size)
        )

        self.rFF_S_mutome = nn.Linear(
            self.config.embedding_size, self.config.embedding_size, bias=False
        )

        self.MutomeMAB = MAB(
            self.config.embedding_size,
            self.config.mutome_heads,
            batch_first=True,
            dropout=self.config.p_dropout,
        )
        self.MutomePMA = MAB(
            self.config.embedding_size,
            self.config.mutome_heads,
            batch_first=True,
            dropout=self.config.p_dropout,
        )

        nn.init.xavier_uniform_(self.S)

    def forward(self, X):
        batch_size = X.shape[0]
        X = torch.cat([self.S.repeat(batch_size, 1, 1), X], dim=1)
        Y = self.MutomePMA(X, X)
        return Y


class PoolingRNN(BasePooler):
    def __init__(self, config):
        super().__init__(config)

        self.LayerRNN = nn.LSTM(
            input_size=self.config.embedding_size,
            hidden_size=self.config.embedding_size,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
            dropout=self.config.p_dropout,
        )


class PoolingRNNsum(PoolingRNN):
    def forward(self, X):
        batch_size = X.shape[0]
        _, (ht_m, _) = self.LayerRNN(X)
        Y = ht_m.squeeze(0).sum(dim=1).view(batch_size, -1, self.config.embedding_size)
        return Y


class PoolingRNNmean(PoolingRNN):
    def forward(self, X):
        batch_size = X.shape[0]
        _, (ht_m, _) = self.rnn_mutome(X)
        Y = ht_m.squeeze(0).mean(dim=1).view(batch_size, -1, self.config.embedding_size)
        return Y


class PoolingRNNPMA(PoolingRNN):
    def __init__(self, config):
        super().__init__(config)

        self.S = nn.Parameter(
            torch.Tensor(1, self.config.k_seeds, self.config.embedding_size)
        )
        self.rFF_S_mutome = nn.Linear(
            self.config.embedding_size, self.config.embedding_size, bias=False
        )

        self.MutomePMA = MAB(
            self.config.embedding_size,
            self.config.mutome_heads,
            batch_first=True,
            dropout=self.config.p_dropout,
        )

        nn.init.xavier_uniform_(self.S)

    def forward(self, X):
        batch_size = X.shape[0]
        _, (ht_m, _) = self.rnn_mutome(X)

        S = self.S.repeat(batch_size, 1, 1)
        Y = self.MutomePMA(S, ht_m)
        return Y


_pooling_dict = {
    "max": PoolingMax,
    "min": PoolingMin,
    "mean": PoolingMean,
    "PMA": PoolingPMA,
    "ISAB": PoolingISAB,
    "MAB": PoolingMAB,
    "RNN_sum": PoolingRNNsum,
    "RNN_mean": PoolingRNNmean,
    "RNN_PMA": PoolingRNNPMA,
}


def list_pooler():
    return list(_pooling_dict.keys())


def get_pooler(name):
    try:
        if type(name) is str:
            return _pooling_dict[name]
        else:
            return name
    except:
        raise ValueError("Could not retrieve pooling method '{}'".format(name))
