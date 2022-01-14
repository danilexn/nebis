# coding: utf-8
import torch
import torch.nn as nn

from nebis.models.base import Base
from nebis.models import SetQuence
from nebis.models.downstream import get_downstream
from nebis.models.pooling import get_pooler


class SetOmic(SetQuence):
    def __init__(self, config, BERT=None):
        super().__init__(config, BERT=BERT)
        self.OmicNorm = nn.LayerNorm(self.config.embedding_size)
        self.OmicEmbedding = nn.Embedding(
            self.config.max_numeric + 2, self.config.embedding_size, padding_idx=0
        )
        self.OmicLevelEmbedding = nn.Embedding(
            self.config.digitize_bins + 2, self.config.embedding_size, padding_idx=0
        )
        self.OmicPosIdentifiers = nn.Parameter(
            torch.arange(1, self.config.max_numeric + 2, 1), requires_grad=False,
        )
        self.PoolerMutome = get_pooler(self.config.pooling_sequence)(self.config)
        self.PoolerOmics = get_pooler(self.config.pooling_numeric)(self.config)

    def forward(self, X_mutome=None, X_omics=None):
        batch_size = X_mutome.shape[0]
        pooled_out = self.forward_BERT(X_mutome=X_mutome, X_omics=X_omics)

        # Do the omics
        X_pos_omic_embed = self.OmicEmbedding(
            self.OmicPosIdentifiers.repeat(batch_size, 1)
        )
        X_lev_omic_embed = self.OmicLevelEmbedding(X_omics)
        X_omics = X_pos_omic_embed[:, 1:, :] + X_lev_omic_embed

        # Change dimensionality of BERT output
        X_mutome = pooled_out.view(batch_size, -1, self.config.embedding_size)

        # Input is the encoding of BERT
        H_mutome = self.BertNorm(X_mutome)
        H_mutome = self.PoolerMutome(H_mutome)

        H_omics = self.OmicNorm(X_omics)
        H_omics = self.PoolerOmics(H_omics)

        H = H_mutome[:, 0, :] + H_omics[:, 0, :]
        H = H.view(-1, 1, self.config.embedding_size)
        Y = self.Downstream(H)

        # (downstream output, embeddings)
        return Y, H


class SetOnlyOmic(Base):
    def __init__(self, config, BERT=None):
        super().__init__(config)
        self.OmicNorm = nn.LayerNorm(self.config.embedding_size)
        self.OmicEmbedding = nn.Embedding(
            self.config.max_numeric + 2, self.config.embedding_size, padding_idx=0
        )
        self.OmicLevelEmbedding = nn.Embedding(
            self.config.digitize_bins + 2, self.config.embedding_size, padding_idx=0
        )
        self.OmicPosIdentifiers = nn.Parameter(
            torch.arange(1, self.config.max_numeric + 2, 1), requires_grad=False,
        )

        self.PoolerOmics = get_pooler(self.config.pooling_numeric)(self.config)

    def forward(self, X_mutome=None, X_omics=None):
        batch_size = X_omics.shape[0]

        # Do the omics
        X_pos_omic_embed = self.OmicEmbedding(
            self.OmicPosIdentifiers.repeat(batch_size, 1)
        )
        X_lev_omic_embed = self.OmicLevelEmbedding(X_omics)
        X_omics = X_pos_omic_embed[:, 1:, :] + X_lev_omic_embed

        H_omics = self.OmicNorm(X_omics)
        H_omics = self.PoolerOmics(H_omics)

        H = H_omics[:, 0, :]
        H = H.view(-1, 1, self.config.embedding_size)
        Y = self.Downstream(H)

        # (downstream output, embeddings)
        return Y, H


class ConsensusPooler(Base):
    def __init__(self, config):
        super().__init__(config)
        self.BertNorm = nn.LayerNorm(self.config.embedding_size)
        self.OmicNorm = nn.LayerNorm(self.config.embedding_size)
        self.PoolerMutome = get_pooler(self.config.pooling_sequence)(self.config)
        self.PoolerOmics = get_pooler(self.config.pooling_numeric)(self.config)
        self.Downstream = get_downstream(self.config.downstream)(self.config)

    def forward(self, X_mutome=None, X_omics=None):
        H_mutome = self.BertNorm(X_mutome)
        H_mutome = self.PoolerMutome(H_mutome)

        H_omics = self.OmicNorm(X_omics)
        H_omics = self.PoolerOmics(H_omics)

        H = H_mutome[:, 0, :] + H_omics[:, 0, :]
        H = H.view(-1, 1, self.config.embedding_size)
        Y = self.Downstream(H)

        # (downstream output, embeddings)
        return [Y, H]


class SetOmicConsensus(SetOmic):
    def __init__(self, config, BERT=None):
        super(SetOmicConsensus).__init__(config, BERT=BERT)
        self.Pooling = nn.ModuleList(
            [ConsensusPooler(self.config) for i in range(self.config.consensus_size)]
        )

    def forward(self, X_mutome=None, X_omics=None):
        batch_size = X_mutome.shape[0]
        pooled_out = self.forward_BERT(X_mutome=X_mutome, X_omics=X_omics)

        # Change dimensionality of BERT output
        X_mutome = pooled_out.view(batch_size, -1, self.config.embedding_size)

        # Do the omics
        X_pos_omic_embed = self.OmicEmbedding(
            self.OmicPosIdentifiers.repeat(batch_size, 1)
        )
        X_lev_omic_embed = self.OmicLevelEmbedding(X_omics)
        X_omics = X_pos_omic_embed[:, 1:, :] + X_lev_omic_embed

        Ps = [pooler(X_mutome, X_omics) for pooler in self.Pooling]

        # Consensus by sum of logits
        Y = torch.cat([P[0] for P in Ps], dim=0).sum(0).view(batch_size, -1)
        H = torch.cat([P[1] for P in Ps], dim=0).sum(0).view(batch_size, -1)

        # (downstream output, embeddings)
        return Y, H
