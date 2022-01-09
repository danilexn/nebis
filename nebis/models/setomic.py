# coding: utf-8
import torch
import torch.nn as nn

from transformers import BertModel
from nebis.models.base import Base
from nebis.models.downstream import get_downstream
from nebis.models.pooling import get_pooler


class SetOmic(Base):
    def __init__(self, config, BERT=None):
        super().__init__(config)
        self.BERT = BertModel(self.config.bert_config) if BERT is None else BERT
        self.BertNorm = nn.LayerNorm(self.config.embedding_size)

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
        self.Downstream = get_downstream(self.config.downstream)(self.config)
        self.loss = self.Downstream.loss

        # self.init_weights()

    def forward(self, X_mutome=None, X_omics=None):
        # Select only the sequences that are not fully padded, up to split
        input_ids = X_mutome.view(-1, self.config.sequence_length)
        attention_mask = torch.where(input_ids > 0, 1, 0).to(self.config.device)
        batch_size = int(
            input_ids.flatten().shape[0]
            / (self.config.sequence_length * self.config.max_mutations)
        )

        with torch.no_grad():
            _, pooled_out = self.BERT(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
            )

        # Change dimensionality of BERT output
        X_mutome = pooled_out.view(batch_size, -1, self.config.embedding_size)

        # Do the omics
        X_pos_omic_embed = self.OmicEmbedding(
            self.OmicPosIdentifiers.repeat(batch_size, 1)
        )
        X_lev_omic_embed = self.OmicLevelEmbedding(X_omics)
        X_omics = X_pos_omic_embed[:, 1:, :] + X_lev_omic_embed

        # Input is the encoding of BERT
        H_mutome = self.BertNorm(X_mutome)
        H_mutome = self.PoolerMutome(H_mutome)

        # TODO: normalise omics
        H_omics = self.PoolerOmics(X_omics)

        H = H_mutome[:, 0, :] + H_omics[:, 0, :]
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


class SetOmicConsensus(Base):
    def __init__(self, config, BERT=None):
        super().__init__(config)
        self.BERT = BertModel(self.config.bert_config) if BERT is None else BERT

        self.OmicEmbedding = nn.Embedding(
            self.config.max_numeric + 2, self.config.embedding_size, padding_idx=0
        )
        self.OmicLevelEmbedding = nn.Embedding(
            self.config.digitize_bins + 2, self.config.embedding_size, padding_idx=0
        )
        self.OmicPosIdentifiers = nn.Parameter(
            torch.arange(1, self.config.max_numeric + 2, 1), requires_grad=False,
        )

        self.Pooling = nn.ModuleList(
            [ConsensusPooler(self.config) for i in range(self.config.consensus_size)]
        )
        self.Downstream = get_downstream(self.config.downstream)(self.config)
        self.loss = self.Downstream.loss

        # self.init_weights()

    def forward(self, X_mutome=None, X_omics=None):
        # Select only the sequences that are not fully padded, up to split
        input_ids = X_mutome.view(-1, self.config.sequence_length)
        attention_mask = torch.where(input_ids > 0, 1, 0).to(self.config.device)
        batch_size = int(
            input_ids.flatten().shape[0]
            / (self.config.sequence_length * self.config.max_mutations)
        )

        with torch.no_grad():
            _, pooled_out = self.BERT(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
            )

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
