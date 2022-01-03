# coding: utf-8
import torch
import torch.nn as nn

from transformers import BertModel
from bideset.models.base import Base
from bideset.models.downstream import get_downstream
from bideset.models.pooling import get_pooler

class SetOmic(Base):
    def __init__(self, config):
        super().__init__(config)
        self.BERT = BertModel(self.config.bert_config)
        self.BertNorm = nn.LayerNorm(self.config.embedding_size)

        self.PoolerMutome = get_pooler(self.config.pooling_mutome)(self.config)
        self.PoolerOmics = get_pooler(self.config.pooling_omics)(self.config)
        self.Downstream = get_downstream(self.config.downstream)(self.config)
        self.loss = self.Downstream.loss

        # self.init_weights()

    def forward(self, X_mutome=None, X_omic=None):
        # Select only the sequences that are not fully padded, up to split
        input_ids = X_mutome.view(-1, self.sequence_length)
        attention_mask = torch.where(input_ids > 0, 1, 0).to(self.config.device)
        batch_size = int(
            input_ids.flatten().shape[0] / (self.sequence_length * self.split)
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
        X_lev_omic_embed = self.OmicLevelEmbedding(X_omic)
        X_omics = X_pos_omic_embed[:, 1:, :] + X_lev_omic_embed

        # Input is the encoding of BERT
        H_mutome = self.BertNorm(X_mutome)
        H_mutome = self.PoolerMutome(H_mutome)

        # TODO: normalise omics
        H_omics = self.PoolerOmics(X_omics)

        H = H_mutome[:, 0, :] + H_omics[:, 0, :]
        Y = self.Downstream(H)

        # (downstream output, embeddings)
        return Y, H