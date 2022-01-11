# coding: utf-8
from multiprocessing import Value
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from transformers import BertModel
from nebis.models.base import Base
from nebis.models.downstream import get_downstream
from nebis.models.pooling import get_pooler


class SetQuence(Base):
    def __init__(self, config, BERT=None):
        super().__init__(config)
        self.BERT = self.get_BERT_model(BERT)
        self.BertNorm = nn.LayerNorm(self.config.embedding_size)

        self.Pooling = get_pooler(self.config.pooling_sequence)(self.config)
        self.Downstream = get_downstream(self.config.downstream)(self.config)
        self.loss = self.Downstream.loss

        # self.init_weights()

    def get_BERT_model(self, BERT):
        if BERT is None:
            return BertModel(self.config.bert_config)
        elif isinstance(BERT, BertModel):
            return BERT
        elif isinstance(BERT, SetQuence):
            return BERT.BERT
        else:
            raise ValueError("Could not load a BERT model")

    def forward_BERT(self, X_mutome=None, X_omics=None):
        batch_size = X_mutome.shape[0]
        input_ids = X_mutome.view(batch_size, -1, self.config.sequence_length)

        if batch_size != 1 and X_mutome.shape[1] > self.config.max_mutations:
            input_ids = input_ids[:, 0 : self.config.max_mutations, :]
            input_ids = input_ids.view(-1, self.config.sequence_length)
            attention_mask = torch.where(input_ids > 0, 1, 0)
        elif batch_size == 1:
            input_ids = X_mutome.view(-1, self.config.sequence_length)
            attention_mask = torch.where(input_ids > 0, 1, 0)
            _attention = (
                attention_mask.view(batch_size, -1, self.config.sequence_length)
                .max(dim=2)[0]
                .view(batch_size, -1, 1)
            )

            max_length = torch.where(_attention.flatten() == 0)[0]
            if max_length.nelement() == 0:
                max_length = self.config.max_mutations
            else:
                max_length = max_length[0] - 1

            if max_length > self.config.max_mutations:
                max_length = self.config.max_mutations

            input_ids = input_ids[0:max_length, :]
            attention_mask = attention_mask[0:max_length, :]

        if self.config.finetune:
            _, pooled_out = self.BERT(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
            )
        else:
            with torch.no_grad():
                _, pooled_out = self.BERT(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                )

        return pooled_out

    def forward(self, X_mutome=None, X_omics=None):
        batch_size = X_mutome.shape[0]
        pooled_out = self.forward_BERT(X_mutome=X_mutome, X_omics=X_omics)

        # Change dimensionality of BERT output
        pooled_output = pooled_out.view(batch_size, -1, self.config.embedding_size)

        # Input is the encoding of BERT
        H = self.BertNorm(pooled_output)
        H = self.Pooling(H)
        Y = self.Downstream(H)

        # (downstream output, embeddings)
        return Y, H

    def explain(self):
        pass


class ConsensusPooler(Base):
    def __init__(self, config):
        super().__init__(config)
        self.BertNorm = nn.LayerNorm(self.config.embedding_size)
        self.Pooling = get_pooler(self.config.pooling_sequence)(self.config)
        self.Downstream = get_downstream(self.config.downstream)(self.config)

    def forward(self, X_mutome=None, X_omics=None):
        # Input is the encoding of BERT
        H = self.BertNorm(X_mutome)
        H = self.Pooling(H)
        Y = self.Downstream(H)

        # (downstream output, embeddings)
        return [Y, H]


class SetQuenceConsensus(SetQuence):
    def __init__(self, config, BERT=None):
        super(SetQuenceConsensus).__init__(config, BERT=BERT)
        self.Pooling = nn.ModuleList(
            [ConsensusPooler(self.config) for i in range(self.config.consensus_size)]
        )

    def forward(self, X_mutome=None, X_omics=None):
        # Select only the sequences that are not fully padded, up to split
        batch_size = X_mutome.shape[0]
        pooled_out = self.forward_BERT(X_mutome=X_mutome, X_omics=X_omics)

        # Change dimensionality of BERT output
        pooled_output = pooled_out.view(batch_size, -1, self.config.embedding_size)

        Ps = [pooler(pooled_output) for pooler in self.Pooling]

        # Consensus by sum of logits
        Y = torch.cat([P[0] for P in Ps], dim=0).mean(dim=0).view(batch_size, -1)
        H = torch.cat([P[1] for P in Ps], dim=0).mean(dim=0).view(batch_size, -1)

        # (downstream output, embeddings)
        return Y, H
