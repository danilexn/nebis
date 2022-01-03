# coding: utf-8
import torch
import torch.nn as nn

from transformers import BertModel
from nebis.models.base import Base
from nebis.models.downstream import get_downstream
from nebis.models.pooling import get_pooler

class SetQuence(Base):
    def __init__(self, config):
        super().__init__(config)
        self.BERT = BertModel(self.config.bert_config)
        self.BertNorm = nn.LayerNorm(self.config.embedding_size)

        self.Pooling = get_pooler(self.config.pooling)(self.config)
        self.Downstream = get_downstream(self.config.downstream)(self.config)
        self.loss = self.Downstream.loss

        # self.init_weights()

    def forward(self, input_ids=None, attention_mask=None):
        # Select only the sequences that are not fully padded, up to split
        input_ids = input_ids.view(-1, self.config.sequence_length)
        attention_mask = torch.where(input_ids > 0, 1, 0)
        batch_size = int(
            input_ids.flatten().shape[0] / (self.config.sequence_length * self.config.split)
        )

        if self.config.fine_tune:
            _attention = (
                attention_mask.view(batch_size, self.config.split, self.config.sequence_length)
                .max(dim=2)[0]
                .view(batch_size, self.config.split, 1)
            )

            max_length = torch.where(_attention.flatten() == 0)[0]
            if max_length.nelement() == 0:
                max_length = self.config.split
            else:
                max_length = max_length[0] - 1

            if max_length > self.config.fine_tune_max_mutations:
                max_length = self.config.fine_tune_max_mutations

            # Apply filter to the input sequences and to the attention mask
            input_ids = input_ids[0:max_length, :]
            attention_mask = attention_mask[0:max_length, :]

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

        # Change dimensionality of BERT output
        pooled_output = pooled_out.view(batch_size, -1, self.config.embedding_size)

        # Input is the encoding of BERT
        H = self.BertNorm(pooled_output)
        H = self.Pooler(H)
        Y = self.Downstream(H)

        # (downstream output, embeddings)
        return Y, H