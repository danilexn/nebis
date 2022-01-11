import torch
from nebis.models import SetOmic


class SetOmicExplainer(SetOmic):
    def forward(self, X_mutome=None, X_omics=None):
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

        H_omics = self.OmicNorm(X_omics)
        H_omics = self.PoolerOmics(H_omics)

        H = H_mutome[:, 0, :] + H_omics[:, 0, :]
        H = H.view(-1, 1, self.config.embedding_size)
        Y = self.Downstream(H)

        # (downstream output, embeddings)
        return Y, H

    def forward_embeds(self, X_mutome=None, X_omics=None):
        input_ids = X_mutome.view(-1, self.config.sequence_length)
        attention_mask = torch.where(input_ids > 0, 1, 0).to(self.config.device)
        batch_size = int(
            input_ids.flatten().shape[0]
            / (self.config.sequence_length * self.config.max_mutations)
        )

        with torch.no_grad():
            X_seq_muto_embed = self.BERT.embeddings(
                input_ids, token_type_ids=None, position_ids=None
            )
            X_seq_muto_embed = X_seq_muto_embed.view(
                batch_size, -1, self.config.embedding_size
            )

            X_lev_omic_embed = self.OmicLevelEmbedding(X_omics)
            X_pos_omic_embed = self.OmicEmbedding(
                self.OmicPosIdentifiers.repeat(batch_size, 1)
            ).view(batch_size, -1, self.config.embedding_size)

        Pooler = self.PoolerOmics

        return ([X_seq_muto_embed, X_lev_omic_embed], [X_pos_omic_embed, Pooler])
