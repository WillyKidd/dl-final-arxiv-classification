import torch
import torch.nn as nn


class ArxivClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        dim_model=128,
        num_heads=4,
        num_layers=2,
        dim_ffn=256,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim_model)
        self.pos_embedding = nn.Embedding(max_len, dim_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_ffn,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(dim_model, num_classes)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.size()
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        position_ids = position_ids.clamp(0, self.pos_embedding.num_embeddings - 1)

        x = self.token_embedding(input_ids) + self.pos_embedding(position_ids)

        key_padding_mask = attention_mask == 0

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        x = x[:, 0]

        logits = self.classifier(x)
        return logits
