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
        max_len=1024,
        dropout=0.25,
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

        # cross attention for title over abstract
        self.cross_attention = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout, batch_first=True)

        # need to multiple dim_model by 2 if doing plain separate encodings (w/o cross attention)
        self.classifier = nn.Linear(dim_model, num_classes)

    # If doing joint encoding, change arguments to: self, input_ids, attention_mask
    def forward(self, title_ids, attention_mask_title, abstract_ids, attention_mask_abstract):
        # Uncomment if you want to do joint encodings
        """batch_size, seq_len = input_ids.size()
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
        return logits"""

        # Encoding the titles
        batch_size, title_length = title_ids.size()
        pos_ids_title = (torch.arange(title_length, device=title_ids.device).unsqueeze(0).expand(batch_size, title_length))
        pos_ids_title = pos_ids_title.clamp(0, self.pos_embedding.num_embeddings - 1)

        title_x = self.token_embedding(title_ids) + self.pos_embedding(pos_ids_title)
        title_mask = attention_mask_title == 0
        title_encoded = self.transformer_encoder(title_x, src_key_padding_mask=title_mask)

        # Encoding the abstract
        batch_size, abstract_len = abstract_ids.size()
        pos_ids_abstract = (torch.arange(abstract_len, device=abstract_ids.device).unsqueeze(0).expand(batch_size, abstract_len))
        pos_ids_abstract = pos_ids_abstract.clamp(0, self.pos_embedding.num_embeddings - 1)

        abstract_x = self.token_embedding(abstract_ids) + self.pos_embedding(pos_ids_abstract)
        abstract_mask = attention_mask_abstract == 0
        abstract_encoded = self.transformer_encoder(abstract_x, src_key_padding_mask=abstract_mask)

        # use this for when cross attention is not used
        # title_encoded = title_encoded[:, 0]
        # abstract_encoded = abstract_encoded[:, 0]
        # combined = torch.cat([title_encoded, abstract_encoded], dim=1)
        # logits = self.classifier(combined)

        # use this for when cross attention is used
        cross_output, _ = self.cross_attention(title_encoded, abstract_encoded, abstract_encoded, key_padding_mask=abstract_mask)
        cross_output = cross_output[:, 0]
        logits = self.classifier(cross_output)

        return logits
