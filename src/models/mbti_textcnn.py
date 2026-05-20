"""Stronger MBTI classifier: multi-kernel TextCNN + post attention (no HuggingFace)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.simple_classifier import PAD_ID


class TextCNNEncoder(nn.Module):
    """Kim-style TextCNN on char sequences."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = num_filters * len(kernel_sizes)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        # (N, seq)
        x = self.embedding(input_ids).transpose(1, 2)
        parts = []
        for conv in self.convs:
            c = F.relu(conv(x))
            parts.append(c.max(dim=2).values)
        return self.dropout(torch.cat(parts, dim=1))


class MbtiTextCNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 16,
        embed_dim: int = 256,
        num_filters: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.post_encoder = TextCNNEncoder(
            vocab_size, embed_dim=embed_dim, num_filters=num_filters, dropout=dropout
        )
        dim = self.post_encoder.out_dim
        self.post_attn = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        post_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 3:
            h = self.post_encoder.encode(input_ids)
            return self.head(self.dropout(h))

        b, p, s = input_ids.shape
        flat = input_ids.view(b * p, s)
        h = self.post_encoder.encode(flat).view(b, p, -1)

        scores = self.post_attn(h).squeeze(-1)
        if post_mask is not None:
            scores = scores.masked_fill(post_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * weights).sum(dim=1)
        return self.head(self.dropout(pooled))
