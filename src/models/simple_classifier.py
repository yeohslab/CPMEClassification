"""Lightweight char-CNN text classifiers (no HuggingFace download)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn as nn

PAD_ID = 0
UNK_ID = 1


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


class CharVocab:
    def __init__(self, char2id: dict[str, int], max_length: int = 128):
        self.char2id = char2id
        self.id2char = {i: c for c, i in char2id.items()}
        self.max_length = max_length

    @classmethod
    def build(cls, texts: list[str], max_length: int = 128, min_freq: int = 2) -> CharVocab:
        freq: dict[str, int] = {}
        for text in texts:
            for ch in normalize_text(text):
                freq[ch] = freq.get(ch, 0) + 1
        char2id = {"<pad>": PAD_ID, "<unk>": UNK_ID}
        for ch, count in sorted(freq.items(), key=lambda x: -x[1]):
            if count >= min_freq and ch not in char2id:
                char2id[ch] = len(char2id)
        return cls(char2id, max_length)

    def encode(self, text: str) -> list[int]:
        ids = [
            self.char2id.get(ch, UNK_ID)
            for ch in normalize_text(text)[: self.max_length]
        ]
        if len(ids) < self.max_length:
            ids += [PAD_ID] * (self.max_length - len(ids))
        return ids

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"char2id": self.char2id, "max_length": self.max_length},
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: Path) -> CharVocab:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["char2id"], data["max_length"])


class CharCNNEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_filters: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq) or (batch, posts, seq)
        if input_ids.dim() == 3:
            b, p, s = input_ids.shape
            x = input_ids.view(b * p, s)
            h = self._encode(x).view(b, p, -1)
            mask = (input_ids != PAD_ID).any(dim=-1).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (h * mask).sum(dim=1) / denom
        return self._encode(input_ids)

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids).transpose(1, 2)
        x = self.act(self.conv(x))
        mask = (input_ids != PAD_ID).unsqueeze(1).float()
        x = x * mask
        denom = mask.sum(dim=2).clamp(min=1.0)
        return (x.sum(dim=2) / denom)


class SimpleEmotionModel(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int = 6):
        super().__init__()
        self.encoder = CharCNNEncoder(vocab_size)
        self.head = nn.Linear(128, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(input_ids))


