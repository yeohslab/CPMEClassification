"""Training helpers for char-CNN models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from data.dataset import get_project_root
from models.simple_classifier import CharVocab


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(device)
        print(f"[device] cuda ({name})", flush=True)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] mps", flush=True)
    else:
        device = torch.device("cpu")
        print("[device] cpu", flush=True)
    return device


def save_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


class TrainProgress:
    def __init__(self, total: int, desc: str = "train", log_every: int = 20):
        self.total = max(total, 1)
        self.desc = desc
        self.log_every = max(1, log_every)
        self.step = 0
        self.start = time.time()

    def update(self, loss: float | None = None) -> None:
        self.step += 1
        if self.step == 1 or self.step % self.log_every == 0 or self.step == self.total:
            elapsed = time.time() - self.start
            pct = 100.0 * self.step / self.total
            eta = (elapsed / self.step) * (self.total - self.step) if self.step else 0
            loss_str = f" loss={loss:.4f}" if loss is not None else ""
            bar_len = 24
            filled = int(bar_len * self.step / self.total)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(
                f"[{self.desc}] [{bar}] {self.step}/{self.total} "
                f"({pct:.1f}%) elapsed={elapsed:.0f}s eta={eta:.0f}s{loss_str}",
                flush=True,
            )


def collate_emotion(batch, vocab: CharVocab):
    ids = torch.tensor([vocab.encode(b["text"]) for b in batch], dtype=torch.long)
    labels = torch.stack([b["labels"] for b in batch])
    return ids, labels


def collate_mbti(batch, vocab: CharVocab, max_posts: int = 8):
    batch_ids = []
    post_masks = []
    mbti_ids = []
    for b in batch:
        posts = list(b["texts"])
        n = min(len(posts), max_posts)
        if n < max_posts:
            posts = posts[:n] + [""] * (max_posts - n)
        else:
            posts = posts[:max_posts]
            n = max_posts
        batch_ids.append([vocab.encode(p) for p in posts])
        pm = torch.zeros(max_posts, dtype=torch.float)
        pm[:n] = 1.0
        post_masks.append(pm)
        mbti_ids.append(b["mbti_id"])
    return (
        torch.tensor(batch_ids, dtype=torch.long),
        torch.stack(post_masks),
        torch.tensor(mbti_ids, dtype=torch.long),
    )


def checkpoint_dir(name: str) -> Path:
    p = get_project_root() / "models" / "checkpoints" / name
    p.mkdir(parents=True, exist_ok=True)
    return p
