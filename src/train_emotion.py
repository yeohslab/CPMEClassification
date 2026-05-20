"""Train single-post emotion model (char-CNN)."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, mean_absolute_error
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import (
    EMOTION_LABELS,
    EmotionDataset,
    load_emotion_split,
    load_posts_dataframe,
)
from models.simple_classifier import CharVocab, SimpleEmotionModel
from training_utils import TrainProgress, checkpoint_dir, collate_emotion, get_device, save_config


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    mae = mean_absolute_error(labels, preds)
    pred_bin = (preds >= 0.5).astype(int)
    label_bin = (labels >= 0.5).astype(int)
    macro_f1 = f1_score(label_bin, pred_bin, average="macro", zero_division=0)
    dom_acc = (labels.argmax(1) == preds.argmax(1)).mean()
    return {"mae": float(mae), "macro_f1": float(macro_f1), "dominant_acc": float(dom_acc)}


@torch.no_grad()
def eval_emotion_simple(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())
    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return total_loss / max(len(loader), 1), preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    df = load_posts_dataframe()
    split = load_emotion_split()
    train_ds = EmotionDataset(split["train_indices"], df)
    val_ds = EmotionDataset(split["val_indices"], df)

    texts = [df.iloc[i]["post"] for i in split["train_indices"]]
    vocab = CharVocab.build(texts, max_length=args.max_length)
    print(f"[emotion] vocab_size={len(vocab.char2id)}", flush=True)

    model = SimpleEmotionModel(len(vocab.char2id)).to(device)
    collate_fn = partial(collate_emotion, vocab=vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = checkpoint_dir("emotion")
    best_f1 = -1.0
    save_config(out_dir / "config.json", {**vars(args), "emotion_labels": EMOTION_LABELS})
    vocab.save(out_dir / "vocab.json")

    def train_epoch(model, loader, optimizer, scheduler, criterion, device, epoch=1, log_every=50):
        model.train()
        progress = TrainProgress(len(loader), desc=f"emotion e{epoch}", log_every=log_every)
        total_loss = 0.0
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.update(loss.item())
        return total_loss / max(len(loader), 1)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, None, criterion, device, epoch, args.log_every)
        val_loss, preds, labels = eval_emotion_simple(model, val_loader, criterion, device)
        metrics = compute_metrics(preds, labels)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"mae={metrics['mae']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"dominant_acc={metrics['dominant_acc']:.4f}",
            flush=True,
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": len(vocab.char2id),
                    "max_length": args.max_length,
                    "emotion_labels": EMOTION_LABELS,
                    "metrics": metrics,
                },
                out_dir / "best.pt",
            )

    print(f"Saved {out_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
