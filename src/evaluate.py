"""Evaluate checkpoints on the test split."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, mean_absolute_error
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import (
    EMOTION_LABELS,
    EmotionDataset,
    MbtiUserDataset,
    load_emotion_split,
    load_mbti_split,
    load_posts_dataframe,
    load_user_files,
)
from models.mbti_textcnn import MbtiTextCNNModel
from models.simple_classifier import CharVocab, SimpleEmotionModel
from training_utils import checkpoint_dir, collate_emotion, collate_mbti, get_device


@torch.no_grad()
def eval_emotion(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        logits = model(input_ids)
        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())
    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return preds, labels


@torch.no_grad()
def eval_mbti(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    for input_ids, post_mask, mbti_ids in loader:
        input_ids = input_ids.to(device)
        post_mask = post_mask.to(device)
        pred = model(input_ids, post_mask).argmax(-1)
        all_preds.extend(pred.cpu().tolist())
        all_true.extend(mbti_ids.tolist())
    return all_preds, all_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["emotion", "mbti", "both"], default="both")
    args = parser.parse_args()
    device = get_device()

    if args.task in ("emotion", "both"):
        ckpt = torch.load(checkpoint_dir("emotion") / "best.pt", map_location=device, weights_only=False)
        vocab = CharVocab.load(checkpoint_dir("emotion") / "vocab.json")
        df = load_posts_dataframe()
        split = load_emotion_split()
        test_ds = EmotionDataset(split["test_indices"], df)
        model = SimpleEmotionModel(ckpt["vocab_size"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        loader = DataLoader(
            test_ds,
            batch_size=64,
            shuffle=False,
            collate_fn=partial(collate_emotion, vocab=vocab),
        )
        preds, labels = eval_emotion(model, loader, device)
        mae = mean_absolute_error(labels, preds)
        macro_f1 = f1_score(
            (labels >= 0.5).astype(int),
            (preds >= 0.5).astype(int),
            average="macro",
            zero_division=0,
        )
        dom_acc = (labels.argmax(1) == preds.argmax(1)).mean()
        print(f"[Emotion test] MAE={mae:.4f} macro_f1={macro_f1:.4f} dominant_acc={dom_acc:.4f}")

    if args.task in ("mbti", "both"):
        ckpt = torch.load(checkpoint_dir("mbti") / "best.pt", map_location=device, weights_only=False)
        vocab = CharVocab.load(checkpoint_dir("mbti") / "vocab.json")
        users = load_user_files()
        split = load_mbti_split()
        max_posts = ckpt.get("max_posts", 16)
        test_ds = MbtiUserDataset(split["test_indices"], users, max_posts=max_posts, training=False)
        model = MbtiTextCNNModel(
            ckpt["vocab_size"],
            embed_dim=ckpt.get("embed_dim", 256),
            num_filters=ckpt.get("num_filters", 128),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        loader = DataLoader(
            test_ds,
            batch_size=24,
            shuffle=False,
            collate_fn=partial(collate_mbti, vocab=vocab, max_posts=max_posts),
        )
        preds, true = eval_mbti(model, loader, device)
        acc = sum(p == t for p, t in zip(preds, true)) / max(len(true), 1)
        macro_f1 = f1_score(true, preds, average="macro", zero_division=0)
        print(f"[MBTI test] accuracy={acc:.4f} macro_f1={macro_f1:.4f}")


if __name__ == "__main__":
    main()
