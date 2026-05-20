"""Train multi-post MBTI model (TextCNN + attention)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import MBTI_TYPES, MbtiUserDataset, load_mbti_split, load_user_files
from models.mbti_textcnn import MbtiTextCNNModel
from models.simple_classifier import CharVocab
from training_utils import TrainProgress, checkpoint_dir, collate_mbti, get_device, save_config


def class_weights_from_users(users, train_indices: list[int], num_classes: int = 16) -> torch.Tensor:
    counts = Counter(users[i]["mbti_id"] for i in train_indices)
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        n = counts.get(c, 1)
        weights.append(total / (num_classes * n))
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_posts", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    users = load_user_files()
    split = load_mbti_split()
    train_ds = MbtiUserDataset(
        split["train_indices"], users, max_posts=args.max_posts, training=True
    )
    val_ds = MbtiUserDataset(
        split["val_indices"], users, max_posts=args.max_posts, training=False
    )

    texts = []
    for idx in split["train_indices"]:
        texts.extend(users[idx]["posts"])
    vocab = CharVocab.build(texts, max_length=args.max_length, min_freq=1)
    print(f"[mbti] vocab={len(vocab.char2id)} max_posts={args.max_posts}", flush=True)

    collate_fn = partial(collate_mbti, vocab=vocab, max_posts=args.max_posts)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = MbtiTextCNNModel(
        len(vocab.char2id),
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
    ).to(device)
    cw = class_weights_from_users(users, split["train_indices"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = checkpoint_dir("mbti")
    best_f1 = -1.0
    save_config(
        out_dir / "config.json",
        {**vars(args), "mbti_types": MBTI_TYPES},
    )
    vocab.save(out_dir / "vocab.json")

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = TrainProgress(len(train_loader), desc=f"mbti e{epoch}", log_every=args.log_every)
        correct = total = 0
        for input_ids, post_mask, mbti_ids in train_loader:
            input_ids = input_ids.to(device)
            post_mask = post_mask.to(device)
            mbti_ids = mbti_ids.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, post_mask)
            loss = criterion(logits, mbti_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            correct += (logits.argmax(-1) == mbti_ids).sum().item()
            total += mbti_ids.size(0)
            progress.update(loss.item())
        scheduler.step()
        train_acc = correct / max(total, 1)

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for input_ids, post_mask, mbti_ids in val_loader:
                input_ids = input_ids.to(device)
                post_mask = post_mask.to(device)
                logits = model(input_ids, post_mask.to(device))
                pred = logits.argmax(-1)
                all_preds.extend(pred.cpu().tolist())
                all_true.extend(mbti_ids.tolist())
        val_acc = sum(p == t for p, t in zip(all_preds, all_true)) / max(len(all_true), 1)
        macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        print(
            f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} macro_f1={macro_f1:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": len(vocab.char2id),
                    "max_length": args.max_length,
                    "max_posts": args.max_posts,
                    "embed_dim": args.embed_dim,
                    "num_filters": args.num_filters,
                    "mbti_types": MBTI_TYPES,
                    "val_acc": val_acc,
                    "macro_f1": macro_f1,
                },
                out_dir / "best.pt",
            )

    print(f"\nBest macro_f1={best_f1:.4f}", flush=True)
    print(classification_report(all_true, all_preds, target_names=MBTI_TYPES, zero_division=0))
    print(f"Saved {out_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
