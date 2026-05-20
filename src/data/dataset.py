"""CPME data loading, splitting, and PyTorch datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

EMOTION_LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
MBTI_TYPES = [
    "ENFJ", "ENFP", "ENTJ", "ENTP",
    "ESFJ", "ESFP", "ESTJ", "ESTP",
    "INFJ", "INFP", "INTJ", "INTP",
    "ISFJ", "ISFP", "ISTJ", "ISTP",
]
MBTI_TO_ID = {t: i for i, t in enumerate(MBTI_TYPES)}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_splits_dir() -> Path:
    d = get_project_root() / "splits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_posts_dataframe(data_dir: Path | None = None) -> pd.DataFrame:
    """Load posts from processed/posts.csv cache when available."""
    cached = get_project_root() / "processed" / "posts.csv"
    if cached.exists():
        print(f"[data] load {cached}", flush=True)
        return pd.read_csv(cached)
    print("[data] building posts.csv from raw data/ ...", flush=True)
    df = load_all_posts(data_dir)
    cached.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cached, index=False, encoding="utf-8")
    return df


def load_all_posts(data_dir: Path | None = None) -> pd.DataFrame:
    """Flatten all CSV rows into a single DataFrame."""
    data_dir = data_dir or get_data_dir()
    rows: list[dict[str, Any]] = []
    for mbti_dir in sorted(data_dir.iterdir()):
        if not mbti_dir.is_dir():
            continue
        mbti = mbti_dir.name
        if mbti not in MBTI_TO_ID:
            continue
        for csv_path in sorted(mbti_dir.glob("*.csv")):
            user_id = f"{mbti}/{csv_path.stem}"
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                rows.append({
                    "post": str(row["posts"]).strip(),
                    "mbti": mbti,
                    "user_id": user_id,
                    **{label: float(row[label]) for label in EMOTION_LABELS},
                })
    return pd.DataFrame(rows)


def load_user_files(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """One record per user CSV file with all posts."""
    data_dir = data_dir or get_data_dir()
    users: list[dict[str, Any]] = []
    for mbti_dir in sorted(data_dir.iterdir()):
        if not mbti_dir.is_dir():
            continue
        mbti = mbti_dir.name
        if mbti not in MBTI_TO_ID:
            continue
        for csv_path in sorted(mbti_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            posts = [str(p).strip() for p in df["posts"].tolist()]
            users.append({
                "user_id": f"{mbti}/{csv_path.stem}",
                "mbti": mbti,
                "mbti_id": MBTI_TO_ID[mbti],
                "posts": posts,
            })
    return users


def split_indices(n: int, seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)):
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = indices[:n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val :]
    return train, val, test


def prepare_splits(seed: int = 42) -> dict[str, Any]:
    """Build and persist emotion (row) and mbti (user file) splits."""
    splits_dir = get_splits_dir()
    df = load_all_posts()
    users = load_user_files()

    row_train, row_val, row_test = split_indices(len(df), seed=seed)
    user_train, user_val, user_test = split_indices(len(users), seed=seed)

    emotion_split = {
        "train_indices": row_train,
        "val_indices": row_val,
        "test_indices": row_test,
        "total": len(df),
    }
    mbti_split = {
        "train_indices": user_train,
        "val_indices": user_val,
        "test_indices": user_test,
        "total": len(users),
    }

    with open(splits_dir / "emotion.json", "w", encoding="utf-8") as f:
        json.dump(emotion_split, f)
    with open(splits_dir / "mbti_users.json", "w", encoding="utf-8") as f:
        json.dump(mbti_split, f)

    processed_dir = get_project_root() / "processed"
    processed_dir.mkdir(exist_ok=True)
    df.to_csv(processed_dir / "posts.csv", index=False, encoding="utf-8")

    meta = {
        "emotion_labels": EMOTION_LABELS,
        "mbti_types": MBTI_TYPES,
        "num_posts": len(df),
        "num_users": len(users),
    }
    with open(processed_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"emotion": emotion_split, "mbti": mbti_split, "meta": meta}


def load_emotion_split() -> dict[str, list[int]]:
    with open(get_splits_dir() / "emotion.json", encoding="utf-8") as f:
        return json.load(f)


def load_mbti_split() -> dict[str, list[int]]:
    with open(get_splits_dir() / "mbti_users.json", encoding="utf-8") as f:
        return json.load(f)


class EmotionDataset(Dataset):
    def __init__(self, indices: list[int], df: pd.DataFrame):
        self.indices = indices
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[self.indices[idx]]
        labels = torch.tensor(
            [row[label] for label in EMOTION_LABELS], dtype=torch.float32
        )
        return {"text": row["post"], "labels": labels}


class MbtiUserDataset(Dataset):
    def __init__(
        self,
        user_indices: list[int],
        users: list[dict[str, Any]],
        max_posts: int = 8,
        training: bool = True,
    ):
        self.user_indices = user_indices
        self.users = users
        self.max_posts = max_posts
        self.training = training

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        user = self.users[self.user_indices[idx]]
        posts = user["posts"]
        if self.training and len(posts) > self.max_posts:
            posts = random.sample(posts, self.max_posts)
        elif len(posts) > self.max_posts:
            posts = posts[: self.max_posts]
        return {
            "texts": posts,
            "mbti_id": user["mbti_id"],
            "mbti": user["mbti"],
        }
