"""CLI: build train/val/test splits and processed/posts.csv."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import prepare_splits  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_splits(seed=args.seed)
    meta = result["meta"]
    print(f"Prepared {meta['num_posts']} posts, {meta['num_users']} user files.")
    print(f"Emotion split: train={len(result['emotion']['train_indices'])}, "
          f"val={len(result['emotion']['val_indices'])}, "
          f"test={len(result['emotion']['test_indices'])}")
    print(f"MBTI split: train={len(result['mbti']['train_indices'])}, "
          f"val={len(result['mbti']['val_indices'])}, "
          f"test={len(result['mbti']['test_indices'])}")


if __name__ == "__main__":
    main()
