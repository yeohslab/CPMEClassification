"""Generate PDF figures for CPME training report (matplotlib)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PIC = Path(__file__).resolve().parents[1] / "pic"
EMOTION_LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
MBTI_TYPES = [
    "ENFJ", "ENFP", "ENTJ", "ENTP", "ESFJ", "ESFP", "ESTJ", "ESTP",
    "INFJ", "INFP", "INTJ", "INTP", "ISFJ", "ISFP", "ISTJ", "ISTP",
]

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load_posts() -> pd.DataFrame:
    cached = ROOT / "processed" / "posts.csv"
    if cached.exists():
        return pd.read_csv(cached)
    rows = []
    for mbti_dir in sorted((ROOT / "data").iterdir()):
        if not mbti_dir.is_dir() or mbti_dir.name not in MBTI_TYPES:
            continue
        for csv_path in sorted(mbti_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                rows.append({lab: float(row[lab]) for lab in EMOTION_LABELS} | {"mbti": mbti_dir.name})
    return pd.DataFrame(rows)


def fig_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, 1.0, "原始 CSV\n(data/)"),
        (2.0, 1.0, "prepare\_dataset\nposts.csv + splits"),
        (4.2, 1.8, "train\_emotion\nchar-CNN"),
        (4.2, 0.2, "train\_mbti\nTextCNN+Attn"),
        (6.6, 1.0, "evaluate.py\n测试集指标"),
        (8.4, 1.0, "export\_onnx\n浏览器推理"),
    ]
    colors = ["#E8F4FD", "#D4EDDA", "#FFF3CD", "#FFF3CD", "#F8D7DA", "#E2D9F3"]
    for (x, y, text), c in zip(boxes, colors):
        ax.add_patch(FancyBboxPatch(
            (x, y), 1.5, 0.9, boxstyle="round,pad=0.05", fc=c, ec="#333", lw=1.2
        ))
        ax.text(x + 0.75, y + 0.45, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((1.7, 1.45), (2.0, 1.45)),
        ((3.5, 1.45), (4.2, 2.0)),
        ((3.5, 1.45), (4.2, 0.55)),
        ((5.7, 2.0), (6.6, 1.55)),
        ((5.7, 0.55), (6.6, 1.35)),
        ((8.1, 1.45), (8.4, 1.45)),
    ]
    for (a, b) in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", color="#444", lw=1.2, mutation_scale=12))
    ax.set_title("CPMEClassification 训练与部署流水线", fontsize=11, pad=8)
    fig.savefig(out)
    plt.close(fig)


def fig_data_split(out: Path) -> None:
    tasks = ["情绪（帖子）", "MBTI（用户）"]
    train = [64000, 1280]
    val = [8000, 160]
    test = [8000, 160]
    x = np.arange(len(tasks))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, train, w, label="训练 80%", color="#4C78A8")
    ax.bar(x, val, w, label="验证 10%", color="#F58518")
    ax.bar(x + w, test, w, label="测试 10%", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("样本数")
    ax.legend()
    ax.set_title("数据集划分（seed=42，比例 8:1:1）")
    for i, vals in enumerate(zip(train, val, test)):
        for j, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + max(train) * 0.01, f"{v:,}", ha="center", fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fig_mbti_dist(df: pd.DataFrame, out: Path) -> None:
    counts = df.groupby("mbti").size().reindex(MBTI_TYPES).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(MBTI_TYPES))
    ax.barh(y, counts.values, color=plt.cm.tab20(np.linspace(0, 1, 16)))
    ax.set_yticks(y)
    ax.set_yticklabels(MBTI_TYPES, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("帖子数")
    ax.set_title("各 MBTI 类型帖子规模（扁平化后）")
    fig.savefig(out)
    plt.close(fig)


def fig_emotion_mean(df: pd.DataFrame, out: Path) -> None:
    means = [df[lab].mean() for lab in EMOTION_LABELS]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#E45756", "#F58518", "#72B7B2", "#BAB0AC", "#4C78A8", "#EECA3B"]
    ax.bar(EMOTION_LABELS, means, color=colors)
    ax.set_ylim(0, max(means) * 1.15)
    ax.set_ylabel("全库平均强度")
    ax.set_title("六维情绪标注的全局均值")
    for i, m in enumerate(means):
        ax.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def fig_emotion_corr(df: pd.DataFrame, out: Path) -> None:
    corr = df[EMOTION_LABELS].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTION_LABELS)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("情绪维度 Pearson 相关矩阵")
    fig.savefig(out)
    plt.close(fig)


def _draw_block(ax, xy, w, h, text, fc="#E8F4FD"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def fig_emotion_arch(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    blocks = [
        (0.3, 0.55, 1.4, 0.9, "字符序列\nmax\_len=128"),
        (2.1, 0.55, 1.4, 0.9, "Embedding\n128-d"),
        (3.9, 0.55, 1.6, 0.9, "Conv1d k=3\n+ ReLU"),
        (5.9, 0.55, 1.5, 0.9, "掩码平均池化"),
        (7.8, 0.55, 1.4, 0.9, "Linear\n6 logits"),
    ]
    for x, y, w, h, t in blocks:
        _draw_block(ax, (x, y), w, h, t)
    for i in range(4):
        ax.annotate("", xy=(blocks[i + 1][0], 1.0), xytext=(blocks[i][0] + blocks[i][2], 1.0),
                    arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_title("SimpleEmotionModel：单帖 char-CNN 结构", fontsize=11)
    fig.savefig(out)
    plt.close(fig)


def fig_mbti_arch(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    _draw_block(ax, (0.2, 1.2), 1.6, 1.0, "用户 $P$ 条帖子\n$P\\leq 16$", "#FFF3CD")
    _draw_block(ax, (2.2, 1.2), 2.0, 1.0, "每帖 TextCNN\n核 2,3,4,5", "#D4EDDA")
    _draw_block(ax, (4.6, 1.2), 1.8, 1.0, "帖子向量\n$h_1..h_P$", "#E8F4FD")
    _draw_block(ax, (6.8, 1.2), 1.8, 1.0, "注意力池化\nsoftmax", "#F8D7DA")
    _draw_block(ax, (9.0, 1.2), 1.5, 1.0, "16 类\nMBTI", "#E2D9F3")
    for x in [1.8, 4.2, 6.4, 8.6]:
        ax.annotate("", xy=(x + 0.35, 1.7), xytext=(x, 1.7),
                    arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_title("MbtiTextCNNModel：多帖编码 + 用户级注意力", fontsize=11)
    fig.savefig(out)
    plt.close(fig)


def fig_loss_tradeoff(out: Path) -> None:
    """Illustrative BCE vs CE + class weight effect (conceptual)."""
    epochs = np.arange(1, 11)
    emo = 0.45 * np.exp(-0.35 * epochs) + 0.08 + np.random.default_rng(0).normal(0, 0.01, 10)
    mbti = 2.2 * np.exp(-0.22 * epochs) + 0.35 + np.random.default_rng(1).normal(0, 0.03, 10)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, emo, "o-", color="#4C78A8", label="情绪 val loss (示意)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE 验证损失", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(epochs, mbti, "s-", color="#E45756", label="MBTI val loss (示意)")
    ax2.set_ylabel("CE 验证损失", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right")
    ax1.set_title("典型训练曲线形态（示意，非固定 checkpoint）")
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    PIC.mkdir(parents=True, exist_ok=True)
    print("[fig] loading posts...", flush=True)
    df = load_posts()
    specs = [
        ("pipeline_overview.pdf", lambda: fig_pipeline(PIC / "pipeline_overview.pdf")),
        ("data_split.pdf", lambda: fig_data_split(PIC / "data_split.pdf")),
        ("mbti_class_dist.pdf", lambda: fig_mbti_dist(df, PIC / "mbti_class_dist.pdf")),
        ("emotion_mean.pdf", lambda: fig_emotion_mean(df, PIC / "emotion_mean.pdf")),
        ("emotion_corr.pdf", lambda: fig_emotion_corr(df, PIC / "emotion_corr.pdf")),
        ("emotion_architecture.pdf", lambda: fig_emotion_arch(PIC / "emotion_architecture.pdf")),
        ("mbti_architecture.pdf", lambda: fig_mbti_arch(PIC / "mbti_architecture.pdf")),
        ("training_loss_tradeoff.pdf", lambda: fig_loss_tradeoff(PIC / "training_loss_tradeoff.pdf")),
    ]
    for name, fn in specs:
        fn()
        print(f"  wrote {name}", flush=True)
    print(f"Done -> {PIC}", flush=True)


if __name__ == "__main__":
    main()
