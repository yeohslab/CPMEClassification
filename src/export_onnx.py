"""Export char-CNN models to ONNX for browser inference."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import EMOTION_LABELS, MBTI_TYPES
from models.mbti_textcnn import MbtiTextCNNModel
from models.simple_classifier import CharVocab, SimpleEmotionModel
from training_utils import checkpoint_dir, get_device


class EmotionOnnx(torch.nn.Module):
    def __init__(self, model: SimpleEmotionModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        return torch.sigmoid(self.model(input_ids))


class MbtiOnnx(torch.nn.Module):
    def __init__(self, model: MbtiTextCNNModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, post_mask: torch.Tensor):
        return torch.softmax(self.model(input_ids, post_mask), dim=-1)


def export_all(out_dir: Path) -> None:
    device = get_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    emo_ckpt = torch.load(checkpoint_dir("emotion") / "best.pt", map_location=device, weights_only=False)
    emo_max_len = emo_ckpt.get("max_length", 128)
    emo_model = SimpleEmotionModel(emo_ckpt["vocab_size"]).to(device)
    emo_model.load_state_dict(emo_ckpt["model_state"])
    emo_model.eval()
    wrapper = EmotionOnnx(emo_model).to(device).eval()
    dummy = torch.ones(1, emo_max_len, dtype=torch.long, device=device)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_dir / "emotion.onnx"),
        input_names=["input_ids"],
        output_names=["scores"],
        dynamic_axes={"input_ids": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    shutil.copy(checkpoint_dir("emotion") / "vocab.json", out_dir / "vocab_emotion.json")
    with open(out_dir / "emotion_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": EMOTION_LABELS,
                "max_length": emo_max_len,
                "vocab": "vocab_emotion.json",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    mbti_ckpt = torch.load(checkpoint_dir("mbti") / "best.pt", map_location=device, weights_only=False)
    mbti_max_len = mbti_ckpt.get("max_length", 160)
    mbti_max_posts = mbti_ckpt.get("max_posts", 16)
    mbti_model = MbtiTextCNNModel(
        mbti_ckpt["vocab_size"],
        embed_dim=mbti_ckpt.get("embed_dim", 256),
        num_filters=mbti_ckpt.get("num_filters", 128),
    ).to(device)
    mbti_model.load_state_dict(mbti_ckpt["model_state"])
    mbti_model.eval()
    wrapper_m = MbtiOnnx(mbti_model).to(device).eval()
    dummy_ids = torch.ones(1, mbti_max_posts, mbti_max_len, dtype=torch.long, device=device)
    dummy_mask = torch.ones(1, mbti_max_posts, device=device)
    torch.onnx.export(
        wrapper_m,
        (dummy_ids, dummy_mask),
        str(out_dir / "mbti.onnx"),
        input_names=["input_ids", "post_mask"],
        output_names=["probs"],
        opset_version=18,
        dynamo=False,
    )
    shutil.copy(checkpoint_dir("mbti") / "vocab.json", out_dir / "vocab_mbti.json")
    with open(out_dir / "mbti_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": MBTI_TYPES,
                "max_length": mbti_max_len,
                "max_posts": mbti_max_posts,
                "vocab": "vocab_mbti.json",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Exported to {out_dir}")


def smoke(out_dir: Path) -> None:
    vocab = CharVocab.load(out_dir / "vocab_emotion.json")
    ids = np.array([vocab.encode("今天心情很好")], dtype=np.int64)
    sess = ort.InferenceSession(str(out_dir / "emotion.onnx"), providers=["CPUExecutionProvider"])
    scores = sess.run(None, {"input_ids": ids})[0][0]
    print("Emotion:", dict(zip(EMOTION_LABELS, scores.round(3))))

    vocab_m = CharVocab.load(out_dir / "vocab_mbti.json")
    with open(out_dir / "mbti_meta.json") as f:
        meta = json.load(f)
    max_posts = meta["max_posts"]
    posts = ["我喜欢独处", "逻辑分析很重要", "计划要提前"]
    padded = posts + [""] * (max_posts - len(posts))
    ids = np.array([[vocab_m.encode(p) for p in padded]], dtype=np.int64)
    mask = np.zeros((1, max_posts), dtype=np.float32)
    mask[0, : len(posts)] = 1.0
    sess_m = ort.InferenceSession(str(out_dir / "mbti.onnx"), providers=["CPUExecutionProvider"])
    probs = sess_m.run(None, {"input_ids": ids, "post_mask": mask})[0][0]
    print("MBTI:", MBTI_TYPES[int(np.argmax(probs))])


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "web" / "public" / "models"
    export_all(out)
    smoke(out)
