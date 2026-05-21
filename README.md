# CPEM Classification — 情绪 & MBTI 推理

基于 CPME 中文微博风格数据集的双模块推理项目：

| 模块 | 输入 | 输出 |
|------|------|------|
| 情绪 | 1 条帖子 | 6 维情绪强度（0–1） |
| MBTI | 8–16 条帖子（同一用户） | 16 型 MBTI |

演示站（GitHub Pages）：`https://yeohslab.github.io/CPMEClassification/`

## 环境

```bash
python -m pip install -r requirements.txt
cd web && npm install
```

## 训练

```bash
cd src
python prepare_dataset.py          # 首次运行
python train_emotion.py --epochs 3 --batch_size 128
python train_mbti.py --epochs 10 --batch_size 24 --max_posts 16
python evaluate.py                 # 测试集指标
python export_onnx.py              # 导出到 web/public/models/
```

一键脚本：`powershell -File scripts/train_all.ps1`

## 本地演示

```bash
cd web
npm run dev
```

## 项目结构

```
data/                 # 原始 CSV
processed/            # posts.csv 缓存
splits/               # 划分索引
src/
  train_emotion.py    # 情绪 char-CNN
  train_mbti.py       # MBTI TextCNN + 注意力
  export_onnx.py
  evaluate.py
web/                  # Vite 前端 + ONNX 浏览器推理
models/checkpoints/   # 训练权重（本地，已 gitignore）
```

## 说明

- 使用本地 **char-CNN**，无需 HuggingFace，可在 CPU 上训练。
- 情绪标注为 6 类：`angry, fear, happy, neutral, sad, surprise`（与当前数据一致，不含微情绪列）。
- MBTI 为统计模型演示，**不构成**心理测评建议。

## 引用

Zhou et al., arXiv:2411.08347 — [CPME 数据集](https://github.com/yeaso/Chinese-Affective-Computing-Dataset)
