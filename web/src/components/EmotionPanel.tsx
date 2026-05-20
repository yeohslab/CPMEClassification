import { useState } from "react";
import {
  EMOTION_LABELS,
  EMOTION_LABELS_ZH,
  predictEmotion,
} from "../inference";

interface Props {
  modelsReady: boolean;
}

export default function EmotionPanel({ modelsReady }: Props) {
  const [text, setText] = useState(
    "今天心情不错，阳光很好，想出去走走。"
  );
  const [scores, setScores] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!text.trim()) {
      setError("请输入帖子内容");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await predictEmotion(text.trim());
      setScores(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const dominant =
    scores &&
    EMOTION_LABELS.reduce((a, b) => (scores[a] >= scores[b] ? a : b));

  return (
    <div className="panel">
      <p>输入一条微博风格帖子，预测 6 维情绪强度（0–1）。</p>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="输入帖子…"
        disabled={!modelsReady}
      />
      <button
        className="primary"
        onClick={handlePredict}
        disabled={!modelsReady || loading}
      >
        {loading ? "分析中…" : "分析情绪"}
      </button>
      {error && <p className="status error">{error}</p>}
      {scores && dominant && (
        <>
          <div className="dominant">
            主情绪：<strong>{EMOTION_LABELS_ZH[dominant] ?? dominant}</strong>（
            {dominant}，强度 {(scores[dominant] * 100).toFixed(1)}%）
          </div>
          <div className="emotion-bars">
            {EMOTION_LABELS.map((label) => (
              <div className="emotion-row" key={label}>
                <label>{EMOTION_LABELS_ZH[label]}</label>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{ width: `${scores[label] * 100}%` }}
                  />
                </div>
                <span className="score">{(scores[label] * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
