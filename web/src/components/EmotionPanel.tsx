import { useState } from "react";
import {
  EMOTION_LABELS,
  EMOTION_LABELS_ZH,
  predictEmotion,
} from "../inference";
import { DECOR } from "../mbtiMeta";

interface Props {
  modelsReady: boolean;
  onResult?: (dominant: string, score: number) => void;
}

export default function EmotionPanel({ modelsReady, onResult }: Props) {
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
      const dominant = EMOTION_LABELS.reduce((a, b) =>
        result[a] >= result[b] ? a : b
      );
      onResult?.(dominant, result[dominant]);
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
    <div className="panel-layout">
      <div className="panel panel--main">
        <header className="panel__header">
          <h2 className="panel__title">单帖情绪分析</h2>
          <p className="panel__desc">
            输入一条微博风格帖子，预测 6 维情绪强度（0–1）。
          </p>
        </header>
        <textarea
          className="input-textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="输入帖子…"
          disabled={!modelsReady}
          aria-label="帖子内容"
        />
        <button
          type="button"
          className="btn btn--primary"
          onClick={handlePredict}
          disabled={!modelsReady || loading}
        >
          {loading ? "分析中…" : "分析情绪"}
        </button>
        {error && (
          <p className="status status--error" role="alert">
            {error}
          </p>
        )}
        {scores && dominant && (
          <div className="emotion-results">
            <div
              className={`dominant-badge dominant-badge--${dominant}`}
            >
              主情绪：
              <strong>{EMOTION_LABELS_ZH[dominant] ?? dominant}</strong>
              <span className="dominant-badge__meta">
                （{(scores[dominant] * 100).toFixed(1)}%）
              </span>
            </div>
            <div className="emotion-bars" aria-label="情绪强度">
              {EMOTION_LABELS.map((label) => (
                <div className="emotion-row" key={label}>
                  <label className={`emotion-row__label emotion-row__label--${label}`}>
                    {EMOTION_LABELS_ZH[label]}
                  </label>
                  <div className="bar-track">
                    <div
                      className={`bar-fill bar-fill--${label}`}
                      style={{ width: `${scores[label] * 100}%` }}
                    />
                  </div>
                  <span className="emotion-row__score">
                    {(scores[label] * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      <aside className="panel-aside">
        <img
          src={DECOR.emotionSide}
          alt=""
          className="panel-aside__img"
        />
        <p className="panel-aside__caption">
          情绪向量反映文本表层情感倾向，与 MBTI 类型相互独立。
        </p>
      </aside>
    </div>
  );
}
