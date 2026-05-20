import { useState } from "react";
import { predictMbti } from "../inference";

interface Props {
  modelsReady: boolean;
}

const PLACEHOLDER = `我喜欢独处思考，逻辑分析很重要。
计划要提前安排好，不喜欢临时变动。
更关注内心感受而非表面热闹。`;

export default function MbtiPanel({ modelsReady }: Props) {
  const [text, setText] = useState(PLACEHOLDER);
  const [mbti, setMbti] = useState<string | null>(null);
  const [top3, setTop3] = useState<{ type: string; prob: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    const posts = text
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    if (posts.length === 0) {
      setError("请至少输入一行帖子（每行一条）");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await predictMbti(posts);
      setMbti(result.mbti);
      setTop3(result.probs);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel">
      <p>输入同一用户的多条帖子（每行一条，建议 8–16 条），预测 MBTI 类型。</p>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="每行一条帖子…"
        rows={8}
        disabled={!modelsReady}
      />
      <button
        className="primary"
        onClick={handlePredict}
        disabled={!modelsReady || loading}
      >
        {loading ? "分析中…" : "预测 MBTI"}
      </button>
      {error && <p className="status error">{error}</p>}
      {mbti && (
        <>
          <div className="mbti-result">预测类型：{mbti}</div>
          <div className="mbti-top">
            Top-3：
            {top3.map((t) => (
              <div key={t.type}>
                {t.type} — {(t.prob * 100).toFixed(1)}%
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
