import { useState } from "react";
import { predictMbti } from "../inference";

interface Props {
  modelsReady: boolean;
}

const MAX_POSTS = 16;

const DEFAULT_POSTS = [
  "我喜欢独处思考，逻辑分析很重要。",
  "计划要提前安排好，不喜欢临时变动。",
  "更关注内心感受而非表面热闹。",
];

export default function MbtiPanel({ modelsReady }: Props) {
  const [posts, setPosts] = useState<string[]>(DEFAULT_POSTS);
  const [mbti, setMbti] = useState<string | null>(null);
  const [top3, setTop3] = useState<{ type: string; prob: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updatePost = (index: number, value: string) => {
    setPosts((prev) => prev.map((p, i) => (i === index ? value : p)));
  };

  const addPost = () => {
    if (posts.length >= MAX_POSTS) return;
    setPosts((prev) => [...prev, ""]);
  };

  const removePost = (index: number) => {
    if (posts.length <= 1) return;
    setPosts((prev) => prev.filter((_, i) => i !== index));
  };

  const handlePredict = async () => {
    const valid = posts.map((p) => p.trim()).filter(Boolean);
    if (valid.length === 0) {
      setError("请至少填写一条帖子");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await predictMbti(valid);
      setMbti(result.mbti);
      setTop3(result.probs);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const atMax = posts.length >= MAX_POSTS;

  return (
    <div className="panel">
      <p>
        输入同一用户的多条帖子（每条一格，建议 8–16 条）。需要更多时点击「添加帖子」。
      </p>
      <div className="mbti-posts">
        {posts.map((post, i) => (
          <div className="mbti-post-row" key={i}>
            <label htmlFor={`mbti-post-${i}`}>帖子 {i + 1}</label>
            <textarea
              id={`mbti-post-${i}`}
              value={post}
              onChange={(e) => updatePost(i, e.target.value)}
              placeholder="输入一条帖子…"
              rows={2}
              disabled={!modelsReady}
            />
            {posts.length > 1 && (
              <button
                type="button"
                className="mbti-remove"
                onClick={() => removePost(i)}
                disabled={!modelsReady}
                title="删除此条"
                aria-label={`删除帖子 ${i + 1}`}
              >
                ×
              </button>
            )}
          </div>
        ))}
        <button
          type="button"
          className="mbti-add"
          onClick={addPost}
          disabled={!modelsReady || atMax}
        >
          + 添加帖子
          {atMax ? `（最多 ${MAX_POSTS} 条）` : ""}
        </button>
      </div>
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
