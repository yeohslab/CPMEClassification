import { useState } from "react";
import { predictMbti } from "../inference";
import MbtiResultCard from "./MbtiResultCard";

interface Props {
  modelsReady: boolean;
  onResult?: (mbti: string, top3: { type: string; prob: number }[]) => void;
}

const MAX_POSTS = 16;

const DEFAULT_POSTS = [
  "我喜欢独处思考，逻辑分析很重要。",
  "计划要提前安排好，不喜欢临时变动。",
  "更关注内心感受而非表面热闹。",
];

export default function MbtiPanel({ modelsReady, onResult }: Props) {
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
      onResult?.(result.mbti, result.probs);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const atMax = posts.length >= MAX_POSTS;

  return (
    <div className="panel-layout panel-layout--mbti">
      <div className="panel panel--main">
        <header className="panel__header">
          <h2 className="panel__title">多帖 MBTI 预测</h2>
          <p className="panel__desc">
            输入同一用户的多条帖子（建议 8–16 条），模型将聚合文本特征预测
            16 型之一。
          </p>
        </header>
        <div className="mbti-posts">
          {posts.map((post, i) => (
            <div className="mbti-post-row" key={i}>
              <label htmlFor={`mbti-post-${i}`} className="mbti-post-row__label">
                帖子 {i + 1}
              </label>
              <textarea
                id={`mbti-post-${i}`}
                className="input-textarea input-textarea--compact"
                value={post}
                onChange={(e) => updatePost(i, e.target.value)}
                placeholder="输入一条帖子…"
                rows={2}
                disabled={!modelsReady}
              />
              {posts.length > 1 && (
                <button
                  type="button"
                  className="btn btn--icon"
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
            className="btn btn--dashed"
            onClick={addPost}
            disabled={!modelsReady || atMax}
          >
            + 添加帖子
            {atMax ? `（最多 ${MAX_POSTS} 条）` : ""}
          </button>
        </div>
        <button
          type="button"
          className="btn btn--primary"
          onClick={handlePredict}
          disabled={!modelsReady || loading}
        >
          {loading ? "分析中…" : "预测 MBTI"}
        </button>
        {error && (
          <p className="status status--error" role="alert">
            {error}
          </p>
        )}
      </div>
      {mbti && top3.length > 0 && (
        <div className="panel-aside panel-aside--result">
          <MbtiResultCard mbti={mbti} top3={top3} />
        </div>
      )}
    </div>
  );
}
