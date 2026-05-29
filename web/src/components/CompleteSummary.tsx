import { DECOR } from "../mbtiMeta";
import MbtiResultCard from "./MbtiResultCard";
import { EMOTION_LABELS_ZH } from "../inference";

interface Props {
  dominantEmotion: string | null;
  dominantScore: number | null;
  mbti: string | null;
  top3: { type: string; prob: number }[];
  onRestart: () => void;
}

export default function CompleteSummary({
  dominantEmotion,
  dominantScore,
  mbti,
  top3,
  onRestart,
}: Props) {
  return (
    <section className="complete-page" aria-labelledby="complete-title">
      <div className="complete-page__hero">
        <img
          src={DECOR.complete}
          alt=""
          className="complete-page__decor"
          aria-hidden="true"
        />
        <div>
          <h1 id="complete-title" className="complete-page__title">
            体验完成
          </h1>
          <p className="complete-page__lead">
            以下为你的演示结果摘要。再次体验可从头开始。
          </p>
        </div>
      </div>

      <div className="complete-page__grid">
        <div className="summary-card">
          <h2 className="summary-card__heading">情绪主倾向</h2>
          {dominantEmotion ? (
            <p className="summary-card__value">
              <span
                className={`emotion-badge emotion-badge--${dominantEmotion}`}
              >
                {EMOTION_LABELS_ZH[dominantEmotion] ?? dominantEmotion}
              </span>
              {dominantScore != null && (
                <span className="summary-card__meta">
                  强度 {(dominantScore * 100).toFixed(1)}%
                </span>
              )}
            </p>
          ) : (
            <p className="summary-card__empty">未进行情绪分析</p>
          )}
        </div>

        {mbti && top3.length > 0 ? (
          <MbtiResultCard mbti={mbti} top3={top3} compact />
        ) : (
          <div className="summary-card">
            <h2 className="summary-card__heading">MBTI 预测</h2>
            <p className="summary-card__empty">未进行 MBTI 预测</p>
          </div>
        )}
      </div>

      <button type="button" className="btn btn--cta" onClick={onRestart}>
        重新开始
      </button>
    </section>
  );
}
