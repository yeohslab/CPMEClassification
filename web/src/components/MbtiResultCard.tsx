import { getMbtiProfile, ROLE_LABELS } from "../mbtiMeta";

interface Props {
  mbti: string;
  top3: { type: string; prob: number }[];
  compact?: boolean;
}

export default function MbtiResultCard({ mbti, top3, compact }: Props) {
  const profile = getMbtiProfile(mbti);
  const role = profile?.role ?? "sentinel";
  const roleLabel = ROLE_LABELS[role];

  return (
    <div
      className={`mbti-card${compact ? " mbti-card--compact" : ""}`}
      data-role={role}
    >
      <div className="mbti-card__visual">
        {profile && (
          <img
            src={profile.illustration}
            alt={`${profile.nicknameZh}（${mbti}）`}
            className="mbti-card__img"
          />
        )}
      </div>
      <div className="mbti-card__body">
        <span className="mbti-card__role">
          {roleLabel.zh} · {roleLabel.en}
        </span>
        <h2 className="mbti-card__type">{mbti}</h2>
        {profile && (
          <p className="mbti-card__nickname">
            {profile.nicknameZh}
            <span className="mbti-card__nickname-en"> {profile.nickname}</span>
          </p>
        )}
        {profile?.illustrationPlaceholder && (
          <p className="mbti-card__placeholder-note">角色插画待补全（INFJ）</p>
        )}
        <div className="mbti-card__probs" aria-label="Top-3 类型概率">
          {top3.map((t) => {
            const p = getMbtiProfile(t.type);
            return (
              <div className="mbti-prob-row" key={t.type}>
                <span className="mbti-prob-row__label">
                  {t.type}
                  {p ? ` · ${p.nicknameZh}` : ""}
                </span>
                <div className="mbti-prob-row__track">
                  <div
                    className="mbti-prob-row__fill"
                    style={{ width: `${t.prob * 100}%` }}
                  />
                </div>
                <span className="mbti-prob-row__pct">
                  {(t.prob * 100).toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
