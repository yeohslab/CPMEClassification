import { DECOR } from "../mbtiMeta";

interface Props {
  onStart: () => void;
}

export default function LandingHero({ onStart }: Props) {
  return (
    <section className="hero" aria-labelledby="hero-title">
      <div className="hero__bg" aria-hidden="true">
        <img src={DECOR.heroMountains} alt="" />
      </div>
      <div className="hero__content">
        <p className="hero__eyebrow">CPME · 浏览器内 ONNX 推理</p>
        <h1 id="hero-title" className="hero__title">
          读懂文字里的情绪，
          <br />
          探索发帖背后的人格倾向
        </h1>
        <p className="hero__subtitle">
          基于中文微博风格数据训练的双模块演示：单帖六维情绪强度，多帖
          MBTI 类型预测。界面灵感来自人格类型框架，仅供研究与娱乐。
        </p>
        <button type="button" className="btn btn--cta" onClick={onStart}>
          开始体验
        </button>
        <ul className="hero__stats" aria-label="项目特点">
          <li>
            <strong>6</strong>
            <span>情绪维度</span>
          </li>
          <li>
            <strong>16</strong>
            <span>MBTI 类型</span>
          </li>
          <li>
            <strong>本地</strong>
            <span>隐私友好推理</span>
          </li>
        </ul>
      </div>
    </section>
  );
}
