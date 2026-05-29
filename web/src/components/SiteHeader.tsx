import StepProgress, { type FlowStep } from "./StepProgress";

interface Props {
  step: FlowStep;
  onHome?: () => void;
}

export default function SiteHeader({ step, onHome }: Props) {
  return (
    <header className="site-header">
      <div className="site-header__inner">
        <button
          type="button"
          className="site-header__brand"
          onClick={onHome}
          aria-label="返回首页"
        >
          <span className="site-header__logo" aria-hidden="true">
            <span className="dot dot--a" />
            <span className="dot dot--b" />
            <span className="dot dot--c" />
            <span className="dot dot--d" />
          </span>
          <span className="site-header__title">CPME 人格情绪 Lab</span>
        </button>
        <StepProgress current={step} />
      </div>
    </header>
  );
}
