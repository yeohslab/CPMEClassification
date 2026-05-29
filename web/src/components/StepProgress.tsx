export type FlowStep = "landing" | "emotion" | "mbti" | "complete";

const STEPS: { id: FlowStep; label: string }[] = [
  { id: "emotion", label: "情绪分析" },
  { id: "mbti", label: "MBTI 预测" },
  { id: "complete", label: "总结" },
];

interface Props {
  current: FlowStep;
}

function stepIndex(step: FlowStep): number {
  if (step === "landing") return -1;
  return STEPS.findIndex((s) => s.id === step);
}

export default function StepProgress({ current }: Props) {
  if (current === "landing") return null;

  const active = stepIndex(current);

  return (
    <nav className="step-progress" aria-label="体验进度">
      <ol className="step-progress__list">
        {STEPS.map((step, i) => {
          const done = i < active;
          const currentStep = i === active;
          return (
            <li
              key={step.id}
              className={[
                "step-progress__item",
                done ? "step-progress__item--done" : "",
                currentStep ? "step-progress__item--current" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              aria-current={currentStep ? "step" : undefined}
            >
              <span className="step-progress__dot" aria-hidden="true">
                {done ? "✓" : i + 1}
              </span>
              <span className="step-progress__label">{step.label}</span>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
