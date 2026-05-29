import { useEffect, useState } from "react";
import CompleteSummary from "./components/CompleteSummary";
import EmotionPanel from "./components/EmotionPanel";
import LandingHero from "./components/LandingHero";
import MbtiPanel from "./components/MbtiPanel";
import SiteFooter from "./components/SiteFooter";
import SiteHeader from "./components/SiteHeader";
import type { FlowStep } from "./components/StepProgress";
import { loadModels } from "./inference";

export default function App() {
  const [step, setStep] = useState<FlowStep>("landing");
  const [status, setStatus] = useState("正在加载模型…");
  const [ready, setReady] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [dominantEmotion, setDominantEmotion] = useState<string | null>(null);
  const [dominantScore, setDominantScore] = useState<number | null>(null);
  const [mbtiResult, setMbtiResult] = useState<string | null>(null);
  const [mbtiTop3, setMbtiTop3] = useState<{ type: string; prob: number }[]>(
    []
  );

  useEffect(() => {
    loadModels(setStatus)
      .then(() => {
        setReady(true);
        setStatus("模型已就绪");
      })
      .catch((e) => {
        setLoadError(
          e instanceof Error
            ? e.message
            : "模型加载失败。请先运行训练并导出 ONNX 到 web/public/models/"
        );
        setStatus("");
      });
  }, []);

  const handleRestart = () => {
    setDominantEmotion(null);
    setDominantScore(null);
    setMbtiResult(null);
    setMbtiTop3([]);
    setStep("landing");
  };

  const showModelStatus = step !== "landing" && step !== "complete";

  return (
    <div className="app">
      {step !== "landing" && (
        <SiteHeader step={step} onHome={handleRestart} />
      )}

      <main className="app__main">
        {step === "landing" && (
          <LandingHero onStart={() => setStep("emotion")} />
        )}

        {step === "emotion" && (
          <div className="step-page">
            {showModelStatus && (
              <ModelStatus
                ready={ready}
                loadError={loadError}
                status={status}
              />
            )}
            <EmotionPanel
              modelsReady={ready}
              onResult={(d, s) => {
                setDominantEmotion(d);
                setDominantScore(s);
              }}
            />
            <div className="step-actions">
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => setStep("mbti")}
              >
                跳过，直接去 MBTI
              </button>
              <button
                type="button"
                className="btn btn--cta"
                onClick={() => setStep("mbti")}
                disabled={!ready}
              >
                下一步：MBTI 预测
              </button>
            </div>
          </div>
        )}

        {step === "mbti" && (
          <div className="step-page">
            {showModelStatus && (
              <ModelStatus
                ready={ready}
                loadError={loadError}
                status={status}
              />
            )}
            <MbtiPanel
              modelsReady={ready}
              onResult={(m, t) => {
                setMbtiResult(m);
                setMbtiTop3(t);
              }}
            />
            <div className="step-actions">
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => setStep("emotion")}
              >
                上一步
              </button>
              <button
                type="button"
                className="btn btn--cta"
                onClick={() => setStep("complete")}
              >
                查看总结
              </button>
            </div>
          </div>
        )}

        {step === "complete" && (
          <CompleteSummary
            dominantEmotion={dominantEmotion}
            dominantScore={dominantScore}
            mbti={mbtiResult}
            top3={mbtiTop3}
            onRestart={handleRestart}
          />
        )}
      </main>

      <SiteFooter />
    </div>
  );
}

function ModelStatus({
  ready,
  loadError,
  status,
}: {
  ready: boolean;
  loadError: string | null;
  status: string;
}) {
  if (loadError) {
    return (
      <p className="model-status model-status--error" role="alert">
        {loadError}
      </p>
    );
  }
  if (!ready) {
    return (
      <p className="model-status" aria-live="polite">
        <span className="spinner" aria-hidden="true" />
        {status || "正在加载模型…"}
      </p>
    );
  }
  return null;
}
