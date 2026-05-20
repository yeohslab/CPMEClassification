import { useEffect, useState } from "react";
import EmotionPanel from "./components/EmotionPanel";
import MbtiPanel from "./components/MbtiPanel";
import { loadModels } from "./inference";

type Tab = "emotion" | "mbti";

export default function App() {
  const [tab, setTab] = useState<Tab>("emotion");
  const [status, setStatus] = useState("正在加载模型…");
  const [ready, setReady] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

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

  return (
    <>
      <h1>CPME 情绪 & MBTI 推理</h1>
      <p className="subtitle">
        基于 CPME 数据集训练 · 浏览器内 ONNX 推理（GitHub Pages）
      </p>
      <p className={`status ${loadError ? "error" : ""}`}>
        {loadError ?? status}
      </p>
      <div className="tabs">
        <button
          className={tab === "emotion" ? "active" : ""}
          onClick={() => setTab("emotion")}
        >
          单帖情绪
        </button>
        <button
          className={tab === "mbti" ? "active" : ""}
          onClick={() => setTab("mbti")}
        >
          多帖 MBTI
        </button>
      </div>
      {tab === "emotion" ? (
        <EmotionPanel modelsReady={ready} />
      ) : (
        <MbtiPanel modelsReady={ready} />
      )}
      <p className="disclaimer">
        免责声明：本工具为基于统计学习的演示模型，结果仅供研究娱乐，不构成心理测评或人格诊断建议。
      </p>
    </>
  );
}
