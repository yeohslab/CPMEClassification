import * as ort from "onnxruntime-web";
import { fetchVocab } from "./tokenizer";

const BASE = import.meta.env.BASE_URL;

export const EMOTION_LABELS = [
  "angry",
  "fear",
  "happy",
  "neutral",
  "sad",
  "surprise",
] as const;

export const EMOTION_LABELS_ZH: Record<string, string> = {
  angry: "愤怒",
  fear: "恐惧",
  happy: "喜悦",
  neutral: "中性",
  sad: "悲伤",
  surprise: "惊讶",
};

export const MBTI_TYPES = [
  "ENFJ", "ENFP", "ENTJ", "ENTP",
  "ESFJ", "ESFP", "ESTJ", "ESTP",
  "INFJ", "INFP", "INTJ", "INTP",
  "ISFJ", "ISFP", "ISTJ", "ISTP",
] as const;

interface EmotionMeta {
  max_length: number;
  labels: string[];
  vocab: string;
}

interface MbtiMeta {
  max_length: number;
  max_posts: number;
  labels: string[];
  vocab: string;
}

let emotionSession: ort.InferenceSession | null = null;
let mbtiSession: ort.InferenceSession | null = null;
let emotionMeta: EmotionMeta | null = null;
let mbtiMeta: MbtiMeta | null = null;
let emotionVocab: ReturnType<typeof fetchVocab> extends Promise<infer T> ? T : never;
let mbtiVocab: typeof emotionVocab;

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

export async function loadModels(
  onProgress?: (msg: string) => void
): Promise<void> {
  onProgress?.("Loading metadata...");
  emotionMeta = await fetchJson<EmotionMeta>("models/emotion_meta.json");
  mbtiMeta = await fetchJson<MbtiMeta>("models/mbti_meta.json");

  onProgress?.("Loading vocabulary...");
  emotionVocab = await fetchVocab(BASE, emotionMeta.vocab);
  mbtiVocab = await fetchVocab(BASE, mbtiMeta.vocab);

  onProgress?.("Loading emotion ONNX...");
  emotionSession = await ort.InferenceSession.create(`${BASE}models/emotion.onnx`, {
    executionProviders: ["wasm"],
  });

  onProgress?.("Loading MBTI ONNX...");
  mbtiSession = await ort.InferenceSession.create(`${BASE}models/mbti.onnx`, {
    executionProviders: ["wasm"],
  });

  onProgress?.("Ready");
}

export async function predictEmotion(text: string): Promise<Record<string, number>> {
  if (!emotionSession || !emotionMeta || !emotionVocab) throw new Error("Models not loaded");
  const ids = new BigInt64Array(emotionVocab.encode(text).map((x) => BigInt(x)));
  const input = new ort.Tensor("int64", ids, [1, emotionMeta.max_length]);
  const out = await emotionSession.run({ input_ids: input });
  const scores = out.scores!.data as Float32Array;
  const result: Record<string, number> = {};
  EMOTION_LABELS.forEach((label, i) => {
    result[label] = Math.round(scores[i] * 1000) / 1000;
  });
  return result;
}

export async function predictMbti(posts: string[]): Promise<{
  mbti: string;
  probs: { type: string; prob: number }[];
}> {
  if (!mbtiSession || !mbtiMeta || !mbtiVocab) throw new Error("Models not loaded");
  const maxPosts = mbtiMeta.max_posts;
  const valid = posts.filter((p) => p.trim()).slice(0, maxPosts);
  if (valid.length === 0) throw new Error("Enter at least one post");

  const padded = [...valid];
  while (padded.length < maxPosts) padded.push("");

  const matrix = padded.map((p) => mbtiVocab.encode(p));
  const flat = matrix.flat();
  const ids = new BigInt64Array(flat.map((x) => BigInt(x)));
  const postMask = new Float32Array(maxPosts);
  for (let i = 0; i < maxPosts; i++) postMask[i] = i < valid.length ? 1 : 0;

  const feeds = {
    input_ids: new ort.Tensor("int64", ids, [1, maxPosts, mbtiMeta.max_length]),
    post_mask: new ort.Tensor("float32", postMask, [1, maxPosts]),
  };
  const out = await mbtiSession.run(feeds);
  const probs = out.probs!.data as Float32Array;
  const ranked = MBTI_TYPES.map((type, i) => ({ type, prob: probs[i] }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 3);
  return { mbti: ranked[0]!.type, probs: ranked };
}
