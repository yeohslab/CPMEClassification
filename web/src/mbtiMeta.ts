export type MbtiRole = "analyst" | "diplomat" | "sentinel" | "explorer";

export interface MbtiProfile {
  nickname: string;
  nicknameZh: string;
  role: MbtiRole;
  illustration: string;
  illustrationPlaceholder?: boolean;
}

const base = import.meta.env.BASE_URL;

function asset(path: string): string {
  return `${base}assets/${path}`;
}

export const ROLE_LABELS: Record<MbtiRole, { en: string; zh: string }> = {
  analyst: { en: "Analysts", zh: "分析家" },
  diplomat: { en: "Diplomats", zh: "外交家" },
  sentinel: { en: "Sentinels", zh: "守护者" },
  explorer: { en: "Explorers", zh: "探险家" },
};

export const MBTI_PROFILE: Record<string, MbtiProfile> = {
  INTJ: {
    nickname: "Architect",
    nicknameZh: "建筑师",
    role: "analyst",
    illustration: asset("architect.svg"),
  },
  INTP: {
    nickname: "Logician",
    nicknameZh: "逻辑学家",
    role: "analyst",
    illustration: asset("logician.svg"),
  },
  ENTJ: {
    nickname: "Commander",
    nicknameZh: "指挥官",
    role: "analyst",
    illustration: asset("commander.svg"),
  },
  ENTP: {
    nickname: "Debater",
    nicknameZh: "辩论家",
    role: "analyst",
    illustration: asset("debater.svg"),
  },
  INFJ: {
    nickname: "Advocate",
    nicknameZh: "提倡者",
    role: "diplomat",
    illustration: asset("complete-lg.svg"),
    illustrationPlaceholder: true,
  },
  INFP: {
    nickname: "Mediator",
    nicknameZh: "调停者",
    role: "diplomat",
    illustration: asset("mediator.svg"),
  },
  ENFJ: {
    nickname: "Protagonist",
    nicknameZh: "主人公",
    role: "diplomat",
    illustration: asset("protagonist.svg"),
  },
  ENFP: {
    nickname: "Campaigner",
    nicknameZh: "竞选者",
    role: "diplomat",
    illustration: asset("campaigner.svg"),
  },
  ISTJ: {
    nickname: "Logistician",
    nicknameZh: "物流师",
    role: "sentinel",
    illustration: asset("logistician.svg"),
  },
  ISFJ: {
    nickname: "Defender",
    nicknameZh: "守卫者",
    role: "sentinel",
    illustration: asset("defender.svg"),
  },
  ESTJ: {
    nickname: "Executive",
    nicknameZh: "总经理",
    role: "sentinel",
    illustration: asset("executive.svg"),
  },
  ESFJ: {
    nickname: "Consul",
    nicknameZh: "执政官",
    role: "sentinel",
    illustration: asset("consul.svg"),
  },
  ISTP: {
    nickname: "Virtuoso",
    nicknameZh: "鉴赏家",
    role: "explorer",
    illustration: asset("virtuoso.svg"),
  },
  ISFP: {
    nickname: "Adventurer",
    nicknameZh: "探险家",
    role: "explorer",
    illustration: asset("adventurer.svg"),
  },
  ESTP: {
    nickname: "Entrepreneur",
    nicknameZh: "企业家",
    role: "explorer",
    illustration: asset("entrepreneur.svg"),
  },
  ESFP: {
    nickname: "Entertainer",
    nicknameZh: "表演者",
    role: "explorer",
    illustration: asset("entertainer.svg"),
  },
};

export function getMbtiProfile(type: string): MbtiProfile | undefined {
  return MBTI_PROFILE[type.toUpperCase()];
}

export const DECOR = {
  heroMountains: asset("header-mountains-desktop.svg"),
  emotionSide: asset("potential-lg.svg"),
  complete: asset("complete-lg.svg"),
  results: asset("results-lg.svg"),
  career: asset("career_values.svg"),
} as const;
