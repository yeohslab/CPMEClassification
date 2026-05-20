export interface VocabData {
  char2id: Record<string, number>;
  max_length: number;
}

const PAD = 0;
const UNK = 1;

export function loadVocab(data: VocabData) {
  const char2id = data.char2id;
  const maxLength = data.max_length;
  return {
    encode(text: string): number[] {
      const normalized = text.replace(/\s+/g, " ").trim();
      const ids: number[] = [];
      for (const ch of normalized.slice(0, maxLength)) {
        ids.push(char2id[ch] ?? UNK);
      }
      while (ids.length < maxLength) ids.push(PAD);
      return ids;
    },
    maxLength,
  };
}

export async function fetchVocab(base: string, file: string) {
  const res = await fetch(`${base}models/${file}`);
  if (!res.ok) throw new Error(`Failed to load vocab ${file}`);
  return loadVocab((await res.json()) as VocabData);
}
