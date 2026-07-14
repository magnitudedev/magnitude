import { Option } from "effect"
import type { ModelFamily } from "@magnitudedev/ai"
import { dot, lit, num, opt, sep, ver } from "./classifier/symbols"
import type { Family } from "./classifier/classify"
import { classify } from "./classifier/classify"

const withVision = (id: string): ModelFamily => ({
  id,
  capabilities: { vision: true },
})

const withoutVision = (id: string): ModelFamily => ({
  id,
  capabilities: { vision: false },
})

const FAMILY_DEFINITIONS: readonly Family[] = [
  {
    familyId: "llama-3",
    patterns: [
      { pattern: [lit("llama"), sep(), opt("v"), lit("3"), dot(), ver()], priority: 100 },
      { pattern: [lit("llama"), sep(), opt("v"), lit("3")], priority: 90 },
      { pattern: [lit("distill"), sep(), lit("llama")], priority: 95 },
      { pattern: [lit("nemotron")], priority: 95 },
    ],
  },
  {
    familyId: "llama-4",
    patterns: [
      { pattern: [lit("llama"), sep(), lit("4")], priority: 100 },
    ],
  },
  {
    familyId: "qwen-2.5",
    patterns: [
      { pattern: [lit("qwen"), sep(), lit("2"), dot(), ver()], priority: 100 },
      { pattern: [lit("qwen"), sep(), lit("2")], priority: 90 },
      { pattern: [lit("qwen"), sep(), lit("1"), dot(), ver()], priority: 100 },
      { pattern: [lit("qwen"), sep(), lit("1")], priority: 90 },
      { pattern: [lit("distill"), sep(), lit("qwen")], priority: 95 },
    ],
  },
  {
    familyId: "qwen-3",
    patterns: [
      { pattern: [lit("qwen"), sep(), lit("3")], priority: 80 },
    ],
  },
  {
    familyId: "qwen-3.5",
    patterns: [
      { pattern: [lit("qwen"), sep(), lit("3"), dot(), lit("5")], priority: 100 },
      { pattern: [lit("qwen"), sep(), lit("3"), dot(), lit("6")], priority: 100 },
      { pattern: [lit("qwen"), sep(), lit("3"), dot(), lit("7")], priority: 100 },
      { pattern: [lit("qwen"), sep(), lit("3"), dot(), ver()], priority: 90 },
    ],
    metadataPatterns: [{
      architectures: ["qwen35", "qwen35moe"],
      tokenizerModels: ["gpt2"],
      tokenizerPres: ["qwen35"],
    }],
  },
  {
    familyId: "deepseek-v3",
    patterns: [
      { pattern: [lit("deepseek"), sep(), lit("v"), lit("3"), dot(), ver()], priority: 100 },
      { pattern: [lit("deepseek"), sep(), lit("v"), lit("3")], priority: 90 },
      { pattern: [lit("deepseek"), sep(), lit("v"), lit("4")], priority: 100 },
      {
        pattern: [lit("deepseek"), sep(), lit("r"), lit("1")],
        priority: 100,
        exclude: [lit("distill")],
      },
    ],
  },
  {
    familyId: "glm-5",
    patterns: [
      { pattern: [lit("glm"), sep(), lit("5"), dot(), lit("1")], priority: 100 },
      { pattern: [lit("glm"), sep(), lit("5"), dot(), lit("2")], priority: 100 },
      { pattern: [lit("glm"), sep(), lit("5"), dot(), ver()], priority: 90 },
      { pattern: [lit("glm"), sep(), lit("5")], priority: 80 },
      { pattern: [lit("glm")], priority: 50 },
    ],
  },
  {
    familyId: "glm-4",
    patterns: [
      { pattern: [lit("glm"), sep(), lit("4"), dot(), lit("5")], priority: 100 },
      { pattern: [lit("glm"), sep(), lit("4"), dot(), lit("6")], priority: 100 },
      { pattern: [lit("glm"), sep(), lit("4"), dot(), lit("7")], priority: 100 },
      { pattern: [lit("glm"), sep(), lit("4"), dot(), ver()], priority: 90 },
      { pattern: [lit("glm"), sep(), lit("4")], priority: 80 },
    ],
  },
  {
    familyId: "kimi-k2",
    patterns: [
      { pattern: [lit("kimi"), sep(), lit("k"), lit("2"), dot(), ver()], priority: 100 },
      { pattern: [lit("kimi"), sep(), lit("k"), lit("2")], priority: 90 },
    ],
  },
  {
    familyId: "gpt-oss",
    patterns: [
      { pattern: [lit("gpt"), sep(), lit("oss")], priority: 100 },
    ],
  },
  {
    familyId: "mistral-v3-tekken",
    patterns: [
      { pattern: [lit("mistral"), sep(), lit("nemo")], priority: 100 },
      { pattern: [lit("mistral"), sep(), lit("small")], priority: 100 },
      { pattern: [lit("mistral"), sep(), lit("large"), sep(), lit("3")], priority: 100 },
      { pattern: [lit("ministral")], priority: 100 },
      { pattern: [lit("devstral")], priority: 100 },
      { pattern: [lit("pixtral")], priority: 100 },
      { pattern: [lit("codestral"), sep(), lit("mamba")], priority: 100 },
    ],
  },
  {
    familyId: "mistral-v3-spm",
    patterns: [
      { pattern: [lit("mixtral")], priority: 100 },
      { pattern: [lit("mistral"), sep(), lit("7")], priority: 90 },
      { pattern: [lit("mistral"), sep(), lit("large"), sep(), lit("2")], priority: 85 },
      { pattern: [lit("mistral"), sep(), lit("large"), sep(), num()], priority: 80 },
      { pattern: [lit("codestral"), sep(), lit("22")], priority: 80 },
    ],
  },
  {
    familyId: "gemma-2",
    patterns: [
      { pattern: [lit("gemma"), sep(), lit("2"), dot(), ver()], priority: 100 },
      { pattern: [lit("gemma"), sep(), lit("2")], priority: 90 },
      { pattern: [lit("gemma"), sep(), lit("1")], priority: 90 },
    ],
  },
  {
    familyId: "gemma-3",
    patterns: [
      { pattern: [lit("gemma"), sep(), lit("3")], priority: 100 },
    ],
  },
  {
    familyId: "gemma-4",
    patterns: [
      { pattern: [lit("gemma"), sep(), lit("4")], priority: 100 },
    ],
  },
  {
    familyId: "phi-3",
    patterns: [
      { pattern: [lit("phi"), sep(), lit("3"), dot(), ver()], priority: 100 },
      { pattern: [lit("phi"), sep(), lit("3")], priority: 90 },
    ],
  },
  {
    familyId: "phi-4",
    patterns: [
      { pattern: [lit("phi"), sep(), lit("4")], priority: 100 },
    ],
  },
  {
    familyId: "minimax-m2",
    patterns: [
      { pattern: [lit("minimax"), sep(), lit("m"), lit("2")], priority: 100 },
      { pattern: [lit("minimax"), sep(), lit("m"), lit("3")], priority: 90 },
    ],
  },
  {
    familyId: "command-r",
    patterns: [
      { pattern: [lit("command")], priority: 100 },
    ],
  },
  {
    familyId: "deepseek-v2",
    patterns: [
      { pattern: [lit("deepseek"), sep(), lit("v"), lit("2")], priority: 90 },
    ],
  },
  {
    familyId: "exaone-4",
    patterns: [
      { pattern: [lit("exaone")], priority: 100 },
    ],
  },
  {
    familyId: "hunyuan-large",
    patterns: [
      { pattern: [lit("hunyuan"), sep(), lit("large")], priority: 100 },
    ],
  },
  {
    familyId: "hunyuan-a13b",
    patterns: [
      { pattern: [lit("hunyuan"), sep(), lit("a"), lit("13")], priority: 100 },
    ],
  },
  {
    familyId: "yi",
    patterns: [
      { pattern: [lit("yi")], priority: 100 },
    ],
  },
  {
    familyId: "grok-1",
    patterns: [
      { pattern: [lit("grok")], priority: 100 },
    ],
  },
  {
    familyId: "starcoder2",
    patterns: [
      { pattern: [lit("starcoder")], priority: 100 },
    ],
  },
  {
    familyId: "falcon-3",
    patterns: [
      { pattern: [lit("falcon"), sep(), lit("3")], priority: 100 },
    ],
  },
  {
    familyId: "falcon-2",
    patterns: [
      { pattern: [lit("falcon"), sep(), lit("2")], priority: 100 },
      { pattern: [lit("falcon")], priority: 50 },
    ],
  },
  {
    familyId: "dbrx",
    patterns: [
      { pattern: [lit("dbrx")], priority: 100 },
    ],
  },
  {
    familyId: "olmo-2",
    patterns: [
      { pattern: [lit("olmo")], priority: 100 },
    ],
  },
  {
    familyId: "granite",
    patterns: [
      { pattern: [lit("granite")], priority: 100 },
    ],
  },
  {
    familyId: "internlm-2",
    patterns: [
      { pattern: [lit("internlm")], priority: 100 },
    ],
  },
]

const VISION_FAMILY_IDS = new Set<string>([
  "glm-5",
  "kimi-k2",
  "qwen-3.5",
  "gpt-oss",
  "minimax-m2",
  "llama-4",
  "gemma-3",
  "gemma-4",
  "phi-3",
  "mistral-v3-tekken",
  "qwen-3",
  "internlm-2",
])

export const MODEL_FAMILIES: readonly ModelFamily[] = FAMILY_DEFINITIONS.map(
  (family): ModelFamily =>
    VISION_FAMILY_IDS.has(family.familyId)
      ? withVision(family.familyId)
      : withoutVision(family.familyId),
)

const FAMILY_BY_ID = new Map<string, ModelFamily>(MODEL_FAMILIES.map((f) => [f.id, f]))

export function getModelFamily(id: string): ModelFamily | null {
  return FAMILY_BY_ID.get(id) ?? null
}

export function classifyModelFamily(id: string): Option.Option<string> {
  const result = classify(id, FAMILY_DEFINITIONS)
  return result.matched ? Option.some(result.familyId) : Option.none()
}

export interface ModelFamilyMetadata {
  readonly architecture?: string
  readonly tokenizerModel?: string
  readonly tokenizerPre?: string
}

function normalizeMetadataValue(value: string | undefined): string | undefined {
  const normalized = value?.trim().toLowerCase()
  return normalized || undefined
}

function metadataPatternMatches(
  metadata: ModelFamilyMetadata,
  pattern: NonNullable<Family["metadataPatterns"]>[number],
): boolean {
  const architecture = normalizeMetadataValue(metadata.architecture)
  const tokenizerModel = normalizeMetadataValue(metadata.tokenizerModel)
  const tokenizerPre = normalizeMetadataValue(metadata.tokenizerPre)

  return (
    (!pattern.architectures || (architecture !== undefined && pattern.architectures.includes(architecture))) &&
    (!pattern.tokenizerModels || (tokenizerModel !== undefined && pattern.tokenizerModels.includes(tokenizerModel))) &&
    (!pattern.tokenizerPres || (tokenizerPre !== undefined && pattern.tokenizerPres.includes(tokenizerPre)))
  )
}

/** Classify only when all required structured metadata for a family agrees. */
export function classifyModelFamilyFromMetadata(
  metadata: ModelFamilyMetadata,
): Option.Option<string> {
  for (const family of FAMILY_DEFINITIONS) {
    if (family.metadataPatterns?.some((pattern) => metadataPatternMatches(metadata, pattern))) {
      return Option.some(family.familyId)
    }
  }
  return Option.none()
}

/** Reject a name-based match when present structured metadata contradicts it. */
export function modelFamilyMetadataConflicts(
  familyId: string,
  metadata: ModelFamilyMetadata,
): boolean {
  const family = FAMILY_DEFINITIONS.find((candidate) => candidate.familyId === familyId)
  if (!family?.metadataPatterns?.length) return false

  const architecture = normalizeMetadataValue(metadata.architecture)
  const tokenizerModel = normalizeMetadataValue(metadata.tokenizerModel)
  const tokenizerPre = normalizeMetadataValue(metadata.tokenizerPre)

  return family.metadataPatterns.every((pattern) =>
    (architecture !== undefined && pattern.architectures !== undefined && !pattern.architectures.includes(architecture)) ||
    (tokenizerModel !== undefined && pattern.tokenizerModels !== undefined && !pattern.tokenizerModels.includes(tokenizerModel)) ||
    (tokenizerPre !== undefined && pattern.tokenizerPres !== undefined && !pattern.tokenizerPres.includes(tokenizerPre)),
  )
}

export { FAMILY_DEFINITIONS }
