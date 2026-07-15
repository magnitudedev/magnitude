import type { LocalModelCatalogEntry } from "./types"

const APACHE_LICENSE = {
  id: "apache-2.0",
  url: "https://www.apache.org/licenses/LICENSE-2.0",
  acknowledgementRequired: false,
} as const

const UNSLOTH_FIDELITY_SOURCE = "https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs"
const QUANT_FIDELITY_STUDY = "https://arxiv.org/abs/2606.19558"

/**
 * Cross-model guidance for an Unsloth Dynamic quant tier when no measurement of
 * the exact catalog artifact has been published. These labels intentionally say
 * "fidelity", not "accuracy": the cited study found that KLD is useful for
 * detecting badly degraded quants, but cannot rank downstream task quality once
 * candidates are already close to the reference.
 */
const QUANT_TIER_FIDELITY = {
  q4: {
    rank: 40,
    label: "Good fidelity with some possible quality loss",
    evidence: "Cross-model guidance, not a measurement of this exact artifact. In a published Devstral cohort, UD-Q4_K_XL had KLD 0.0200 and composite score 0.695 and remained in the study's near-baseline group. Exact results can differ by model and calibration.",
  },
  q5: {
    rank: 50,
    label: "High fidelity with only minor quality loss",
    evidence: "Cross-model guidance, not a measurement of this exact artifact. In a published Devstral cohort, UD-Q5_K_XL had KLD 0.0072 and composite score 0.692 and remained in the study's near-baseline group. Lower KLD means less distribution drift, not guaranteed higher task accuracy.",
  },
  q6: {
    rank: 60,
    label: "Very high fidelity with minimal quality loss",
    evidence: "Cross-model guidance, not a measurement of this exact artifact. In a published Devstral cohort, UD-Q6_K_XL had KLD 0.0019 and composite score 0.694 and remained in the study's near-baseline group. Exact results can differ by model and calibration.",
  },
  q8: {
    rank: 80,
    label: "Near-original fidelity with the least quality loss",
    evidence: "Cross-model guidance, not a measurement of this exact artifact. In a published Devstral cohort, UD-Q8_K_XL had KLD 0.0009 and composite score 0.700 and remained in the study's near-baseline group. This describes closeness to BF16 outputs, not coding accuracy.",
  },
} as const

export const conventionalQuantFidelityLabel = (
  bitsClass: "q4" | "q5" | "q6" | "q8",
): string => QUANT_TIER_FIDELITY[bitsClass].label

interface ArtifactInput {
  readonly modelId: string
  readonly family: string
  readonly displayName: string
  readonly architecture: "dense" | "moe"
  readonly totalParametersBillions: number
  readonly activeParametersBillions?: number
  readonly modelMaximumContextTokens: number
  readonly repo: string
  readonly revision: string
  readonly file: string
  readonly sizeBytes: number
  readonly sha256: string
  readonly format: string
  readonly bitsClass: "q4" | "q5" | "q6" | "q8"
  readonly quantAwareCheckpoint?: boolean
  readonly fidelityRank: number
  readonly fidelityLabel: string
  readonly fidelityEvidence: string
  readonly fidelitySourceUrl?: string
  readonly modelQualityRank: number
}

const artifact = (input: ArtifactInput): LocalModelCatalogEntry => ({
  id: `${input.modelId}:${input.format}`,
  modelId: input.modelId,
  family: input.family,
  displayName: input.displayName,
  architecture: input.architecture,
  totalParametersBillions: input.totalParametersBillions,
  ...(input.activeParametersBillions !== undefined
    ? { activeParametersBillions: input.activeParametersBillions }
    : {}),
  modelMaximumContextTokens: input.modelMaximumContextTokens,
  supportedContextTokens: [16_384, 32_768, 65_536, 131_072],
  repo: input.repo,
  revision: input.revision,
  quantTag: input.format,
  files: [{ path: input.file, sizeBytes: input.sizeBytes, sha256: input.sha256 }],
  quantization: {
    format: input.format,
    bitsClass: input.bitsClass,
    quantAwareCheckpoint: input.quantAwareCheckpoint ?? false,
    fidelityRank: input.fidelityRank,
    fidelityLabel: input.fidelityLabel,
    fidelityEvidence: input.fidelityEvidence,
    fidelitySourceUrl: input.fidelitySourceUrl ?? UNSLOTH_FIDELITY_SOURCE,
  },
  license: APACHE_LICENSE,
  modelQualityRank: input.modelQualityRank,
})

const qwenConventional = (
  common: Omit<ArtifactInput, "file" | "sizeBytes" | "sha256" | "format" | "bitsClass" | "fidelityRank" | "fidelityLabel" | "fidelityEvidence">,
  variants: readonly [format: "UD-Q4_K_XL" | "UD-Q5_K_XL" | "UD-Q6_K_XL" | "UD-Q8_K_XL", size: number, sha256: string][],
): LocalModelCatalogEntry[] => variants.map(([format, sizeBytes, sha256]) => {
  const bitsClass = format.slice(3, 5).toLowerCase() as "q4" | "q5" | "q6" | "q8"
  const fidelity = QUANT_TIER_FIDELITY[bitsClass]
  return artifact({
    ...common,
    file: `${common.displayName.replaceAll(" ", "-")}-${format}.gguf`,
    sizeBytes,
    sha256,
    format,
    bitsClass,
    fidelityRank: fidelity.rank,
    fidelityLabel: fidelity.label,
    fidelityEvidence: fidelity.evidence,
    fidelitySourceUrl: QUANT_FIDELITY_STUDY,
  })
})

const QWEN_4B = qwenConventional({
  modelId: "qwen3.5-4b",
  family: "qwen3.5",
  displayName: "Qwen3.5 4B",
  architecture: "dense",
  totalParametersBillions: 4,
  modelMaximumContextTokens: 262_144,
  repo: "unsloth/Qwen3.5-4B-GGUF",
  revision: "e87f176479d0855a907a41277aca2f8ee7a09523",
  modelQualityRank: 10,
}, [
  ["UD-Q4_K_XL", 2_912_109_728, "b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7"],
  ["UD-Q5_K_XL", 3_250_869_408, "b4c36a8e14a80c21bcab5a067ce342b2e70e28f60b4aa95ad12203fa17b87426"],
  ["UD-Q6_K_XL", 4_145_548_448, "87f58d94410b81429268d8389a3d686e6c6bffecf7852772720fbea059cbbb9d"],
  ["UD-Q8_K_XL", 5_952_048_288, "e786a3c6570474c3885199bfb5adc54325aa7521a314e10b0aaefe16a54ba42f"],
])

const QWEN_9B = qwenConventional({
  modelId: "qwen3.5-9b",
  family: "qwen3.5",
  displayName: "Qwen3.5 9B",
  architecture: "dense",
  totalParametersBillions: 9,
  modelMaximumContextTokens: 262_144,
  repo: "unsloth/Qwen3.5-9B-GGUF",
  revision: "3885219b6810b007914f3a7950a8d1b469d598a5",
  modelQualityRank: 20,
}, [
  ["UD-Q4_K_XL", 5_966_095_584, "6f5d30666c2d8ae16a306e616d95341dcf3cc46810df84d7e6f5a7d1e4c1b293"],
  ["UD-Q5_K_XL", 6_743_680_224, "96cf42ddb97f9572410a72b9ed6f2299b1e887ee08da4c2a6c01e897cfa9f673"],
  ["UD-Q6_K_XL", 8_756_929_760, "33b0050fb9c19abcf815647a78464dad959a06dadaecb0b96af798669f9074d4"],
  ["UD-Q8_K_XL", 12_974_040_288, "2c4e08e0e72c68d8c1835a26f5be4075894df9ea5be9cc20a246517afd6a0cb6"],
])

const QWEN_27B = qwenConventional({
  modelId: "qwen3.6-27b",
  family: "qwen3.6",
  displayName: "Qwen3.6 27B",
  architecture: "dense",
  totalParametersBillions: 27,
  modelMaximumContextTokens: 262_144,
  repo: "unsloth/Qwen3.6-27B-GGUF",
  revision: "82d411acf4a06cfb8d9b073a5211bf410bfc29bf",
  modelQualityRank: 45,
}, [
  ["UD-Q4_K_XL", 17_612_564_704, "ff6941ded525b34eb159496762c29dd0ec6e71dc31b74d57e75d871a03eec259"],
  ["UD-Q5_K_XL", 20_038_256_864, "ac310abf2895aa397121bad6c0be89466af41f0f1606a21c1131b110eeb19d0e"],
  ["UD-Q6_K_XL", 25_636_485_344, "8746881d40f280b1b6b858c656a347c754ed3d9cc8d2e1ad46b3635b87f611f8"],
  ["UD-Q8_K_XL", 35_325_163_744, "19a2f4733a863088bc06665bf307dca95f7d4370b4d8690340cdff9992fe48c6"],
])

const QWEN_35B = qwenConventional({
  modelId: "qwen3.6-35b-a3b",
  family: "qwen3.6",
  displayName: "Qwen3.6 35B-A3B",
  architecture: "moe",
  totalParametersBillions: 35,
  activeParametersBillions: 3,
  modelMaximumContextTokens: 262_144,
  repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
  revision: "a483e9e6cbd595906af30beda3187c2663a1118c",
  modelQualityRank: 50,
}, [
  ["UD-Q4_K_XL", 22_360_456_160, "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450"],
  ["UD-Q5_K_XL", 26_592_508_896, "25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506"],
  ["UD-Q6_K_XL", 31_843_777_504, "f6b6c6d5cfa6f00d964eeb7add28eb14ce7481734d506b90681007678cd2c484"],
  ["UD-Q8_K_XL", 38_451_182_560, "b762215c5f507f4865df4ac3d1afa803828afa41e05ecac3fac431a67bbd88e8"],
]).map((entry) => {
  if (entry.quantization.bitsClass !== "q4" && entry.quantization.bitsClass !== "q5") return entry

  const isQ4 = entry.quantization.bitsClass === "q4"
  return {
    ...entry,
    quantization: {
      ...entry.quantization,
      fidelityRank: isQ4 ? 40 : 50,
      fidelityLabel: isQ4
        ? "Good fidelity in model-specific testing"
        : "High fidelity in model-specific testing",
      fidelityEvidence: isQ4
        ? "Exact-artifact measurement: UD-Q4_K_XL had KLD 0.0135 against BF16 and composite benchmark score 0.728, placing it in the study's near-baseline group. KLD measures output-distribution drift and does not rank task accuracy within that group."
        : "Exact-artifact measurement: UD-Q5_K_XL had KLD 0.0082 against BF16 and composite benchmark score 0.724, placing it in the study's near-baseline group. Its lower KLD than Q4 did not produce a higher composite score, so KLD is not an accuracy ranking.",
      fidelitySourceUrl: QUANT_FIDELITY_STUDY,
    },
  }
})

const GEMMA_QAT: LocalModelCatalogEntry[] = [
  artifact({
    modelId: "gemma-4-26b-a4b-it-qat",
    family: "gemma-4",
    displayName: "Gemma 4 26B-A4B",
    architecture: "moe",
    totalParametersBillions: 26,
    activeParametersBillions: 4,
    modelMaximumContextTokens: 262_144,
    repo: "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
    revision: "c1f25db7cf31985b52caa1db777eb72d17ca1c7c",
    file: "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    sizeBytes: 14_249_045_120,
    sha256: "dcf179a91153e3a7ece792e48ef872180d9d6ef9b7677f0a0bd3e83cfe624d5e",
    format: "UD-Q4_K_XL",
    bitsClass: "q4",
    quantAwareCheckpoint: true,
    fidelityRank: 58,
    fidelityLabel: "Near-original fidelity from quantization-aware training",
    fidelityEvidence: "Checkpoint-level evidence: Google describes Gemma 4 QAT as preserving similar quality to BF16. This file is Unsloth's UD-Q4_K_XL conversion of those QAT weights; no KLD or downstream delta is published for this exact 26B artifact.",
    fidelitySourceUrl: "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF",
    modelQualityRank: 34,
  }),
  artifact({
    modelId: "gemma-4-31b-it-qat",
    family: "gemma-4",
    displayName: "Gemma 4 31B",
    architecture: "dense",
    totalParametersBillions: 31,
    modelMaximumContextTokens: 262_144,
    repo: "unsloth/gemma-4-31B-it-qat-GGUF",
    revision: "1f1e54258d4a2cf7522856a5789045d9f2ea6d16",
    file: "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
    sizeBytes: 17_287_668_064,
    sha256: "9188a71055550f1e60b875d02b7abb63625ac11b4a6f148d6b22b3b28ba3d335",
    format: "UD-Q4_K_XL",
    bitsClass: "q4",
    quantAwareCheckpoint: true,
    fidelityRank: 58,
    fidelityLabel: "Near-original fidelity from quantization-aware training",
    fidelityEvidence: "Checkpoint-level evidence: Google describes Gemma 4 QAT as preserving similar quality to BF16. This file is Unsloth's UD-Q4_K_XL conversion of those QAT weights; no KLD or downstream delta is published for this exact 31B artifact.",
    fidelitySourceUrl: "https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF",
    modelQualityRank: 36,
  }),
]

const NEMOTRON_SUPER: LocalModelCatalogEntry = {
  id: "nemotron-3-super-120b-a12b:MXFP4_MOE",
  modelId: "nemotron-3-super-120b-a12b",
  family: "nemotron-3",
  displayName: "NVIDIA Nemotron 3 Super 120B-A12B",
  architecture: "moe",
  totalParametersBillions: 120,
  activeParametersBillions: 12,
  modelMaximumContextTokens: 131_072,
  supportedContextTokens: [32_768],
  repo: "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
  revision: "036038fb30334a2d56a146c6f0d4871ab5edccbb",
  quantTag: "MXFP4_MOE",
  files: [
    {
      path: "MXFP4_MOE/NVIDIA-Nemotron-3-Super-120B-A12B-MXFP4_MOE-00001-of-00003.gguf",
      sizeBytes: 7_872_576,
      sha256: "1891b15a4e3e11f5dfb0050062c35fc2eeee9e7a37bc8f407c3886491c21bb30",
    },
    {
      path: "MXFP4_MOE/NVIDIA-Nemotron-3-Super-120B-A12B-MXFP4_MOE-00002-of-00003.gguf",
      sizeBytes: 49_831_407_232,
      sha256: "62ead0ba22b3a4a4ac9277d6845fa09c7860933dd0060c51819c2b751d086923",
    },
    {
      path: "MXFP4_MOE/NVIDIA-Nemotron-3-Super-120B-A12B-MXFP4_MOE-00003-of-00003.gguf",
      sizeBytes: 32_220_461_216,
      sha256: "cfb638b699befba01199d70a4e44ace39cb7f144d9edc651eadd369f6d43c60c",
    },
  ],
  quantization: {
    format: "MXFP4_MOE",
    bitsClass: "mxfp4",
    quantAwareCheckpoint: false,
    fidelityRank: 70,
    fidelityLabel: "Near-original fidelity in benchmark comparisons",
    fidelityEvidence: "Checkpoint-level benchmark evidence: NVIDIA reports NVFP4 versus BF16 scores of 78.44 versus 78.69 on LiveCodeBench v6, 83.33 versus 83.73 on MMLU-Pro, and 79.42 versus 79.23 on GPQA. NVIDIA pre-trained most linear layers in NVFP4 while retaining selected layers in BF16 or MXFP8. This Unsloth MXFP4_MOE GGUF is a separate conversion, so the Details view must preserve that distinction rather than presenting these as exact-GGUF measurements.",
    fidelitySourceUrl: "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
  },
  license: {
    id: "other",
    url: "https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
    acknowledgementRequired: false,
  },
  modelQualityRank: 65,
}

const GLM_52: LocalModelCatalogEntry = {
  id: "glm-5.2:UD-Q4_K_XL",
  modelId: "glm-5.2",
  family: "glm-5.2",
  displayName: "GLM 5.2",
  architecture: "moe",
  totalParametersBillions: 753,
  modelMaximumContextTokens: 1_048_576,
  supportedContextTokens: [32_768],
  repo: "unsloth/GLM-5.2-GGUF",
  revision: "abc55e72527792c6e77069c99b4cb7de16fa9f23",
  quantTag: "UD-Q4_K_XL",
  files: [
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf", 9_423_744, "3256ac8c290273f0965ff39e93a8bcd07dc99bcd23e923bd4b7306ef39061038"],
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00002-of-00011.gguf", 49_433_942_336, "aaedfb89d314d6967a80005b93a9c460a494babc6c3e4f0138e21891e21572e1"],
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00003-of-00011.gguf", 48_566_415_136, "a2b45b63075b2e1bc8a73c9ce531ccea54c03001286a80f77454871aa93fdca8"],
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00004-of-00011.gguf", 48_566_415_136, "b5404d8d17b63e127aa34c1f98cef64d3722050d8ef1a0792dba816477f4c606"],
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00005-of-00011.gguf", 48_566_415_136, "9ab79e1947115be35da815c1be2812a1451d3ec11f9f5d60dd3ba152e1ed5be2"],
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00006-of-00011.gguf", 48_566_415_136, "43a2631ee392492f8857bae6c88660e0f1cac0fd6bc40d832538ac5421b3167b"],
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00007-of-00011.gguf", 48_566_415_136, "1efd96717a956a160a1717999c7dedbe601b5787ea6220d8185d232e95eff893"],
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00008-of-00011.gguf", 48_566_415_136, "3460334e8148d12402c8f5adf684b132918504bbea4d3aecd74801121e8c8a99"],
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00009-of-00011.gguf", 48_566_415_136, "7f6be8ce1c9dcb973ede026b7341657f8add8617f386f77cc165ff697cf9620d"],
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00010-of-00011.gguf", 48_566_415_136, "6a26bf391e6f1de947e63016d11ada565f7476a06cb90b444f6db334baa949f9"],
    ["UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00011-of-00011.gguf", 29_314_424_736, "27032b927daa606872d887c56631c5278a788b39d219784b262e1df3d4cb851e"],
  ].map(([path, sizeBytes, sha256]) => ({
    path: path as string,
    sizeBytes: sizeBytes as number,
    sha256: sha256 as string,
  })),
  quantization: {
    format: "UD-Q4_K_XL",
    bitsClass: "q4",
    quantAwareCheckpoint: false,
    // Ranking remains at the Q4 tier. The 97.5% measurement is not comparable
    // to other models' BF16-referenced metrics and must not inflate ordering.
    fidelityRank: 40,
    fidelityLabel: "Near-original fidelity in quantization tests",
    fidelityEvidence: "Exact-artifact measurement: Unsloth plots UD-Q4_K_XL at about 97.5% top-1 token agreement with Q8_0 defined as 100%. This is next-token agreement against Q8_0, not BF16 fidelity or coding accuracy; no numeric KLD or downstream delta is tabulated.",
    fidelitySourceUrl: "https://huggingface.co/unsloth/GLM-5.2-GGUF/discussions/3",
  },
  license: {
    id: "mit",
    url: "https://huggingface.co/unsloth/GLM-5.2-GGUF",
    acknowledgementRequired: false,
  },
  modelQualityRank: 80,
}

export const LOCAL_MODEL_CATALOG: readonly LocalModelCatalogEntry[] = [
  ...QWEN_4B,
  ...QWEN_9B,
  ...GEMMA_QAT,
  ...QWEN_27B,
  ...QWEN_35B,
  NEMOTRON_SUPER,
  GLM_52,
]

export const LOCAL_MODEL_CATALOG_BY_ID = new Map(LOCAL_MODEL_CATALOG.map((entry) => [entry.id, entry]))

export const catalogFileUrl = (entry: LocalModelCatalogEntry, path: string): string =>
  `https://huggingface.co/${entry.repo}/resolve/${entry.revision}/${path.split("/").map(encodeURIComponent).join("/")}`

export const catalogSourcePageUrl = (entry: LocalModelCatalogEntry): string =>
  `https://huggingface.co/${entry.repo}/blob/${entry.revision}/${entry.files[0]!.path.split("/").map(encodeURIComponent).join("/")}`
