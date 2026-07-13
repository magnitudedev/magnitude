import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { classifyModelFamily } from "../../family-registry"

interface ClassifyCase {
  readonly id: string
  readonly familyId: string
}

const CASES: readonly ClassifyCase[] = [
  // GLM-5
  { id: "glm-5.2", familyId: "glm-5" },
  { id: "accounts/fireworks/models/glm-5p1", familyId: "glm-5" },
  { id: "z-ai/glm-5", familyId: "glm-5" },
  { id: "zai-org/GLM-5.1", familyId: "glm-5" },
  { id: "glm5.1", familyId: "glm-5" },

  // GLM-4
  { id: "glm-4p5", familyId: "glm-4" },
  { id: "glm-4.5", familyId: "glm-4" },
  { id: "glm-4.7-flash", familyId: "glm-4" },

  // Qwen 3.5
  { id: "qwen3p5-397b-a17b", familyId: "qwen-3.5" },
  { id: "qwen3.5-27b", familyId: "qwen-3.5" },
  { id: "Qwen/Qwen3.5-9B", familyId: "qwen-3.5" },
  { id: "qwen3.5:35b", familyId: "qwen-3.5" },
  { id: "qwen3.6-35b-a3b", familyId: "qwen-3.5" },
  { id: "qwen3.6-plus", familyId: "qwen-3.5" },
  { id: "qwen3.7-max", familyId: "qwen-3.5" },

  // Qwen 3
  { id: "qwen3:32b", familyId: "qwen-3" },
  { id: "qwen3-coder:480b", familyId: "qwen-3" },

  // Qwen 2.5
  { id: "Qwen/Qwen2.5-72B-Instruct", familyId: "qwen-2.5" },
  { id: "qwen2.5:7b", familyId: "qwen-2.5" },

  // Llama 3
  { id: "meta-llama/Llama-3.3-70B-Instruct", familyId: "llama-3" },
  { id: "accounts/fireworks/models/llama-v3p3-70b-instruct", familyId: "llama-3" },
  { id: "meta.llama3-1-8b-instruct-v1:0", familyId: "llama-3" },
  { id: "llama3.3:70b", familyId: "llama-3" },
  { id: "meta/llama3-3@llama-3.3-70b-instruct", familyId: "llama-3" },
  { id: "meta-llama/Meta-Llama-3-8B-Instruct-Lite", familyId: "llama-3" },
  { id: "deepseek-r1-distill-llama-70b", familyId: "llama-3" },
  { id: "nvidia/nemotron-3-ultra-550b-a55b", familyId: "llama-3" },

  // Llama 4
  { id: "meta-llama/Llama-4-Scout-17B-16E-Instruct", familyId: "llama-4" },
  { id: "llama4-scout-instruct-basic", familyId: "llama-4" },
  { id: "meta.llama-4-scout-17b-16e-instruct-v1:0", familyId: "llama-4" },
  { id: "llama4:latest", familyId: "llama-4" },

  // DeepSeek V3
  { id: "deepseek-v4-pro", familyId: "deepseek-v3" },
  { id: "deepseek/deepseek-v4-pro", familyId: "deepseek-v3" },
  { id: "deepseek-ai/DeepSeek-V4-Pro", familyId: "deepseek-v3" },
  { id: "deepseek.deepseek-v3-v1:0", familyId: "deepseek-v3" },
  { id: "deepseek-v3:671b", familyId: "deepseek-v3" },
  { id: "deepseek-ai/DeepSeek-R1", familyId: "deepseek-v3" },
  { id: "deepseek/deepseek-r1-0528", familyId: "deepseek-v3" },

  // DeepSeek V2
  { id: "deepseek-ai/DeepSeek-V2", familyId: "deepseek-v2" },

  // Kimi K2
  { id: "kimi-k2.7", familyId: "kimi-k2" },
  { id: "moonshotai/kimi-k2.7-code", familyId: "kimi-k2" },
  { id: "moonshotai/Kimi-K2.6", familyId: "kimi-k2" },

  // GPT-OSS
  { id: "gpt-oss-120b", familyId: "gpt-oss" },
  { id: "openai/gpt-oss:120b", familyId: "gpt-oss" },
  { id: "gpt-oss-20b", familyId: "gpt-oss" },

  // Mistral SPM
  { id: "mistral:7b", familyId: "mistral-v3-spm" },
  { id: "mistral-7b-instruct", familyId: "mistral-v3-spm" },
  { id: "mistral-large-2407-v1:0", familyId: "mistral-v3-spm" },
  { id: "mixtral-moe-8x22b-instruct", familyId: "mistral-v3-spm" },
  { id: "codestral-22b", familyId: "mistral-v3-spm" },

  // Mistral Tekken
  { id: "mistral:nemo", familyId: "mistral-v3-tekken" },
  { id: "mistral-small-24b-instruct-2501", familyId: "mistral-v3-tekken" },
  { id: "mistral-large-3-675b-instruct-2512", familyId: "mistral-v3-tekken" },
  { id: "ministral-3-14b-instruct-2512", familyId: "mistral-v3-tekken" },
  { id: "codestral-mamba-7b", familyId: "mistral-v3-tekken" },
  { id: "pixtral-12b", familyId: "mistral-v3-tekken" },
  { id: "devstral-small", familyId: "mistral-v3-tekken" },

  // Gemma
  { id: "google/gemma-3-27b-it", familyId: "gemma-3" },
  { id: "gemma3:27b", familyId: "gemma-3" },
  { id: "google/gemma-4-31b-it", familyId: "gemma-4" },
  { id: "gemma-4-e4b", familyId: "gemma-4" },
  { id: "google/gemma-2-9b-it", familyId: "gemma-2" },

  // Phi
  { id: "phi-3-mini-128k-instruct", familyId: "phi-3" },
  { id: "phi-3p5-vision-instruct", familyId: "phi-3" },
  { id: "phi3:medium", familyId: "phi-3" },
  { id: "phi-4", familyId: "phi-4" },

  // MiniMax
  { id: "minimax-m2", familyId: "minimax-m2" },
  { id: "minimax-m2.1", familyId: "minimax-m2" },
  { id: "minimax-m3", familyId: "minimax-m2" },

  // Others
  { id: "command-r", familyId: "command-r" },
  { id: "exaone-4", familyId: "exaone-4" },
  { id: "hunyuan-large", familyId: "hunyuan-large" },
  { id: "hunyuan-a13b", familyId: "hunyuan-a13b" },
  { id: "yi-6b", familyId: "yi" },
  { id: "grok-1", familyId: "grok-1" },
  { id: "starcoder2", familyId: "starcoder2" },
  { id: "falcon-3", familyId: "falcon-3" },
  { id: "falcon-2", familyId: "falcon-2" },
  { id: "dbrx", familyId: "dbrx" },
  { id: "olmo-2", familyId: "olmo-2" },
  { id: "granite", familyId: "granite" },
  { id: "internlm-2", familyId: "internlm-2" },
]

describe("classifyModelFamily", () => {
  for (const { id, familyId } of CASES) {
    it(`classifies ${id} as ${familyId}`, () => {
      expect(Option.getOrElse(classifyModelFamily(id), () => null)).toBe(familyId)
    })
  }

  it("returns None for unknown models", () => {
    expect(Option.isNone(classifyModelFamily("unknown-model-xyz"))).toBe(true)
  })
})
