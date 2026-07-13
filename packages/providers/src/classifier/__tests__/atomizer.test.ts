import { describe, expect, it } from "vitest"
import { atomizeModelId } from "../atomizer"

function atomString(id: string): string {
  return atomizeModelId(id)
    .map((a) => (a.type === "sep" ? "sep" : a.type === "dot" ? "dot" : a.value))
    .join(" ")
}

interface AtomCase {
  readonly id: string
  readonly expected: string
}

const CASES: readonly AtomCase[] = [
  // Path prefix stripping
  { id: "accounts/fireworks/models/glm-5p1", expected: "glm sep 5 dot 1" },
  { id: "z-ai/glm-5", expected: "glm sep 5" },
  { id: "zai-org/GLM-5.1", expected: "glm sep 5 dot 1" },
  { id: "meta-llama/Llama-3.3-70B-Instruct", expected: "llama sep 3 dot 3 sep 70 b sep instruct" },
  { id: "meta.llama3-1-8b-instruct-v1:0", expected: "llama 3 sep 1 sep 8 b sep instruct sep v 1 sep 0" },
  { id: "meta/llama3-3@llama-3.3-70b-instruct", expected: "llama sep 3 dot 3 sep 70 b sep instruct" },

  // Decimal points via `.` and `p`
  { id: "glm-5.2", expected: "glm sep 5 dot 2" },
  { id: "glm5.1", expected: "glm 5 dot 1" },
  { id: "qwen3p5-397b-a17b", expected: "qwen 3 dot 5 sep 397 b sep a 17 b" },
  { id: "qwen/qwen3.5-27b", expected: "qwen 3 dot 5 sep 27 b" },
  { id: "Qwen/Qwen3.5-9B", expected: "qwen 3 dot 5 sep 9 b" },

  // Ollama tags — `:` is a separator, everything after it is kept as atoms
  { id: "qwen3.5:35b", expected: "qwen 3 dot 5 sep 35 b" },
  { id: "qwen3-5@qwen3.5-9b", expected: "qwen 3 dot 5 sep 9 b" },
  { id: "llama3.3:70b", expected: "llama 3 dot 3 sep 70 b" },
  { id: "deepseek-v3:671b", expected: "deepseek sep v 3 sep 671 b" },
  { id: "llama4:latest", expected: "llama 4 sep latest" },
  { id: "mistral:7b", expected: "mistral sep 7 b" },
  { id: "mistral:nemo", expected: "mistral sep nemo" },

  // Non-decimal `.` is a separator (Bedrock provider prefix already stripped)
  { id: "deepseek.deepseek-v3-v1:0", expected: "deepseek sep v 3 sep v 1 sep 0" },

  // Alpha/digit splitting
  { id: "meta-llama/Llama-4-Scout-17B-16E-Instruct", expected: "llama sep 4 sep scout sep 17 b sep 16 e sep instruct" },
  { id: "deepseek-v4-pro", expected: "deepseek sep v 4 sep pro" },
  { id: "deepseek-ai/DeepSeek-V2", expected: "deepseek sep v 2" },
  { id: "deepseek-ai/DeepSeek-R1", expected: "deepseek sep r 1" },
  { id: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", expected: "deepseek sep r 1 sep distill sep llama sep 70 b" },
  { id: "kimi-k2.7", expected: "kimi sep k 2 dot 7" },
  { id: "moonshotai/kimi-k2.7-code", expected: "kimi sep k 2 dot 7 sep code" },
  { id: "moonshotai/Kimi-K2.6", expected: "kimi sep k 2 dot 6" },
  { id: "mistral-7b-instruct", expected: "mistral sep 7 b sep instruct" },
  { id: "mistral-large-2407-v1:0", expected: "mistral sep large sep 2407 sep v 1 sep 0" },
  { id: "mistral-nemo", expected: "mistral sep nemo" },
  { id: "mixtral-moe-8x22b-instruct", expected: "mixtral sep moe sep 8 x 22 b sep instruct" },
  { id: "mistral-large-3-675b-instruct-2512", expected: "mistral sep large sep 3 sep 675 b sep instruct sep 2512" },
  { id: "ministral-3-14b-instruct-2512", expected: "ministral sep 3 sep 14 b sep instruct sep 2512" },
  { id: "phi-3p5-vision-instruct", expected: "phi sep 3 dot 5 sep vision sep instruct" },
  { id: "gpt-oss-120b", expected: "gpt sep oss sep 120 b" },
  { id: "gemma-4-e4b", expected: "gemma sep 4 sep e 4 b" },
]

describe("atomizer", () => {
  for (const { id, expected } of CASES) {
    it(`atomizes ${id}`, () => {
      expect(atomString(id)).toBe(expected)
    })
  }
})
