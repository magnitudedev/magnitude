import { qwen36 } from "../models/qwen3.6-35b-a3b.model"
import { agentCore, defineExperiment, llamaCpp, mlxLm } from "../src/experiment"

export default defineExperiment({
  id: "qwen36-q4-mlx-lm-vs-llama-cpp",
  title: "Qwen3.6 35B-A3B: MLX-LM 4-bit vs llama.cpp UD-Q4_K_XL",
  suite: agentCore({ profile: "smoke" }),
  requestPolicy: {
    contextTokensPerSequence: 8_192, parallelSequences: 1, maxOutputTokens: 128,
    temperature: 0, topP: 1, seed: 42, enableThinking: false,
  },
  variants: [
    {
      id: "llama-cpp-ud-q4-k-xl", artifact: qwen36.artifacts.llamaQ4,
      engine: llamaCpp({ executable: "managed", flashAttention: true, continuousBatching: true, kvCache: { quantization: "none" }, speculativeDecoding: { kind: "none" } }),
    },
    {
      id: "mlx-lm-4bit", artifact: qwen36.artifacts.mlx4,
      engine: mlxLm({ pythonProject: "../engines/mlx-lm", prefillStepSize: 2_048, promptCacheEntries: 32, kvCache: { quantization: "none" }, speculativeDecoding: { kind: "none" } }),
    },
  ],
  execution: { variantOrder: "balanced", blocks: 2 },
})
