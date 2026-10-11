import { qwen36 } from "../models/qwen3.6-35b-a3b.model"
import { agentCore, defineExperiment, llamaCpp, mlxLm } from "../src/experiment"

export default defineExperiment({
  id: "qwen36-q8-mlx-lm-vs-llama-cpp",
  title: "Qwen3.6 35B-A3B: MLX-LM 8-bit vs llama.cpp Q8_0",
  suite: agentCore({ profile: "smoke" }),
  requestPolicy: {
    contextTokensPerSequence: 8_192,
    parallelSequences: 1,
    maxOutputTokens: 128,
    temperature: 0,
    topP: 1,
    seed: 42,
    enableThinking: false,
  },
  variants: [
    {
      id: "llama-cpp-q8-0",
      artifact: qwen36.artifacts.llamaQ8,
      engine: llamaCpp({
        executable: "managed",
        flashAttention: true,
        continuousBatching: true,
        kvCache: { quantization: "none" },
        speculativeDecoding: { kind: "none" },
      }),
    },
    {
      id: "mlx-lm-8bit",
      artifact: qwen36.artifacts.mlx8,
      engine: mlxLm({
        pythonProject: "../engines/mlx-lm",
        prefillStepSize: 2_048,
        promptCacheEntries: 32,
        kvCache: { quantization: "none" },
        speculativeDecoding: { kind: "none" },
      }),
    },
  ],
  execution: { variantOrder: "balanced", blocks: 2 },
})
