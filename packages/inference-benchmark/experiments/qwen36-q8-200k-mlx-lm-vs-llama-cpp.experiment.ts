import { qwen36 } from "../models/qwen3.6-35b-a3b.model"
import { contextSweep, defineExperiment, llamaCpp, mlxLm } from "../src/experiment"

export default defineExperiment({
  id: "qwen36-q8-200k-mlx-lm-vs-llama-cpp",
  title: "Qwen3.6 35B-A3B Q8 at 200K: MLX-LM vs llama.cpp",
  suite: contextSweep({
    checkpoints: [200_000],
    charactersPerToken: 4,
    samplesPerCheckpoint: 1,
  }),
  requestPolicy: {
    contextTokensPerSequence: 229_376,
    parallelSequences: 1,
    maxOutputTokens: 128,
    requestTimeoutMs: 3_600_000,
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
        promptCacheEntries: 4,
        kvCache: { quantization: "none" },
        speculativeDecoding: { kind: "none" },
      }),
    },
  ],
  execution: { variantOrder: "declared", blocks: 1 },
})
