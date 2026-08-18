import { agentCore, defineExperiment, llamaCpp, mlxVlm, type ModelArtifactDefinition } from "../src/experiment"

export const defineMtpComparison = (options: {
  readonly id: string
  readonly title: string
  readonly llamaTarget: ModelArtifactDefinition
  readonly llamaDrafter: ModelArtifactDefinition
  readonly mlxTarget: ModelArtifactDefinition
  readonly mlxDrafter: ModelArtifactDefinition
}) => defineExperiment({
  id: options.id,
  title: options.title,
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
      id: "llama-cpp-mtp",
      artifact: options.llamaTarget,
      engine: llamaCpp({
        executable: "managed",
        flashAttention: true,
        continuousBatching: true,
        kvCache: { quantization: "none" },
        speculativeDecoding: { kind: "mtp", draftArtifact: options.llamaDrafter, maxDraftTokens: 2 },
      }),
    },
    {
      id: "mlx-vlm-mtp",
      artifact: options.mlxTarget,
      engine: mlxVlm({
        pythonProject: "../engines/mlx-vlm",
        prefillStepSize: 2_048,
        kvCache: { quantization: "none" },
        speculativeDecoding: { kind: "mtp", draftArtifact: options.mlxDrafter, maxDraftTokens: 2 },
      }),
    },
  ],
  execution: { variantOrder: "balanced", blocks: 2 },
})
