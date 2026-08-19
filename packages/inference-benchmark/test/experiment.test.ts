import { describe, expect, it } from "vitest"
import { defineExperiment, defineModel, existingEndpoint, icn, llamaCpp, mlxLm, mlxVlm, resolveExecutionOrder } from "../src/experiment"

describe("TypeScript experiments", () => {
  it("builds a serializable model/artifact/engine experiment", () => {
    const model = defineModel({
      id: "model",
      source: { repository: "owner/base", revision: "base-revision" },
      artifacts: {
        gguf: {
          kind: "gguf", repository: "owner/gguf", revision: "revision", file: "model.gguf",
          sizeBytes: 1, sha256: "a".repeat(64), quantization: { family: "gguf", scheme: "Q8_0" },
        },
        mlx: {
          kind: "mlx", repository: "owner/mlx", revision: "revision", manifest: "model.lock.json",
          quantization: { family: "mlx-affine", bits: 8, groupSize: 64 },
        },
      },
    })
    const experiment = defineExperiment({
      id: "experiment", title: "Experiment", suite: { kind: "agent-core", profile: "smoke" },
      requestPolicy: {
        contextTokensPerSequence: 4096, parallelSequences: 1, maxOutputTokens: 32,
        temperature: 0, topP: 1, seed: 42, enableThinking: false,
      },
      variants: [
        { id: "llama", artifact: model.artifacts.gguf, engine: llamaCpp({ executable: "managed", flashAttention: true, continuousBatching: true, kvCache: { quantization: "none" }, speculativeDecoding: { kind: "none" } }) },
        { id: "mlx", artifact: model.artifacts.mlx, engine: mlxLm({ pythonProject: "engine", prefillStepSize: 2048, promptCacheEntries: 4, kvCache: { quantization: "none" }, speculativeDecoding: { kind: "none" } }) },
      ],
      execution: { variantOrder: "balanced", blocks: 2 },
    })
    expect(experiment.variants.map(({ artifact }) => artifact.modelId)).toEqual(["model", "model"])
    expect(experiment.requestPolicy.requestTimeoutMs).toBe(300_000)
    expect(experiment.variants[0]?.artifact.modelSource).toEqual({ repository: "owner/base", revision: "base-revision" })
    expect(resolveExecutionOrder(experiment)).toEqual([["llama", "mlx"], ["mlx", "llama"]])
  })

  it("rejects executable values", () => {
    expect(() => defineExperiment({ value: () => undefined } as never)).toThrow("non-serializable")
  })

  it("represents ICN and existing endpoints as ordinary experiment engines", () => {
    const model = defineModel({
      id: "model",
      source: { repository: "owner/base", revision: "base" },
      artifacts: {
        gguf: {
          kind: "gguf", repository: "owner/gguf", revision: "artifact", file: "model.gguf",
          sizeBytes: 1, sha256: "a".repeat(64), quantization: { family: "gguf", scheme: "Q4_K_M" },
        },
      },
    })
    const common = {
      title: "Target forms", suite: { kind: "agent-core" as const, profile: "smoke" as const },
      requestPolicy: {
        contextTokensPerSequence: 4096, parallelSequences: 1, maxOutputTokens: 32,
        temperature: 0, topP: 1, seed: 42, enableThinking: false as const,
      },
      execution: { variantOrder: "declared" as const, blocks: 1 },
    }
    const experiment = defineExperiment({
      ...common,
      id: "target-forms",
      variants: [
        { id: "icn", artifact: model.artifacts.gguf, engine: icn({ executable: "managed", speculativeDecoding: { kind: "none" } }) },
        {
          id: "remote", artifact: model.artifacts.gguf,
          engine: existingEndpoint({ endpoint: "http://127.0.0.1:8080", authentication: { kind: "none" }, requestBody: {} }),
        },
      ],
    })
    expect(experiment.variants.map(({ engine }) => engine.kind)).toEqual(["icn", "existing-endpoint"])
  })

  it("requires one semantic MTP draft limit across comparable engines", () => {
    const model = defineModel({
      id: "model", source: { repository: "owner/base", revision: "base" },
      artifacts: {
        gguf: {
          kind: "gguf", repository: "owner/gguf", revision: "artifact", file: "model.gguf",
          sizeBytes: 1, sha256: "a".repeat(64), quantization: { family: "gguf", scheme: "Q8_0" },
        },
        mlx: {
          kind: "mlx", repository: "owner/mlx", revision: "artifact", manifest: "model.lock.json",
          quantization: { family: "mlx-unquantized", dtype: "bfloat16" },
        },
      },
    })
    expect(() => defineExperiment({
      id: "mismatched-mtp", title: "Mismatched MTP", suite: { kind: "agent-core", profile: "smoke" },
      requestPolicy: {
        contextTokensPerSequence: 4096, parallelSequences: 1, maxOutputTokens: 32,
        temperature: 0, topP: 1, seed: 42, enableThinking: false,
      },
      variants: [
        { id: "llama", artifact: model.artifacts.gguf, engine: llamaCpp({ executable: "managed", flashAttention: true, continuousBatching: true, kvCache: { quantization: "none" }, speculativeDecoding: { kind: "mtp", draftArtifact: model.artifacts.gguf, maxDraftTokens: 2 } }) },
        { id: "mlx", artifact: model.artifacts.mlx, engine: mlxVlm({ pythonProject: "engine", prefillStepSize: 2048, kvCache: { quantization: "none" }, speculativeDecoding: { kind: "mtp", draftArtifact: model.artifacts.mlx, maxDraftTokens: 3 } }) },
      ],
      execution: { variantOrder: "balanced", blocks: 2 },
    })).toThrow("same maximum draft-token count")
  })
})
