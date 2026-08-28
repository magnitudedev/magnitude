import { describe, expect, it } from "vitest"
import { mkdtemp, readdir, rm, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { comparisonKindFor } from "../src/benchmark"
import { defineExperiment, defineModel, defineSpeculativeDecodingComparison, existingEndpoint, icn, llamaCpp, loadExperiment, mlxLm, mlxVlm, omlx, resolveExecutionOrder, resolveExperimentPaths } from "../src/experiment"

describe("TypeScript experiments", () => {
  it("loads every checked-in experiment through the current schema", async () => {
    const paths = (await readdir("experiments"))
      .filter((path) => path.endsWith(".experiment.ts"))
      .map((path) => `experiments/${path}`)
    expect(paths.length).toBeGreaterThan(0)
    const program = [
      'import { BunContext } from "@effect/platform-bun"',
      'import { Effect } from "effect"',
      'import { loadExperiment } from "./src/experiment"',
      'const loaded = await Effect.runPromise(loadExperiment(process.argv[1]).pipe(Effect.provide(BunContext.layer)))',
      'if (loaded.comparisonProtocol.kind !== "fixed-speculative-policy") process.exit(2)',
    ].join(";")
    for (const path of paths) {
      const result = Bun.spawnSync(["bun", "-e", program, path])
      expect(result.exitCode, `${path}: ${result.stderr.toString()}`).toBe(0)
    }
  })

  it("builds a serializable model/artifact/engine experiment", () => {
    const model = defineModel({
      id: "model",
      contextLimit: 8_192,
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

  it("represents ICN and existing endpoints as experiment engines", () => {
    const model = defineModel({
      id: "model",
      contextLimit: 8_192,
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
        { id: "icn", artifact: model.artifacts.gguf, engine: icn({ executable: "managed" }) },
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
      contextLimit: 8_192,
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

  it("represents controlled oMLX baseline, embedded speculation, and DFlash variants", async () => {
    const model = defineModel({
      id: "model", contextLimit: 16_384,
      source: { repository: "owner/base", revision: "base" },
      artifacts: {
        target: {
          kind: "mlx", repository: "owner/target", revision: "target", manifest: "target.lock.json",
          quantization: { family: "mlx-affine", bits: 4, groupSize: 64 },
        },
        dflash: {
          kind: "mlx", repository: "owner/dflash", revision: "draft", manifest: "draft.lock.json",
          quantization: { family: "mlx-unquantized", dtype: "bfloat16" },
        },
        gguf: {
          kind: "gguf", repository: "owner/draft", revision: "draft", file: "draft.gguf",
          sizeBytes: 1, sha256: "b".repeat(64), quantization: { family: "gguf", scheme: "F16" },
        },
      },
    })
    const common = { pythonProject: "../engines/omlx", cache: { kind: "disabled" as const }, memoryGuard: { kind: "off" as const } }
    expect(() => defineExperiment({
      id: "mixed-runtime", title: "Mixed runtime", suite: { kind: "agent-core", profile: "smoke" },
      requestPolicy: { contextTokensPerSequence: 4096, parallelSequences: 1, maxOutputTokens: 32, temperature: 0, topP: 1, seed: 42, enableThinking: false },
      variants: [
        { id: "omlx", artifact: model.artifacts.target, engine: omlx({ ...common, speculativeDecoding: { kind: "none" } }) },
        { id: "mlx-lm", artifact: model.artifacts.target, engine: mlxLm({ pythonProject: "../engines/mlx-lm", prefillStepSize: 2048, promptCacheEntries: 1, kvCache: { quantization: "none" }, speculativeDecoding: { kind: "none" } }) },
      ],
      execution: { variantOrder: "balanced", blocks: 2 },
    })).toThrow("cannot be mixed")
    expect(() => defineExperiment({
      id: "mixed-omlx-projects", title: "Mixed oMLX projects", suite: { kind: "agent-core", profile: "smoke" },
      requestPolicy: { contextTokensPerSequence: 4096, parallelSequences: 1, maxOutputTokens: 32, temperature: 0, topP: 1, seed: 42, enableThinking: false },
      variants: [
        { id: "first", artifact: model.artifacts.target, engine: omlx({ ...common, speculativeDecoding: { kind: "none" } }) },
        { id: "second", artifact: model.artifacts.target, engine: omlx({ ...common, pythonProject: "../engines/other-omlx", speculativeDecoding: { kind: "none" } }) },
      ],
      execution: { variantOrder: "balanced", blocks: 2 },
    })).toThrow("one frozen Python project")
    const experiment = defineSpeculativeDecodingComparison({
      id: "acceleration", title: "Acceleration",
      suite: { kind: "agent-core", profile: "smoke" },
      requestPolicy: { contextTokensPerSequence: 4096, parallelSequences: 1, maxOutputTokens: 32, temperature: 0, topP: 1, seed: 42, enableThinking: false },
      variants: [
        { id: "baseline", artifact: model.artifacts.target, engine: omlx({ ...common, speculativeDecoding: { kind: "none" } }) },
        { id: "mtp", artifact: model.artifacts.target, engine: omlx({ ...common, speculativeDecoding: { kind: "mtp", maxDraftTokens: 3 } }) },
        { id: "dspark", artifact: model.artifacts.target, engine: omlx({ ...common, speculativeDecoding: { kind: "dspark", maxDraftTokens: 3 } }) },
        { id: "dflash", artifact: model.artifacts.target, engine: omlx({ ...common, speculativeDecoding: { kind: "dflash", draftArtifact: model.artifacts.dflash, blockSize: 4 } }) },
      ],
      execution: { variantOrder: "balanced", blocks: 4 },
    })
    expect(resolveExecutionOrder(experiment)).toEqual([
      ["baseline", "mtp", "dspark", "dflash"],
      ["mtp", "dspark", "dflash", "baseline"],
      ["dspark", "dflash", "baseline", "mtp"],
      ["dflash", "baseline", "mtp", "dspark"],
    ])
    expect(experiment.comparisonProtocol.kind).toBe("speculative-decoding")
    expect(comparisonKindFor(experiment.comparisonProtocol.kind, [])).toBe("controlled-speculative-decoding")

    const temporary = await mkdtemp(join("test", ".speculative-experiment-"))
    try {
      const path = join(temporary, "comparison.ts")
      await writeFile(path, `export default ${JSON.stringify(experiment)}\n`)
      const loaded = await Effect.runPromise(loadExperiment(path).pipe(Effect.provide(BunContext.layer)))
      expect(loaded.comparisonProtocol.kind).toBe("speculative-decoding")
    } finally {
      await rm(temporary, { recursive: true, force: true })
    }
    const resolved = resolveExperimentPaths(experiment, "/bench/experiments/example.ts")
    expect((resolved.variants[0]?.engine as { pythonProject: string }).pythonProject).toBe("/bench/engines/omlx")
    expect(((resolved.variants[3]?.engine as Extract<typeof experiment.variants[number]["engine"], { kind: "omlx" }>).speculativeDecoding as { draftArtifact: { manifest: string } }).draftArtifact.manifest).toBe("/bench/experiments/draft.lock.json")

    expect(() => omlx({ ...common, speculativeDecoding: { kind: "dflash", draftArtifact: model.artifacts.gguf, blockSize: 4 } } as never)).not.toThrow()
    expect(() => defineSpeculativeDecodingComparison({
      ...experiment,
      variants: [
        { id: "baseline", artifact: model.artifacts.target, engine: omlx({ ...common, speculativeDecoding: { kind: "none" } }) },
        { id: "bad", artifact: model.artifacts.target, engine: omlx({ ...common, speculativeDecoding: { kind: "dflash", draftArtifact: model.artifacts.gguf as never, blockSize: 4 } }) },
      ],
      execution: { variantOrder: "balanced", blocks: 2 },
    })).toThrow("oMLX DFlash requires an MLX draft artifact")

    expect(() => defineSpeculativeDecodingComparison({
      ...experiment,
      variants: [
        { id: "baseline", artifact: model.artifacts.target, engine: omlx({ ...common, speculativeDecoding: { kind: "none" } }) },
        { id: "bad-depth", artifact: model.artifacts.target, engine: omlx({ ...common, speculativeDecoding: { kind: "mtp", maxDraftTokens: 0 } }) },
      ],
      execution: { variantOrder: "balanced", blocks: 2 },
    })).toThrow()
    expect(() => defineSpeculativeDecodingComparison({
      ...experiment,
      requestPolicy: { ...experiment.requestPolicy, parallelSequences: 2 },
    })).toThrow("parallelSequences=1")
    expect(() => defineSpeculativeDecodingComparison({
      ...experiment,
      variants: experiment.variants.slice(0, 2),
      execution: { variantOrder: "declared", blocks: 2 },
    })).toThrow("require balanced blocks")
    expect(() => defineSpeculativeDecodingComparison({
      ...experiment,
      variants: experiment.variants.slice(0, 2),
      execution: { variantOrder: "balanced", blocks: 3 },
    })).toThrow("divisible by the variant count")
    expect(() => defineSpeculativeDecodingComparison({
      ...experiment,
      variants: [
        experiment.variants[0]!,
        { ...experiment.variants[1]!, artifact: model.artifacts.dflash },
      ],
      execution: { variantOrder: "balanced", blocks: 2 },
    })).toThrow("exact same target artifact")
  })
})
