import * as BunContext from "@effect/platform-bun/BunContext"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelFileFormatId,
  ModelFileId,
  ModelFilePartId,
  ModelFileSourceId,
  SourceFileKey,
  type ModelFileRecord,
} from "../model-files"
import {
  BatchSize,
  ContextSize,
  FlashAttentionSelection,
  GpuLayerSelection,
  LlamaBuildNumber,
  LlamaCppExecutableFingerprint,
  LlamaCppInstallationId,
  MicroBatchSize,
  OutputLimit,
  makeLlamaCli,
  makeLlamaExecutionProfile,
  makeLlamaFitAssessmentKey,
  NormalizedLlamaModelPath,
  parseLlamaFitPlacement,
  renderExecutionProfileArguments,
  renderExecutionProfilePreset,
  type LlamaFitAssessmentInput,
} from "."

const MEBIBYTE = 1024 * 1024
const emptyMetadata: ModelFileRecord["metadata"] = {
  name: Option.none(), architecture: Option.none(), ggufFileType: Option.none(), quantization: Option.none(),
  trainedContextTokens: Option.none(), parameterCount: Option.none(), embeddingLength: Option.none(), blockCount: Option.none(),
  attentionHeadCount: Option.none(), vocabularySize: Option.none(), feedForwardLength: Option.none(), expertCount: Option.none(),
  expertUsedCount: Option.none(), tokenizerModel: Option.none(), tokenizerPre: Option.none(), chatTemplate: Option.none(),
  baseModelNames: [], baseModelRepositories: [], inputModalities: Option.none(), outputModalities: Option.none(),
}

const profileInput = {
  contextSize: ContextSize.Tokens({ value: 8_192 }),
  outputLimit: OutputLimit.Tokens({ value: 1_024 }),
  parallelSlots: 2,
  gpuLayers: GpuLayerSelection.Exact({ layers: 42 }),
  splitMode: "layer" as const,
  tensorSplit: Option.some([3, 2] as const),
  kvCache: { key: "q8_0" as const, value: "q8_0" as const },
  flashAttention: FlashAttentionSelection.Enabled(),
  batchSize: BatchSize.Exact({ value: 512 }),
  microBatchSize: MicroBatchSize.Exact({ value: 128 }),
  mmap: true,
  mlock: false,
}

describe("llama.cpp command routing", () => {
  it("parses the documented fit-print grammar into byte counts", () => {
    const placement = parseLlamaFitPlacement("CUDA0 8192 1024 512\nHost 0 0 256")
    expect(Option.isSome(placement)).toBe(true)
    if (Option.isNone(placement)) return
    expect(placement.value).toEqual([
      { device: "CUDA0", modelBytes: 8_589_934_592, contextBytes: 1_073_741_824, computeBytes: 536_870_912 },
      { device: "Host", modelBytes: 0, contextBytes: 0, computeBytes: 268_435_456 },
    ])
  })

  it("rejects incomplete and duplicate fit output", () => {
    expect(Option.isNone(parseLlamaFitPlacement("CUDA0 10 20 30"))).toBe(true)
    expect(Option.isNone(parseLlamaFitPlacement("Host 0 0 1\nHost 0 0 1"))).toBe(true)
  })

  it("keys fit assessments from every invalidating input", async () => {
    const profile = await Effect.runPromise(makeLlamaExecutionProfile(profileInput))
    const base = {
      modelPath: NormalizedLlamaModelPath.make("/models/model.gguf"),
      fileVersion: [{ key: SourceFileKey.make("primary"), sizeBytes: 100, modifiedAtMillis: Option.none<number>() }],
      projectorPath: Option.none<string>(),
      profileId: profile.id,
      fitExecutableFingerprint: LlamaCppExecutableFingerprint.make("fit"),
      hardwareFingerprint: "hardware",
    }
    const key = makeLlamaFitAssessmentKey(base)

    expect(makeLlamaFitAssessmentKey(base)).toBe(key)
    expect(makeLlamaFitAssessmentKey({ ...base, hardwareFingerprint: "changed" })).not.toBe(key)
    expect(makeLlamaFitAssessmentKey({ ...base, projectorPath: Option.some("/models/mmproj.gguf") })).not.toBe(key)
    expect(makeLlamaFitAssessmentKey({ ...base, fitExecutableFingerprint: LlamaCppExecutableFingerprint.make("fit-next") })).not.toBe(key)
  })

  it("renders one validated profile consistently for commands and presets", async () => {
    const profile = await Effect.runPromise(makeLlamaExecutionProfile(profileInput))
    expect(renderExecutionProfileArguments(profile)).toEqual([
      "--parallel", "2", "--kv-unified", "--cont-batching", "--split-mode", "layer", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
      "--ctx-size", "8192", "--n-gpu-layers", "42", "--fit", "off", "--tensor-split", "3,2",
      "--flash-attn", "on", "--batch-size", "512", "--ubatch-size", "128", "--mmap",
    ])
    expect(renderExecutionProfilePreset(profile)).toContain("n-predict = 1024")
    expect(renderExecutionProfilePreset(profile)).toContain("n-gpu-layers = 42")
  })

  it("rejects invalid profile values", async () => {
    const result = await Effect.runPromiseExit(makeLlamaExecutionProfile({
      ...profileInput,
      batchSize: BatchSize.Exact({ value: 0 }),
    }))
    expect(result._tag).toBe("Failure")
  })

  it("uses llama-server for devices and llama-fit-params for fit assessment", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const root = yield* fs.makeTempDirectoryScoped()
      const server = path.join(root, "llama-server")
      const fitParams = path.join(root, "llama-fit-params")
      const serverLog = path.join(root, "server.log")
      const fitLog = path.join(root, "fit.log")
      yield* fs.writeFileString(server, `#!/bin/sh\nprintf '%s\\n' "$*" > '${serverLog}'\nprintf '[{"id":"MTL0","name":"Apple GPU","backend":"Metal","type":"IGPU","total_memory":51539607552,"free_memory":42949672960}]\\n'\n`)
      yield* fs.writeFileString(fitParams, `#!/bin/sh\nprintf '%s\\n' "$*" > '${fitLog}'\nprintf 'MTL0 100 200 300\\nHost 400 500 600\\n'\n`)
      yield* fs.chmod(server, 0o755)
      yield* fs.chmod(fitParams, 0o755)
      const cli = yield* makeLlamaCli({
        id: LlamaCppInstallationId.make("installation"),
        build: LlamaBuildNumber.make(10011),
        commit: Option.none(),
        executables: {
          server: { path: server, fingerprint: LlamaCppExecutableFingerprint.make("server") },
          fitParams: { path: fitParams, fingerprint: LlamaCppExecutableFingerprint.make("fit") },
        },
        ownership: "user",
        discoveries: [{ _tag: "Configured", requestedPath: server }],
      })
      const profile = yield* makeLlamaExecutionProfile({
        contextSize: ContextSize.Tokens({ value: 8192 }),
        outputLimit: OutputLimit.RuntimeDefault(),
        parallelSlots: 1,
        gpuLayers: GpuLayerSelection.Fit(),
        splitMode: "layer",
        tensorSplit: Option.none(),
        kvCache: { key: "f16", value: "f16" },
        flashAttention: FlashAttentionSelection.RuntimeDefault(),
        batchSize: BatchSize.RuntimeDefault(),
        microBatchSize: MicroBatchSize.RuntimeDefault(),
        mmap: true,
        mlock: false,
      })
      const devices = yield* cli.listDevices
      const modelFileId = ModelFileId.make("model")
      const sourceId = ModelFileSourceId.make("source")
      const projectorSizeBytes = 100 * MEBIBYTE
      const record: ModelFileRecord = {
        id: modelFileId,
        sourceId,
        displayName: "model",
        format: ModelFileFormatId.make("gguf"),
        sizeBytes: 200 * MEBIBYTE,
        files: [
          { id: ModelFilePartId.make("primary"), role: "primary", sizeBytes: 100 * MEBIBYTE, sha256: Option.none() },
          { id: ModelFilePartId.make("projector"), role: "projector", sizeBytes: projectorSizeBytes, sha256: Option.none() },
        ],
        metadata: emptyMetadata,
        ownership: "external",
        operations: { delete: false },
        warnings: [],
      }
      const files: LlamaFitAssessmentInput["files"] = {
        record,
        primaryPath: "/models/model.gguf",
        shardPaths: [],
        projectorPath: Option.some("/models/mmproj.gguf"),
        auxiliaryPaths: [],
        version: [
          { key: SourceFileKey.make("primary"), sizeBytes: 100 * MEBIBYTE, modifiedAtMillis: Option.none() },
          { key: SourceFileKey.make("projector"), sizeBytes: projectorSizeBytes, modifiedAtMillis: Option.none() },
        ],
      }
      const fit = yield* cli.assessFit({
        files,
        profile,
      })
      return {
        devices,
        fit,
        serverArguments: yield* fs.readFileString(serverLog),
        fitArguments: yield* fs.readFileString(fitLog),
      }
    }).pipe(Effect.provide(BunContext.layer))))

    expect(result.fit._tag).toBe("Estimated")
    expect(result.devices.devices[0]).toMatchObject({
      id: "MTL0",
      name: Option.some("Apple GPU"),
      backend: Option.some("Metal"),
      type: Option.some("IGPU"),
      physicalId: Option.none(),
      totalMemoryBytes: Option.some(51_539_607_552),
      freeMemoryBytes: Option.some(42_949_672_960),
    })
    if (result.fit._tag !== "Estimated") return
    expect(result.fit.plan.memory).toEqual({
      baseBytes: 2_100 * MEBIBYTE,
      vision: Option.some({
        projectorFileBytes: 100 * MEBIBYTE,
        estimatedProjectorBytes: 120 * MEBIBYTE,
        uncertaintyBytes: 1_536 * MEBIBYTE,
      }),
      estimatedTotalBytes: 3_756 * MEBIBYTE,
    })
    expect(result.serverArguments.trim()).toBe("--list-devices")
    expect(result.fitArguments).toContain("--fit-print on")
    expect(result.fitArguments).toContain("--model /models/model.gguf")
    expect(result.fitArguments).not.toContain("--mmproj")
    expect(result.fitArguments).not.toContain("--kv-unified")
    expect(result.fitArguments).not.toContain("--cont-batching")
  })
})
