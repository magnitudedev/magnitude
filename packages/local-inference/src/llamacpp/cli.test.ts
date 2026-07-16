import { describe, expect, it } from "vitest"
import { Effect, Option } from "effect"
import {
  BatchSize,
  ContextSize,
  FlashAttentionSelection,
  GpuLayerSelection,
  MicroBatchSize,
  OutputLimit,
  makeLlamaExecutionProfile,
  parseLlamaFitPlacement,
  renderExecutionProfileArguments,
  renderExecutionProfilePreset,
} from "./index"

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

describe("llama.cpp CLI contracts", () => {
  it("parses the documented fit-print grammar into exact byte counts", () => {
    const output = [
      "build: 6895 (4341dc8bc)",
      "CUDA0 8192 1024 512",
      "Host 0 0 256",
    ].join("\n")
    const placement = parseLlamaFitPlacement(output)
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

  it("renders one validated profile consistently for commands and presets", async () => {
    const profile = await Effect.runPromise(makeLlamaExecutionProfile(profileInput))
    expect(renderExecutionProfileArguments(profile)).toEqual([
      "--parallel", "2", "--split-mode", "layer", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
      "--ctx-size", "8192", "--n-gpu-layers", "42", "--fit", "off", "--tensor-split", "3,2",
      "--flash-attn", "on", "--batch-size", "512", "--ubatch-size", "128", "--mmap",
    ])
    expect(renderExecutionProfilePreset(profile)).toContain("n-predict = 1024")
    expect(renderExecutionProfilePreset(profile)).toContain("n-gpu-layers = 42")
  })

  it("rejects invalid values at profile construction", async () => {
    const result = await Effect.runPromiseExit(makeLlamaExecutionProfile({
      ...profileInput,
      batchSize: BatchSize.Exact({ value: 0 }),
    }))
    expect(result._tag).toBe("Failure")
  })
})
