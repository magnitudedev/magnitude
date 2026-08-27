import { LocalModelMutationFailed } from "@magnitudedev/acn-protocol"
import { Option } from "effect"
import { describe, expect, it } from "vitest"

import {
  exactBundleTensorStorageBytes,
  localModelRankingCandidates,
  localModelRankingFailure,
} from "./local-model-ranker"

describe("localModelRankingCandidates", () => {
  const configuration = (id: string) => ({
    bundle: {
      _tag: "Standalone" as const,
      package: {
        id,
        source: { _tag: "Local" as const, path: `/models/${id}.gguf` },
        files: [],
        relationships: [],
        properties: {
          format: "gguf",
          quantization: "Q4_K_M",
          quantizationName: "4-bit",
          architecture: "test",
          maximumContextLength: Option.none(),
          intrinsicModelId: Option.none(),
          intrinsicQualityId: Option.none(),
        },
      },
    },
    profile: { contextLength: 1 },
  })

  it("includes a distinct effective installed configuration during an update", () => {
    const desired = configuration("desired")
    const effective = configuration("effective")
    const model = { configuration: desired }

    expect(localModelRankingCandidates([{
      model,
      effectiveConfiguration: Option.some(effective),
    } as never]).map(({ configuration }) => configuration)).toEqual([desired, effective])
  })

  it("does not duplicate an unchanged effective configuration", () => {
    const current = configuration("same")

    expect(localModelRankingCandidates([{
      model: { configuration: current },
      effectiveConfiguration: Option.some(current),
    } as never])).toHaveLength(1)
  })
})

describe("localModelRankingFailure", () => {
  it("preserves typed assessment failure metadata for the public lifecycle", () => {
    expect(localModelRankingFailure(new LocalModelMutationFailed({
      code: "planner_timeout",
      message: "Hardware assessment took longer than five minutes.",
      retryable: true,
    }))).toEqual({
      code: "planner_timeout",
      message: "Hardware assessment took longer than five minutes.",
      retryable: true,
    })
  })
})

describe("exactBundleTensorStorageBytes", () => {
  const model = (files: readonly unknown[]) => ({
    configuration: { bundle: { _tag: "Standalone", package: { files } } },
  }) as Parameters<typeof exactBundleTensorStorageBytes>[0]

  it("sums exact tensor storage and deduplicates immutable content", () => {
    expect(exactBundleTensorStorageBytes(model([
      { role: "weights", sha256: "a", tensorStorageBytes: Option.some(10) },
      { role: "weights", sha256: "a", tensorStorageBytes: Option.some(10) },
      { role: "weights", sha256: "b", tensorStorageBytes: Option.some(15) },
      { role: "projector", sha256: "c", tensorStorageBytes: Option.some(100) },
    ]))).toEqual(Option.some(25))
  })

  it("declines to reject when any required component is unknown", () => {
    expect(exactBundleTensorStorageBytes(model([
      { role: "weights", sha256: "a", tensorStorageBytes: Option.some(10) },
      { role: "weights", sha256: "b", tensorStorageBytes: Option.none() },
    ]))).toEqual(Option.none())
  })
})
