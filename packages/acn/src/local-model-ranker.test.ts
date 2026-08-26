import { LocalModelMutationFailed } from "@magnitudedev/acn-protocol"
import { Option } from "effect"
import { describe, expect, it } from "vitest"

import {
  exactBundleTensorStorageBytes,
  localModelRankingFailure,
} from "./local-model-ranker"

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
