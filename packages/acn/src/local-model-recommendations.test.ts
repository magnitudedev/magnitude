import { LocalModelMutationFailed } from "@magnitudedev/acn-protocol"
import { Option } from "effect"
import { describe, expect, it } from "vitest"

import {
  exactTargetTensorStorageBytes,
  localModelRecommendationFailure,
} from "./local-model-recommendations"

describe("localModelRecommendationFailure", () => {
  it("preserves typed assessment failure metadata for the public lifecycle", () => {
    expect(localModelRecommendationFailure(new LocalModelMutationFailed({
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

describe("exactTargetTensorStorageBytes", () => {
  const model = (files: readonly unknown[]) => ({
    target: { _tag: "Package", package: { files } },
  }) as Parameters<typeof exactTargetTensorStorageBytes>[0]

  it("sums exact tensor storage and deduplicates immutable content", () => {
    expect(exactTargetTensorStorageBytes(model([
      { role: "weights", sha256: "a", tensorStorageBytes: Option.some(10) },
      { role: "weights", sha256: "a", tensorStorageBytes: Option.some(10) },
      { role: "weights", sha256: "b", tensorStorageBytes: Option.some(15) },
      { role: "projector", sha256: "c", tensorStorageBytes: Option.some(100) },
    ]))).toEqual(Option.some(25))
  })

  it("declines to reject when any required component is unknown", () => {
    expect(exactTargetTensorStorageBytes(model([
      { role: "weights", sha256: "a", tensorStorageBytes: Option.some(10) },
      { role: "weights", sha256: "b", tensorStorageBytes: Option.none() },
    ]))).toEqual(Option.none())
  })
})
