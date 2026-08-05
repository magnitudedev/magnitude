import { LocalModelMutationFailed } from "@magnitudedev/acn-protocol"
import { describe, expect, it } from "vitest"

import { localModelPackageMutationFailure } from "./local-model-packages"

describe("localModelPackageMutationFailure", () => {
  it("preserves a typed mutation failure across an outer error boundary", () => {
    const failure = new LocalModelMutationFailed({
      code: "model_download_target_identity_mismatch",
      message: "ICN admitted a different model target than requested.",
      retryable: false,
    })

    expect(localModelPackageMutationFailure("start_model_download_failed", failure)).toBe(failure)
  })

  it("normalizes an untyped infrastructure failure", () => {
    expect(localModelPackageMutationFailure(
      "start_model_download_failed",
      new Error("connection closed"),
    )).toMatchObject({
      code: "start_model_download_failed",
      message: "connection closed",
      retryable: true,
    })
  })
})
