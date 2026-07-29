import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  DownloadAttemptIdSchema,
  ModelPackageIdSchema,
  type DownloadAttempt,
} from "@magnitudedev/protocol"
import { exactAttemptDecision } from "./local-model-packages"

const attemptId = DownloadAttemptIdSchema.make("download_test")
const packageId = ModelPackageIdSchema.make("package_test")

const attempt = (state: DownloadAttempt["_tag"]): DownloadAttempt => {
  switch (state) {
    case "Pending":
      return { _tag: state, id: attemptId, packageId }
    case "Downloading":
      return {
        _tag: state,
        id: attemptId,
        packageId,
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.some(1),
      }
    case "Completed":
      return {
        _tag: state,
        id: attemptId,
        packageId,
      }
    case "Failed":
      return {
        _tag: state,
        id: attemptId,
        packageId,
        completedBytes: 1,
        totalBytes: 2,
        failure: {
          code: "download_failed",
          message: "Download failed",
          retryable: true,
        },
      }
    case "Cancelled":
      return { _tag: state, id: attemptId, packageId }
  }
}

describe("exact download-attempt completion", () => {
  it.each(["Pending", "Downloading"] as const)("waits for %s", (state) => {
    expect(exactAttemptDecision(attempt(state))).toEqual({ _tag: "Wait" })
  })

  it("completes from the exact terminal attempt", () => {
    expect(exactAttemptDecision(attempt("Completed"))).toEqual({ _tag: "Complete" })
  })

  it("preserves an exact terminal failure", () => {
    const decision = exactAttemptDecision(attempt("Failed"))
    expect(decision._tag === "Fail" ? decision.failure : null).toMatchObject({
      code: "download_failed",
      message: "Download failed",
      retryable: true,
    })
  })

  it("maps exact cancellation to the target command failure", () => {
    const decision = exactAttemptDecision(attempt("Cancelled"))
    expect(decision._tag === "Fail" ? decision.failure : null).toMatchObject({
      code: "local_model_download_cancelled",
      retryable: true,
    })
  })
})
