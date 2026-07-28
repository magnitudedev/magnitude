import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  DownloadAttemptIdSchema,
  ModelFileIdSchema,
  ModelPackageIdSchema,
  type DownloadAttempt,
  type ModelOfferingTarget,
} from "@magnitudedev/protocol"
import { targetInstallationOutcome } from "./local-model-packages"

const packageId = ModelPackageIdSchema.make("package_test")
const target: ModelOfferingTarget = {
  _tag: "Package",
  package: {
    id: packageId,
    source: {
      _tag: "HuggingFace",
      repository: "owner/repository",
      revision: "commit",
    },
    files: [{
      id: ModelFileIdSchema.make("file_test"),
      path: "model.gguf",
      role: "weights",
      sizeBytes: 1,
      sha256: "a".repeat(64),
    }],
    relationships: [],
    properties: {
      format: "gguf",
      quantization: "Q4",
      quantizationName: "4-bit",
      architecture: "test",
      maximumContextLength: 1,
    },
  },
}

const attempt = (
  state: DownloadAttempt["_tag"],
): DownloadAttempt => {
  const id = DownloadAttemptIdSchema.make(`download_${state}`)
  switch (state) {
    case "Pending":
      return { _tag: state, id, packageId }
    case "Downloading":
      return {
        _tag: state,
        id,
        packageId,
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.some(1),
      }
    case "Completed":
      return { _tag: state, id, packageId }
    case "Cancelled":
      return { _tag: state, id, packageId }
    case "Failed":
      return {
        _tag: state,
        id,
        packageId,
        completedBytes: 1,
        totalBytes: 2,
        failure: {
          code: "download_failed",
          message: "Download failed",
          retryable: true,
        },
      }
  }
}

describe("target installation outcome", () => {
  it("completes from authoritative installed inventory", () => {
    expect(targetInstallationOutcome(
      target,
      new Map([[packageId, "/models/test"]]),
      [],
    )).toEqual(Option.some({ _tag: "Installed" }))
  })

  it.each(["Pending", "Downloading"] as const)(
    "waits while the latest attempt is %s and installation has not converged",
    (state) => {
      expect(Option.isNone(targetInstallationOutcome(
        target,
        new Map(),
        [attempt(state)],
      ))).toBe(true)
    },
  )

  it("does not treat historical completion as current installation", () => {
    const outcome = targetInstallationOutcome(
      target,
      new Map(),
      [attempt("Completed")],
    )
    expect(Option.isSome(outcome) && outcome.value._tag === "Failed"
      ? outcome.value.failure
      : null).toMatchObject({
        code: "local_model_download_not_active",
        retryable: true,
      })
  })

  it("preserves a terminal download failure", () => {
    const outcome = targetInstallationOutcome(
      target,
      new Map(),
      [attempt("Failed")],
    )
    expect(Option.isSome(outcome) && outcome.value._tag === "Failed"
      ? outcome.value.failure
      : null).toMatchObject({
        code: "download_failed",
        message: "Download failed",
        retryable: true,
    })
  })

  it("fails when the download is cancelled", () => {
    const outcome = targetInstallationOutcome(
      target,
      new Map(),
      [attempt("Cancelled")],
    )
    expect(Option.isSome(outcome) && outcome.value._tag === "Failed"
      ? outcome.value.failure
      : null).toMatchObject({
        code: "local_model_download_cancelled",
        retryable: true,
      })
  })

  it("fails when an uninstalled target has no admitted attempt", () => {
    const outcome = targetInstallationOutcome(target, new Map(), [])
    expect(Option.isSome(outcome) && outcome.value._tag === "Failed"
      ? outcome.value.failure
      : null).toMatchObject({
        code: "local_model_download_not_active",
        retryable: true,
      })
  })
})
