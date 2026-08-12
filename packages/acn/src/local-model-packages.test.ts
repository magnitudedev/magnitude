import {
  DownloadAttemptIdSchema,
  LocalModelMutationFailed,
  ModelFileIdSchema,
  ModelPackageSchema,
  ModelPackageIdSchema,
} from "@magnitudedev/acn-protocol"
import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"

import {
  localModelPackageMutationFailure,
  packageAcquisition,
  projectPackageAcquisition,
} from "./local-model-packages"

describe("localModelPackageMutationFailure", () => {
  it("preserves a typed mutation failure across an outer error boundary", () => {
    const failure = new LocalModelMutationFailed({
      code: "model_download_target_identity_mismatch",
      message: "ICN admitted a different model bundle than requested.",
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

describe("projectPackageAcquisition", () => {
  const failure = new LocalModelMutationFailed({
    code: "network",
    message: "network unavailable",
    retryable: true,
  })
  const failed = {
    _tag: "Failed" as const,
    attemptId: DownloadAttemptIdSchema.make("download-test"),
    completedBytes: 4,
    totalBytes: 10,
    failure,
  }

  it("preserves generated installation provenance", () => {
    expect(projectPackageAcquisition({
      _tag: "Installed",
      path: "/hf/model.gguf",
      origin: "HuggingFaceCache",
    })).toEqual({
      _tag: "Installed",
      path: "/hf/model.gguf",
      origin: "HuggingFaceCache",
    })
  })

  it("shows an unacknowledged ICN failure", () => {
    expect(projectPackageAcquisition({ ...failed, acknowledged: false })).toEqual({
      _tag: "DownloadFailed",
      attemptId: "download-test",
      completedBytes: 4,
      totalBytes: 10,
      failure: {
        code: "network",
        message: "network unavailable",
        retryable: true,
      },
    })
  })

  it("hides an acknowledged ICN failure without ACN-owned state", () => {
    expect(projectPackageAcquisition({ ...failed, acknowledged: true })).toEqual({
      _tag: "NotInstalled",
    })
  })

  it("keeps a completed attempt at publishing until inventory contains the package", () => {
    const modelPackage = Schema.decodeUnknownSync(ModelPackageSchema)({
      id: ModelPackageIdSchema.make("package-test"),
      source: { _tag: "Local", path: "/model.gguf" },
      files: [{
        id: ModelFileIdSchema.make("weights"),
        path: "model.gguf",
        role: "weights",
        sizeBytes: 10,
        sha256: "0".repeat(64),
      }],
      relationships: [],
      properties: {
        format: "gguf",
        quantization: "Q4_K_M",
        quantizationName: "4-bit",
        architecture: "test",
        maximumContextLength: 1,
      },
    })
    const acquisition = packageAcquisition(modelPackage, new Map(), [{
      _tag: "Completed",
      id: DownloadAttemptIdSchema.make("download-test"),
      packageId: modelPackage.id,
    }])

    expect(projectPackageAcquisition(acquisition)).toEqual({
      _tag: "Downloading",
      attemptId: "download-test",
      stage: "publishing",
      completedBytes: 10,
      totalBytes: 10,
      bytesPerSecond: Option.none(),
    })
  })
})
