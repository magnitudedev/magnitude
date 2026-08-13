import { describe, expect, it } from "vitest"
import { Effect, Option, Schema } from "effect"
import {
  DownloadAttempt as NativeDownloadAttemptSchema,
  ModelPackage as NativeModelPackageSchema,
} from "@magnitudedev/icn-protocol/schemas"
import {
  downloadAttemptFromIcn,
  modelPackageFromIcn,
} from "./local-model-icn-adapter"

describe("local model ICN adapter", () => {
  it("projects a decoded native download rate into the protocol representation", async () => {
    const attempt = Schema.decodeUnknownSync(NativeDownloadAttemptSchema)({
      _tag: "Downloading",
      id: "download_test",
      packageId: "package_test",
      stage: "downloading",
      completedBytes: 4_000,
      totalBytes: 10_000,
      bytesPerSecond: 2_000,
    })

    const projected = await Effect.runPromise(downloadAttemptFromIcn(attempt))

    expect(projected).toEqual({
      _tag: "Downloading",
      id: "download_test",
      packageId: "package_test",
      stage: "downloading",
      completedBytes: 4_000,
      totalBytes: 10_000,
      bytesPerSecond: Option.some(2_000),
    })
  })

  it("projects a missing native download rate as None", async () => {
    const attempt = Schema.decodeUnknownSync(NativeDownloadAttemptSchema)({
      _tag: "Downloading",
      id: "download_test",
      packageId: "package_test",
      stage: "downloading",
      completedBytes: 4_000,
      totalBytes: 10_000,
    })

    const projected = await Effect.runPromise(downloadAttemptFromIcn(attempt))

    expect(projected._tag).toBe("Downloading")
    if (projected._tag === "Downloading") {
      expect(projected.bytesPerSecond).toEqual(Option.none())
    }
  })

  it("preserves native failure acknowledgement", async () => {
    const attempt = Schema.decodeUnknownSync(NativeDownloadAttemptSchema)({
      _tag: "Failed",
      id: "download_test",
      packageId: "package_test",
      completedBytes: 4_000,
      totalBytes: 10_000,
      failure: {
        _tag: "NetworkUnavailable",
      },
      acknowledged: true,
    })

    await expect(Effect.runPromise(downloadAttemptFromIcn(attempt))).resolves.toMatchObject({
      _tag: "Failed",
      acknowledged: true,
    })
  })

  it("preserves insufficient-disk facts without parsing diagnostic text", async () => {
    const attempt = Schema.decodeUnknownSync(NativeDownloadAttemptSchema)({
      _tag: "Failed",
      id: "download_disk_full",
      packageId: "package_test",
      completedBytes: 0,
      totalBytes: 35_000_000_000,
      failure: {
        _tag: "InsufficientDiskSpace",
        requiredBytes: 37_923_968_128,
        availableBytes: 33_440_665_600,
      },
      acknowledged: false,
    })

    await expect(Effect.runPromise(downloadAttemptFromIcn(attempt))).resolves.toMatchObject({
      _tag: "Failed",
      failure: {
        _tag: "InsufficientDiskSpace",
        requiredBytes: 37_923_968_128,
        availableBytes: 33_440_665_600,
      },
    })
  })

  it("projects nullable wire tensor storage into the domain Option", async () => {
    const modelPackage = Schema.decodeUnknownSync(NativeModelPackageSchema)({
      id: "package_test",
      source: { _tag: "Local", path: "/models/test.gguf" },
      files: [{
        id: "file_test",
        path: "test.gguf",
        role: "weights",
        sizeBytes: 1_024,
        tensorStorageBytes: null,
        sha256: "a".repeat(64),
      }],
      relationships: [],
      properties: {
        format: "gguf",
        quantization: "Q4_K_M",
        quantizationName: "Q4_K_M",
        architecture: "test",
        maximumContextLength: 131_072,
      },
    })

    const projected = await Effect.runPromise(modelPackageFromIcn(modelPackage))

    expect(projected.files[0]?.tensorStorageBytes).toEqual(Option.none())
  })
})
