import { describe, expect, it } from "vitest"
import { Effect, Option, Schema } from "effect"
import {
  ModelDownload as NativeModelDownloadSchema,
  ModelPackage as NativeModelPackageSchema,
} from "@magnitudedev/icn-protocol/schemas"
import { modelDownloadFromIcn, modelPackageFromIcn } from "./local-model-icn-adapter"

describe("local model ICN adapter", () => {
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

  it("preserves structured model-download failure facts", async () => {
    const download = Schema.decodeUnknownSync(NativeModelDownloadSchema)({
      id: "download_disk_full",
      bundle: {
        _tag: "Standalone",
        package: {
          id: "package_test",
          source: { _tag: "Local", path: "/models/test.gguf" },
          files: [],
          relationships: [],
          properties: {
            format: "gguf",
            quantization: "Q4_K_M",
            quantizationName: "Q4_K_M",
            architecture: "test",
            maximumContextLength: 131_072,
          },
        },
      },
      state: {
        _tag: "Failed",
        completedBytes: 0,
        totalBytes: 35_000_000_000,
        failure: {
          _tag: "InsufficientDiskSpace",
          requiredBytes: 37_923_968_128,
          availableBytes: 33_440_665_600,
        },
        acknowledged: false,
      },
    })

    await expect(Effect.runPromise(modelDownloadFromIcn(download))).resolves.toMatchObject({
      id: "download_disk_full",
      state: {
        _tag: "Failed",
        failure: {
          _tag: "InsufficientDiskSpace",
          requiredBytes: 37_923_968_128,
          availableBytes: 33_440_665_600,
        },
      },
    })
  })
})
