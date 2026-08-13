import { describe, expect, it } from "vitest"
import { Effect, Option, Schema } from "effect"
import { ModelPackage as NativeModelPackageSchema } from "@magnitudedev/icn-protocol/schemas"
import { modelPackageFromIcn } from "./local-model-icn-adapter"

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
})
