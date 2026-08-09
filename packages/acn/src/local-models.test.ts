import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ModelFileIdSchema,
  ModelOfferingTargetIdSchema,
  ModelPackageIdSchema,
  type ModelOfferingTarget,
  type ModelPackageSource,
} from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import {
  availabilityFromProviderProjection,
  resolveTargetPresentation,
} from "./local-models"

const target = (
  source: ModelPackageSource,
  path = "model.gguf",
): ModelOfferingTarget => ({
  _tag: "Package",
  package: {
    id: ModelPackageIdSchema.make("package-test"),
    source,
    files: [{
      id: ModelFileIdSchema.make("file-test"),
      path,
      role: "weights",
      sizeBytes: 1,
      tensorStorageBytes: Option.none(),
      sha256: "a".repeat(64),
    }],
    relationships: [],
    properties: {
      format: "gguf",
      quantization: "Q4_K - Medium",
      quantizationName: "4-bit",
      architecture: "test",
      maximumContextLength: 50_000,
    },
  },
})

describe("local model availability", () => {
  const providerModelIds = [ProviderModelIdSchema.make("test-configuration")]

  it("withholds provider availability until it matches the package snapshot", () => {
    expect(availabilityFromProviderProjection(
      providerModelIds[0],
      new Map([[providerModelIds[0]!, {
        availability: { _tag: "Disabled", reason: "incompatible_runtime" },
      }]]),
      false,
      Option.none(),
    )).toBeUndefined()
  })

  it("keeps an assessed installed configuration available before it has an offering", () => {
    expect(availabilityFromProviderProjection(
      undefined,
      new Map(),
      false,
      Option.none(),
    )).toEqual({ _tag: "Available" })
  })

  it("exposes an authoritative current provider incompatibility", () => {
    expect(availabilityFromProviderProjection(
      providerModelIds[0],
      new Map([[providerModelIds[0]!, {
        availability: { _tag: "Disabled", reason: "incompatible_runtime" },
      }]]),
      true,
      Option.none(),
    )).toEqual({
      _tag: "Unavailable",
      failure: {
        code: "incompatible_runtime",
        message: "This model configuration is not available to the local runtime",
        retryable: true,
      },
    })
  })

  it("uses only the provider offering for the exact configuration", () => {
    const otherProviderModelId = ProviderModelIdSchema.make("other-configuration")
    expect(availabilityFromProviderProjection(
      providerModelIds[0],
      new Map([
        [providerModelIds[0]!, {
          availability: { _tag: "Disabled", reason: "insufficient_resources" },
        }],
        [otherProviderModelId, { availability: { _tag: "Available" } }],
      ]),
      true,
      Option.none(),
    )).toMatchObject({
      _tag: "Unavailable",
      failure: { code: "insufficient_resources" },
    })
  })
})

describe("local model presentation", () => {
  const curatedTargetId = ModelOfferingTargetIdSchema.make("target-curated")
  const otherTargetId = ModelOfferingTargetIdSchema.make("target-other-quant")
  const huggingFaceTarget = target({
    _tag: "HuggingFace",
    repository: "LiquidAI/LFM2.5-2.6B-GGUF",
    revision: "a".repeat(40),
  })
  const curated = new Map([[curatedTargetId, {
    displayName: "Liquid LFM2.5 2.6B",
    description: "Curated model description.",
  }]])

  it("preserves curated metadata for an exact installed target", () => {
    expect(resolveTargetPresentation(
      curatedTargetId,
      huggingFaceTarget,
      curated,
    )).toEqual({
      displayName: "Liquid LFM2.5 2.6B",
      description: "Curated model description.",
    })
  })

  it("does not transfer curated metadata to another target", () => {
    expect(resolveTargetPresentation(
      otherTargetId,
      huggingFaceTarget,
      curated,
    )).toEqual({
      displayName: "LFM2.5-2.6B-GGUF",
      description: "",
    })
  })

  it("derives an uncurated local target name from its file", () => {
    expect(resolveTargetPresentation(
      otherTargetId,
      target({ _tag: "Local", path: "/models" }, "local-model.gguf"),
      new Map(),
    )).toEqual({
      displayName: "local-model.gguf",
      description: "",
    })
  })
})
