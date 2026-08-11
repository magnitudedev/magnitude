import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ModelFileIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  servableModelBundlePackageIds,
  type ModelServingConfiguration,
  type ServableModelBundle,
  type ModelPackageSource,
} from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import {
  availabilityFromProviderProjection,
  resolveBundlePresentation,
} from "./local-models"
import { resolveLocalModelConfigurations } from "./local-model-configuration-resolver"

const standaloneBundle = (
  source: ModelPackageSource,
  path = "model.gguf",
): ServableModelBundle => ({
  _tag: "Standalone",
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

const configuration = (
  id: string,
  bundle: ServableModelBundle,
  contextLength: number,
): ModelServingConfiguration => ({
  id: ModelServingConfigurationIdSchema.make(id),
  bundle,
  profile: { contextLength },
})

describe("local model configuration resolution", () => {
  it("resolves one configuration per bundle with retained, catalog, standard precedence", () => {
    const bundle = standaloneBundle({ _tag: "Local", path: "/models" })
    const standard = configuration("configuration-standard", bundle, 50_000)
    const catalog = configuration("configuration-catalog", bundle, 32_000)
    const retained = configuration("configuration-retained", bundle, 24_000)

    expect([...resolveLocalModelConfigurations({
      retained: [],
      catalog: [],
      installedPackageIds: new Set(servableModelBundlePackageIds(bundle)),
      assessed: new Map([[standard.id, {
        configuration: standard,
        origin: "Standard",
        assessment: { _tag: "Assessing" },
      }]]),
    }).values()].map(({ configuration }) => configuration)).toEqual([standard])
    expect([...resolveLocalModelConfigurations({
      retained: [],
      catalog: [catalog],
      installedPackageIds: new Set(servableModelBundlePackageIds(bundle)),
      assessed: new Map([
        [standard.id, {
          configuration: standard,
          origin: "Standard",
          assessment: { _tag: "Assessing" },
        }],
        [catalog.id, {
          configuration: catalog,
          origin: "Authored",
          assessment: { _tag: "Assessing" },
        }],
      ]),
    }).values()].map(({ configuration }) => configuration)).toEqual([catalog])
    expect([...resolveLocalModelConfigurations({
      retained: [retained],
      catalog: [catalog],
      installedPackageIds: new Set(servableModelBundlePackageIds(bundle)),
      assessed: new Map([
        [standard.id, {
          configuration: standard,
          origin: "Standard",
          assessment: { _tag: "Assessing" },
        }],
        [catalog.id, {
          configuration: catalog,
          origin: "Authored",
          assessment: { _tag: "Assessing" },
        }],
        [retained.id, {
          configuration: retained,
          origin: "Authored",
          assessment: { _tag: "Assessing" },
        }],
      ]),
    }).values()].map(({ configuration }) => configuration)).toEqual([retained])
  })

  it("does not reinterpret removed authored configurations as standard resolutions", () => {
    const bundle = standaloneBundle({ _tag: "Local", path: "/models" })
    const staleCatalog = configuration("configuration-catalog", bundle, 32_000)

    expect(resolveLocalModelConfigurations({
      retained: [],
      catalog: [],
      installedPackageIds: new Set(servableModelBundlePackageIds(bundle)),
      assessed: new Map([[staleCatalog.id, {
        configuration: staleCatalog,
        origin: "Authored",
        assessment: { _tag: "Assessing" },
      }]]),
    }).size).toBe(0)
  })

  it("drops a generated resolution when its installed package disappears", () => {
    const bundle = standaloneBundle({ _tag: "Local", path: "/models" })
    const standard = configuration("configuration-standard", bundle, 50_000)

    expect(resolveLocalModelConfigurations({
      retained: [],
      catalog: [],
      installedPackageIds: new Set(),
      assessed: new Map([[standard.id, {
        configuration: standard,
        origin: "Standard",
        assessment: { _tag: "Assessing" },
      }]]),
    }).size).toBe(0)
  })
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
  const huggingFaceBundle = standaloneBundle({
    _tag: "HuggingFace",
    repository: "LiquidAI/LFM2.5-2.6B-GGUF",
    revision: "a".repeat(40),
  })
  const curated = {
    displayName: "Liquid LFM2.5 2.6B",
    description: "Curated model description.",
  }

  it("preserves curated metadata for an exact installed bundle", () => {
    expect(resolveBundlePresentation(
      huggingFaceBundle,
      curated,
    )).toEqual({
      displayName: "Liquid LFM2.5 2.6B",
      description: "Curated model description.",
    })
  })

  it("does not transfer curated metadata to another bundle", () => {
    expect(resolveBundlePresentation(
      huggingFaceBundle,
      undefined,
    )).toEqual({
      displayName: "LFM2.5-2.6B-GGUF",
      description: "",
    })
  })

  it("derives an uncurated local bundle name from its file", () => {
    expect(resolveBundlePresentation(
      standaloneBundle({ _tag: "Local", path: "/models" }, "local-model.gguf"),
      undefined,
    )).toEqual({
      displayName: "local-model.gguf",
      description: "",
    })
  })
})
