import { Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  CatalogModelIdSchema,
  CatalogVariantIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  ModelReleaseDateSchema,
  ModelVariantLabelSchema,
  type LocalProviderOffering,
  type ModelPackageEntry,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/sdk"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelConfigurationResolver, localModelTargetIdentity } from "./local-model-configuration-resolver"
import {
  LocalProviderOfferings,
  LocalProviderOfferingsLive,
  providerOfferingPackageEvidence,
  sameProviderOfferingPackageEvidence,
} from "./local-provider-offerings"

describe("local provider offering projection", () => {
  it("tracks package facts without changing exact configuration identity", () => {
    const packageId = ModelPackageIdSchema.make("package-a")
    const offering = {
      providerModelId: ProviderModelIdSchema.make("configuration-a"),
      configuration: {
        id: ModelServingConfigurationIdSchema.make("configuration-a"),
        bundle: { _tag: "Standalone", package: { id: packageId } },
        profile: { contextLength: 32_768 },
      },
      capabilities: {},
    } as unknown as LocalProviderOffering
    const entry = {
      package: { id: packageId },
      localState: { _tag: "Installed", path: "/models/a.gguf", origin: "Magnitude" },
      inspection: { _tag: "Pending" },
    } as unknown as ModelPackageEntry
    const evidence = providerOfferingPackageEvidence(
      [offering],
      new Map([[packageId, entry]]),
    )
    expect(evidence[0]).toMatchObject({
      providerModelId: "configuration-a",
      configurationId: "configuration-a",
      packages: [{ packageId: "package-a", installed: true, inspection: "Pending" }],
    })
    expect(sameProviderOfferingPackageEvidence(evidence, evidence)).toBe(true)
  })

  it("uses the capabilities selected by the configuration resolver", async () => {
    const packageId = ModelPackageIdSchema.make("package-a")
    const configurationId = ModelServingConfigurationIdSchema.make("configuration-a")
    const modelPackage = {
      id: packageId,
      source: { _tag: "Local" as const, path: "/models/a.gguf" },
      files: [],
      relationships: [],
      properties: {
        format: "gguf",
        quantization: "Q4_K_M",
        quantizationName: "4-bit",
        architecture: "test",
        maximumContextLength: Option.some(32_768),
        intrinsicModelId: Option.none(),
        intrinsicQualityId: Option.none(),
      },
    }
    const configuration = {
      id: configurationId,
      bundle: { _tag: "Standalone" as const, package: modelPackage },
      profile: { contextLength: 32_768 },
    }
    const low = ReasoningEffortSchema.make("low")
    const medium = ReasoningEffortSchema.make("medium")
    const none = ReasoningEffortSchema.make("none")
    const xhigh = ReasoningEffortSchema.make("xhigh")
    const catalogModel: RecommendableModel = {
      modelId: CatalogModelIdSchema.make("catalog-a"),
      variantId: CatalogVariantIdSchema.make("gguf:q4"),
      configuration,
      displayName: "Catalog model",
      variantLabel: ModelVariantLabelSchema.make("Q4"),
      description: "test",
      releaseDate: ModelReleaseDateSchema.make("2026-01-01"),
      license: "test",
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoning: {
          supported: true,
          efforts: [low, medium],
          defaultEffort: Option.some(medium),
        },
      },
      parameterization: { architecture: "dense", totalParameters: 8_000_000_000 },
      qualityScore: 1,
      qualityScoreProvenance: "test",
      fidelityRank: 1,
      quantizationAware: false,
      qualityEvidence: [],
    }
    const inspectedCapabilities = {
      vision: false,
      tools: true,
      structuredOutput: true,
      reasoning: {
        supported: true as const,
        efforts: [none, low, medium, xhigh],
        defaultEffort: Option.some(xhigh),
      },
    }
    const packageEntry: ModelPackageEntry = {
      package: modelPackage,
      localState: { _tag: "Installed", path: "/models/a.gguf", origin: "Magnitude" },
      inspection: { _tag: "Inspected", capabilities: inspectedCapabilities },
      catalogAttribution: {
        _tag: "Attributed",
        modelId: catalogModel.modelId,
        variantId: catalogModel.variantId,
      },
    }
    const dependencies = Layer.mergeAll(
      Layer.succeed(LocalModelConfigurationResolver, LocalModelConfigurationResolver.of({
        get: Effect.succeed(new Map([[
          localModelTargetIdentity(configuration.bundle),
          {
            servingConfiguration: configuration,
            assessment: { _tag: "Assessing" },
            catalogModel: Option.some(catalogModel),
            targetInspection: { _tag: "Inspected", capabilities: inspectedCapabilities },
          },
        ]])),
        changes: Stream.never,
        settled: Effect.succeed(true),
        resolve: () => Effect.succeed(Option.some({
          servingConfiguration: configuration,
          assessment: { _tag: "Assessing" },
          catalogModel: Option.some(catalogModel),
          targetInspection: { _tag: "Inspected", capabilities: inspectedCapabilities },
        })),
      })),
      Layer.succeed(LocalModelPackages, LocalModelPackages.of({
        initialized: Effect.succeed(true),
        snapshot: Effect.succeed({
          revision: 1,
          state: { inventory: { _tag: "Ready" }, entries: [packageEntry], downloads: [] },
        }),
        changes: Stream.never,
        installedPackageIds: Effect.succeed(new Set([packageId])),
        admitBundle: () => Effect.dieMessage("unused"),
        cancelDownload: () => Effect.dieMessage("unused"),
        acknowledgeFailure: () => Effect.dieMessage("unused"),
        removeBundlePackages: () => Effect.dieMessage("unused"),
      })),
    )

    const offerings = await Effect.runPromise(Effect.gen(function* () {
      const service = yield* LocalProviderOfferings
      return yield* service.list
    }).pipe(
      Effect.provide(LocalProviderOfferingsLive.pipe(Layer.provide(dependencies))),
      Effect.scoped,
    ))

    expect(offerings).toHaveLength(1)
    expect(offerings[0]?.capabilities).toEqual(inspectedCapabilities)
  })

  it("does not delay layer readiness while the initial assessment is stalled", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const packageId = ModelPackageIdSchema.make("package-a")
      const configurationId = ModelServingConfigurationIdSchema.make("configuration-a")
      const modelPackage = {
        id: packageId,
        source: { _tag: "Local" as const, path: "/models/a.gguf" },
        files: [],
        relationships: [],
        properties: {
          format: "gguf",
          quantization: "Q4_K_M",
          quantizationName: "4-bit",
          architecture: "test",
          maximumContextLength: Option.some(32_768),
          intrinsicModelId: Option.none(),
          intrinsicQualityId: Option.none(),
        },
      }
      const configuration = {
        id: configurationId,
        bundle: { _tag: "Standalone" as const, package: modelPackage },
        profile: { contextLength: 32_768 },
      }
      const capabilities = {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoning: { supported: false as const, efforts: [], defaultEffort: Option.none() },
      }
      const packageEntry: ModelPackageEntry = {
        package: modelPackage,
        localState: { _tag: "Installed", path: "/models/a.gguf", origin: "Magnitude" },
        inspection: {
          _tag: "Inspected",
          capabilities,
        },
        catalogAttribution: { _tag: "NotCatalogTarget" },
      }
      const dependencies = Layer.mergeAll(
        Layer.succeed(LocalModelConfigurationResolver, LocalModelConfigurationResolver.of({
          get: Effect.succeed(new Map([[
            localModelTargetIdentity(configuration.bundle),
            {
              servingConfiguration: configuration,
              assessment: { _tag: "Assessing" },
              catalogModel: Option.none(),
              targetInspection: { _tag: "Inspected", capabilities },
            },
          ]])),
          changes: Stream.never,
          settled: Effect.succeed(true),
          resolve: () => Effect.succeed(Option.some({
            servingConfiguration: configuration,
            assessment: { _tag: "Assessing" },
            catalogModel: Option.none(),
            targetInspection: { _tag: "Inspected", capabilities },
          })),
        })),
        Layer.succeed(LocalModelPackages, LocalModelPackages.of({
          initialized: Effect.succeed(true),
          snapshot: Effect.succeed({
            revision: 1,
            state: { inventory: { _tag: "Ready" }, entries: [packageEntry], downloads: [] },
          }),
          changes: Stream.never,
          installedPackageIds: Effect.succeed(new Set([packageId])),
          admitBundle: () => Effect.dieMessage("unused"),
          cancelDownload: () => Effect.dieMessage("unused"),
          acknowledgeFailure: () => Effect.dieMessage("unused"),
          removeBundlePackages: () => Effect.dieMessage("unused"),
        })),
      )
      yield* Layer.build(LocalProviderOfferingsLive.pipe(Layer.provide(dependencies)))
      return true
    }).pipe(Effect.timeout("1 second"))))

    expect(result).toBe(true)
  })
})
