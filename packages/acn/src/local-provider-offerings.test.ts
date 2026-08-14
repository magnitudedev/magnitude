import { Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  type LocalProviderOffering,
  type ModelPackageEntry,
} from "@magnitudedev/acn-protocol"
import {
  IcnHardware,
  IcnInstalledModels,
} from "@magnitudedev/icn"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelConfigurationResolver, localModelTargetIdentity } from "./local-model-configuration-resolver"
import {
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
          maximumContextLength: 32_768,
          intrinsicModelId: Option.none(),
          intrinsicQualityId: Option.none(),
        },
      }
      const configuration = {
        id: configurationId,
        bundle: { _tag: "Standalone" as const, package: modelPackage },
        profile: { contextLength: 32_768 },
      }
      const packageEntry: ModelPackageEntry = {
        package: modelPackage,
        localState: { _tag: "Installed", path: "/models/a.gguf", origin: "Magnitude" },
        inspection: {
          _tag: "Inspected",
          capabilities: {
            vision: false,
            tools: true,
            structuredOutput: true,
            reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
          },
        },
        catalogAttribution: { _tag: "NotCatalogTarget" },
      }
      const dependencies = Layer.mergeAll(
        Layer.succeed(LocalModelConfigurationResolver, LocalModelConfigurationResolver.of({
          get: Effect.succeed(new Map([[
            localModelTargetIdentity(configuration.bundle),
            {
              servingConfiguration: configuration,
              assessment: Option.some({ _tag: "Assessing" }),
              catalogModel: Option.none(),
            },
          ]])),
          changes: Stream.never,
          settled: Effect.succeed(true),
          resolve: () => Effect.succeed(Option.some({
            servingConfiguration: configuration,
            assessment: Option.some({ _tag: "Assessing" }),
            catalogModel: Option.none(),
          })),
        })),
        Layer.succeed(IcnInstalledModels, IcnInstalledModels.of({
          get: Effect.succeed({
            revision: 1,
            state: { revision: 1, reconciliationComplete: true, packages: [] },
          }),
          changes: Stream.never,
          initialized: Effect.succeed(true),
          refresh: Effect.void,
        })),
        Layer.succeed(IcnHardware, IcnHardware.of({
          get: Effect.never,
          changes: Stream.never,
          initialized: Effect.succeed(true),
          refresh: Effect.void,
          assessmentChanges: Stream.never,
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
