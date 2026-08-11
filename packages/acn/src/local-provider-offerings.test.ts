import { Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  type LocalProviderOffering,
  type ModelPackageEntry,
} from "@magnitudedev/acn-protocol"
import {
  IcnCatalog,
  IcnHardware,
  IcnInstalledModels,
} from "@magnitudedev/icn"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { LocalModelAssessor } from "./local-model-assessor"
import { LocalModelPackages } from "./local-model-packages"
import {
  LocalProviderOfferingsLive,
  providerOfferingPackageEvidence,
  sameProviderOfferingPackageEvidence,
} from "./local-provider-offerings"
import { RetainedModelConfigurations } from "./retained-model-configurations"

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
      }
      const dependencies = Layer.mergeAll(
        Layer.succeed(RetainedModelConfigurations, RetainedModelConfigurations.of({
          get: Effect.succeed([configuration]),
          recoveryCompleted: Effect.succeed(true),
          changes: Stream.never,
          resolve: () => Effect.succeed(Option.some(configuration)),
          materialize: Effect.succeed,
          remove: () => Effect.succeed(Option.none()),
          completeRecovery: () => Effect.succeed([]),
        })),
        Layer.succeed(IcnCatalog, IcnCatalog.of({
          get: Effect.succeed({ revision: 1, state: { models: [], diagnostics: [] } }),
          changes: Stream.never,
          ready: Effect.succeed(true),
          refresh: Effect.void,
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
            state: { inventory: { _tag: "Ready" }, entries: [packageEntry] },
          }),
          changes: Stream.never,
          installedPackageIds: Effect.succeed(new Set([packageId])),
          admitBundle: () => Effect.dieMessage("unused"),
          cancelAttempts: () => Effect.dieMessage("unused"),
          acknowledgeFailures: () => Effect.dieMessage("unused"),
          removeBundlePackages: () => Effect.dieMessage("unused"),
        })),
        Layer.succeed(LocalModelAssessor, LocalModelAssessor.of({
          state: Effect.never,
          changes: Stream.never,
        })),
      )
      yield* Layer.build(LocalProviderOfferingsLive.pipe(Layer.provide(dependencies)))
      return true
    }).pipe(Effect.timeout("1 second"))))

    expect(result).toBe(true)
  })
})
