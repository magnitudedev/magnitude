import { Deferred, Effect, Layer, Option, Stream, SubscriptionRef } from "effect"
import { describe, expect, it } from "vitest"
import {
  AssessmentEnvironmentIdSchema,
  ModelFileIdSchema,
  ModelPackageIdSchema,
  ModelReleaseDateSchema,
  type ModelPackageEntry,
} from "@magnitudedev/acn-protocol"
import { IcnHardware, IcnModels } from "@magnitudedev/icn"
import { LocalModelAssessments } from "./local-model-assessments"
import { LocalModelAssessor, LocalModelAssessorLive } from "./local-model-assessor"
import { LocalModelCatalogAdapterLive } from "./local-model-catalog-adapter"
import { LocalModelPackages } from "./local-model-packages"

describe("LocalModelAssessor", () => {
  it("correlates native evidence to authored configuration without reassessing unchanged evidence", async () => {
    let assessmentCalls = 0
    const assessmentRequestCounts: number[] = []

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const firstAssessmentStarted = yield* Deferred.make<void>()
      const releaseFirstAssessment = yield* Deferred.make<void>()
      const packageId = ModelPackageIdSchema.make("package-test")
      const modelPackage = {
        id: packageId,
        source: { _tag: "Local" as const, path: "/models/test.gguf" },
        files: [{
          id: ModelFileIdSchema.make("file-test"),
          path: "test.gguf",
          role: "weights" as const,
          sizeBytes: 1,
          tensorStorageBytes: Option.none(),
          sha256: "a".repeat(64),
        }],
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
      const packageEntry: ModelPackageEntry = {
        package: modelPackage,
        localState: { _tag: "Installed", path: "/models/test.gguf", origin: "Magnitude" },
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
      let packageState = { inventory: { _tag: "Ready" as const }, entries: [packageEntry], downloads: [] }
      const packageStateRef = yield* SubscriptionRef.make(packageState)
      const configuration = {
        bundle: { _tag: "Standalone" as const, package: modelPackage },
        profile: { contextLength: 32_768 },
      }
      const configurationWithSameMaterialIdentity = {
        ...configuration,
        bundle: {
          _tag: "Standalone" as const,
          package: {
            ...modelPackage,
            properties: {
              ...modelPackage.properties,
              quantizationName: "alternate authored configuration",
            },
          },
        },
      }
      const catalogModel = (id: string, desiredConfiguration: typeof configuration) => ({
        id: `${id}:gguf:q4`,
        modelId: id,
        variantId: "gguf:q4",
        desiredConfiguration,
        localState: { _tag: "NotInstalled" as const },
        displayName: "Test",
        variantLabel: "Q4",
        description: "Test",
        releaseDate: ModelReleaseDateSchema.make("2026-01-01"),
        license: "test",
        capabilities: packageEntry.inspection._tag === "Inspected"
          ? packageEntry.inspection.capabilities
          : undefined,
        parameterization: { architecture: "dense" as const, totalParameters: 8_000_000_000 },
        intelligence: {
          score: 1,
          provenance: {
            kind: "artificialAnalysisIntelligenceIndex",
            methodologyVersion: "test",
            asOfDate: "2026-01-01",
            url: "https://example.com/model",
          },
        },
        fidelityRank: 1,
        quantizationAware: false,
      })
      const dependencies = Layer.mergeAll(
        Layer.succeed(IcnModels, IcnModels.of({
          get: Effect.succeed({
            revision: 1,
            state: {
              revision: 1,
              reconciliationComplete: true,
              models: [
                catalogModel("recommendable-test", configuration),
                catalogModel("same-material-test", configurationWithSameMaterialIdentity),
              ] as never,
              diagnostics: [],
            },
          }),
          changes: Stream.never,
          initialized: Effect.succeed(true),
          refresh: Effect.void,
        })),
        Layer.succeed(IcnHardware, IcnHardware.of({
          get: Effect.succeed({
            revision: 1,
            state: {
              native_build: "native-build-test",
              topology_fingerprint: "topology-test",
              system_memory: { physical_capacity_bytes: 64, assess_reserve_bytes: 8 },
              enabled_backends: ["cpu"],
            },
          } as never),
          changes: Stream.never,
          initialized: Effect.succeed(true),
          refresh: Effect.void,
          assessmentChanges: Stream.never,
        })),
        Layer.succeed(LocalModelPackages, LocalModelPackages.of({
          initialized: Effect.succeed(true),
          state: SubscriptionRef.get(packageStateRef),
          changes: packageStateRef.changes,
          installedPackageIds: Effect.succeed(new Set([packageId])),
          refresh: Effect.void,
        })),
        Layer.succeed(LocalModelAssessments, LocalModelAssessments.of({
          assess: (requests) => Effect.gen(function* () {
            assessmentCalls += 1
            assessmentRequestCounts.push(requests.length)
            if (assessmentCalls === 1) {
              yield* Deferred.succeed(firstAssessmentStarted, undefined)
              yield* Deferred.await(releaseFirstAssessment)
            }
            return requests.map((_, index) => index === 0
              ? {
                  _tag: "Assessed",
                  environmentId: AssessmentEnvironmentIdSchema.make("environment-test"),
                  assessments: [{
                    _tag: "Incompatible",
                    configuration,
                    failure: {
                      code: "unsupported_architecture",
                      message: "Unsupported architecture",
                      retryable: false,
                    },
                  }],
                }
              : {
                  _tag: "InvalidBundle",
                  message: "terminal test result",
                })
          }),
        })),
      )
      const testLayer = LocalModelAssessorLive.pipe(
        Layer.provide(LocalModelCatalogAdapterLive.pipe(Layer.provideMerge(dependencies))),
        Layer.provide(dependencies),
      )

      yield* Effect.gen(function* () {
        const assessor = yield* LocalModelAssessor
        yield* Deferred.await(firstAssessmentStarted)
        packageState = {
          ...packageState,
          entries: [{ ...packageEntry, inspection: { _tag: "Pending" as const } }],
        }
        yield* SubscriptionRef.set(packageStateRef, packageState)
        yield* Deferred.succeed(releaseFirstAssessment, undefined)
        yield* Effect.sleep("150 millis")
        expect(assessmentCalls).toBe(2)
        expect(assessmentRequestCounts).toEqual([1, 1])
        const initialState = yield* assessor.state
        expect(initialState).toHaveLength(2)
        expect(initialState[0]?.configuration).toEqual(configuration)
        expect(initialState[0]?.assessment).toEqual({
          _tag: "Incompatible",
          environmentId: AssessmentEnvironmentIdSchema.make("environment-test"),
          failure: {
            code: "unsupported_architecture",
            message: "Unsupported architecture",
            retryable: false,
          },
        })
        yield* SubscriptionRef.set(packageStateRef, packageState)
        yield* SubscriptionRef.set(packageStateRef, packageState)
        yield* Effect.sleep("100 millis")

        expect(assessmentCalls).toBe(2)
        expect((yield* assessor.state)[0]?.assessment._tag)
          .toBe("Incompatible")
      }).pipe(Effect.provide(testLayer))
    })))
  })
})
