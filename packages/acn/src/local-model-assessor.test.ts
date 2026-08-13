import { Effect, Layer, Option, PubSub, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelFileIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  type ModelPackageEntry,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnHardware } from "@magnitudedev/icn"
import { LocalModelAssessments } from "./local-model-assessments"
import { LocalModelAssessor, LocalModelAssessorLive } from "./local-model-assessor"
import { LocalModelPackages } from "./local-model-packages"

describe("LocalModelAssessor", () => {
  it("does not reassess unchanged semantic evidence", async () => {
    let assessmentCalls = 0

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const packageChanges = yield* PubSub.unbounded<{
        readonly revision: number
        readonly state: {
          readonly inventory: { readonly _tag: "Ready" }
          readonly entries: readonly ModelPackageEntry[]
          readonly downloads: readonly never[]
        }
      }>()
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
          maximumContextLength: 32_768,
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
      }
      const packageSnapshot = {
        revision: 1,
        state: { inventory: { _tag: "Ready" as const }, entries: [packageEntry], downloads: [] },
      }
      const configuration = {
        id: ModelServingConfigurationIdSchema.make("configuration-test"),
        bundle: { _tag: "Standalone" as const, package: modelPackage },
        profile: { contextLength: 32_768 },
      }
      const siblingConfiguration = {
        ...configuration,
        id: ModelServingConfigurationIdSchema.make("configuration-sibling"),
      }
      const catalogModel = (id: string, configuration: typeof siblingConfiguration) => ({
        id,
        checkpointId: "checkpoint-test",
        configuration,
        displayName: "Test",
        variantLabel: "Q4",
        description: "Test",
        license: "test",
        capabilities: packageEntry.inspection._tag === "Inspected"
          ? packageEntry.inspection.capabilities
          : undefined,
        qualityScore: 1,
        qualityScoreProvenance: "test",
        fidelityRank: 1,
        quantizationAware: false,
        qualityEvidence: [],
      })
      const dependencies = Layer.mergeAll(
        Layer.succeed(IcnCatalog, IcnCatalog.of({
          get: Effect.succeed({
            revision: 1,
            state: {
              models: [
                catalogModel("recommendable-test", configuration),
                catalogModel("recommendable-sibling", siblingConfiguration),
              ] as never,
              diagnostics: [],
            },
          }),
          changes: Stream.never,
          ready: Effect.succeed(true),
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
          snapshot: Effect.succeed(packageSnapshot),
          changes: Stream.fromPubSub(packageChanges),
          installedPackageIds: Effect.succeed(new Set([packageId])),
          admitBundle: () => Effect.dieMessage("unused"),
          cancelDownload: () => Effect.dieMessage("unused"),
          acknowledgeFailure: () => Effect.dieMessage("unused"),
          removeBundlePackages: () => Effect.dieMessage("unused"),
        })),
        Layer.succeed(LocalModelAssessments, LocalModelAssessments.of({
          assess: (requests) => Effect.sync(() => {
            assessmentCalls += 1
            return requests.map((_, index) => index === 0
              ? {
                  _tag: "Failed" as const,
                  failure: {
                    code: "planning_worker_defect",
                    message: "failed to create llama context",
                    retryable: true,
                  },
                }
              : {
                  _tag: "InvalidBundle" as const,
                  message: "terminal test result",
                })
          }),
        })),
      )
      const testLayer = LocalModelAssessorLive.pipe(Layer.provide(dependencies))

      yield* Effect.gen(function* () {
        const assessor = yield* LocalModelAssessor
        yield* Effect.sleep("100 millis")
        expect(assessmentCalls).toBe(1)
        const initialState = yield* assessor.state
        expect([...initialState.keys()]).toEqual([configuration.id, siblingConfiguration.id])
        expect([...initialState.values()].every(({ assessment }) =>
          assessment._tag === "Failed")).toBe(true)
        expect(initialState.get(configuration.id)?.assessment).toEqual({
          _tag: "Failed",
          failure: {
            code: "planning_worker_defect",
            message: "failed to create llama context",
            retryable: true,
          },
        })
        expect(initialState.get(siblingConfiguration.id)?.assessment).toEqual({
          _tag: "Failed",
          failure: {
            code: "invalid_model_bundle",
            message: "terminal test result",
            retryable: false,
          },
        })

        yield* PubSub.publish(packageChanges, packageSnapshot)
        yield* PubSub.publish(packageChanges, packageSnapshot)
        yield* Effect.sleep("100 millis")

        expect(assessmentCalls).toBe(1)
        expect([...(yield* assessor.state).values()].every(({ assessment }) =>
          assessment._tag === "Failed")).toBe(true)
      }).pipe(Effect.provide(testLayer))
    })))
  })
})
