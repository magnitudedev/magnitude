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
import { RetainedModelConfigurations } from "./retained-model-configurations"

describe("LocalModelAssessor", () => {
  it("does not reassess unchanged semantic evidence", async () => {
    let assessmentCalls = 0

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const packageChanges = yield* PubSub.unbounded<{
        readonly revision: number
        readonly state: {
          readonly inventory: { readonly _tag: "Ready" }
          readonly entries: readonly ModelPackageEntry[]
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
        state: { inventory: { _tag: "Ready" as const }, entries: [packageEntry] },
      }
      const configuration = {
        id: ModelServingConfigurationIdSchema.make("configuration-test"),
        bundle: { _tag: "Standalone" as const, package: modelPackage },
        profile: { contextLength: 32_768 },
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
          cancelAttempts: () => Effect.dieMessage("unused"),
          acknowledgeFailures: () => Effect.dieMessage("unused"),
          removeBundlePackages: () => Effect.dieMessage("unused"),
        })),
        Layer.succeed(LocalModelAssessments, LocalModelAssessments.of({
          assess: (requests) => Effect.sync(() => {
            assessmentCalls += 1
            return requests.map(() => ({
              _tag: "InvalidBundle" as const,
              message: "terminal test result",
            }))
          }),
        })),
      )
      const testLayer = LocalModelAssessorLive.pipe(Layer.provide(dependencies))

      yield* Effect.gen(function* () {
        const assessor = yield* LocalModelAssessor
        yield* Effect.sleep("100 millis")
        expect(assessmentCalls).toBe(1)
        const initialState = yield* assessor.state
        expect([...initialState.keys()]).toEqual([configuration.id])
        expect([...initialState.values()].every(({ assessment }) =>
          assessment._tag === "Failed")).toBe(true)

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
