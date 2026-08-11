import { Effect, Option, Ref, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelServingConfigurationIdSchema,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import { ModelStateSchema } from "@magnitudedev/storage"
import { makeRetainedModelConfigurations } from "./retained-model-configurations"
import { makeLocalModelInstaller } from "./local-model-installer"
import { makeTestModelState } from "./model-state.test-support"
import type { LocalModelPackagesApi } from "./local-model-packages"
import type {
  ResolvedLocalModelConfiguration,
  LocalModelConfigurationResolverApi,
} from "./local-model-configuration-resolver"
import { localModelBundleIdentity } from "./local-model-configuration-resolver"

const model = {
  id: "recommendable-a",
  checkpointId: "checkpoint-a",
  configuration: {
    id: ModelServingConfigurationIdSchema.make("configuration-a"),
    bundle: {
      _tag: "Standalone",
      package: {
        id: "package-a",
        source: { _tag: "Local", path: "/models" },
        files: [],
        relationships: [],
        properties: {
          format: "gguf",
          quantization: "Q4",
          quantizationName: "4-bit",
          architecture: "test",
          maximumContextLength: 65_536,
        },
      },
    },
    profile: { contextLength: 32_768 },
  },
  displayName: "Model A",
  description: "",
  license: "test",
  capabilities: {
    vision: false,
    tools: true,
    structuredOutput: true,
    reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
  },
  qualityScore: 1,
  qualityScoreProvenance: "test",
  fidelityRank: 1,
  quantizationAware: false,
  qualityEvidence: [],
} as unknown as RecommendableModel

describe("LocalModelInstaller", () => {
  const fitsResolution: ResolvedLocalModelConfiguration = {
    configuration: model.configuration,
    assessment: Option.some({
      _tag: "Fits",
      assessment: { _tag: "Fits" } as never,
    }),
  }

  const resolverFrom = (
    current: Ref.Ref<Option.Option<ResolvedLocalModelConfiguration>>,
  ): LocalModelConfigurationResolverApi => ({
    get: Ref.get(current).pipe(Effect.map(Option.match({
      onNone: () => new Map(),
      onSome: (resolved) => new Map([[
        localModelBundleIdentity(resolved.configuration.bundle),
        resolved,
      ]]),
    }))),
    changes: Stream.never,
    catalogReady: Effect.succeed(true),
    resolve: () => Ref.get(current),
  })

  it("materializes an eligible resolution before admission and reinstalls from retained state", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const initial = yield* Schema.decodeUnknown(ModelStateSchema)({
          configurations: [],
          configurationRecoveryCompleted: true,
        })
      const state = yield* makeTestModelState(initial)
      const retained = makeRetainedModelConfigurations(state)
      const resolution = yield* Ref.make(Option.some(fitsResolution))
      const admittedBundles: unknown[] = []
      const packages: LocalModelPackagesApi = {
        initialized: Effect.succeed(true),
        snapshot: Effect.succeed({
          revision: 0,
          state: { inventory: { _tag: "Ready" }, entries: [] },
        }),
        changes: Stream.never,
        installedPackageIds: Effect.succeed(new Set()),
        admitBundle: (bundle) => Effect.sync(() => {
          admittedBundles.push(bundle)
          return { _tag: "AlreadyInstalled" as const }
        }),
        cancelAttempts: () => Effect.void,
        acknowledgeFailures: () => Effect.void,
        removeBundlePackages: () => Effect.void,
      }
      const installer = yield* makeLocalModelInstaller(retained, packages, {
        exclusive: (operation) => operation,
      }, resolverFrom(resolution))
      const first = yield* installer.install(model.configuration.id)
      yield* Ref.set(resolution, Option.none())
      const second = yield* installer.install(model.configuration.id)
      return {
        first,
        second,
        retained: yield* retained.get,
        admittedBundles,
      }
    }))
    expect(result.first.providerModelId).toBe("configuration-a")
    expect(result.second.providerModelId).toBe("configuration-a")
    expect(result.retained.map(({ id }) => id)).toEqual(["configuration-a"])
    expect(result.admittedBundles).toHaveLength(2)
  })

  it("rejects an unretained configuration without a current Fits resolution", async () => {
    const outcomes = await Effect.runPromise(Effect.gen(function* () {
      const initial = yield* Schema.decodeUnknown(ModelStateSchema)({
        configurations: [],
        configurationRecoveryCompleted: true,
      })
      const state = yield* makeTestModelState(initial)
      const retained = makeRetainedModelConfigurations(state)
      let admissions = 0
      const packages = {
        initialized: Effect.succeed(true),
        snapshot: Effect.succeed({
          revision: 0,
          state: { inventory: { _tag: "Ready" as const }, entries: [] },
        }),
        changes: Stream.never,
        installedPackageIds: Effect.succeed(new Set<string>()),
        admitBundle: () => Effect.sync(() => {
          admissions += 1
          return { _tag: "AlreadyInstalled" as const }
        }),
        cancelAttempts: () => Effect.void,
        acknowledgeFailures: () => Effect.void,
        removeBundlePackages: () => Effect.void,
      } satisfies LocalModelPackagesApi
      const resolution = yield* Ref.make<Option.Option<ResolvedLocalModelConfiguration>>(Option.none())
      const installer = yield* makeLocalModelInstaller(retained, packages, {
        exclusive: (operation) => operation,
      }, resolverFrom(resolution))

      const missing = yield* installer.install(model.configuration.id).pipe(Effect.either)
      yield* Ref.set(resolution, Option.some({
        configuration: model.configuration,
        assessment: Option.some({ _tag: "Assessing" }),
      }))
      const assessing = yield* installer.install(model.configuration.id).pipe(Effect.either)
      yield* Ref.set(resolution, Option.some({
        configuration: model.configuration,
        assessment: Option.some({
          _tag: "DoesNotFit",
          assessmentId: "assessment-a" as never,
          environmentId: "environment-a" as never,
          memory: [],
          deficitBytes: 1,
          limitingResource: "system memory",
        }),
      }))
      const doesNotFit = yield* installer.install(model.configuration.id).pipe(Effect.either)
      yield* Ref.set(resolution, Option.some({
        configuration: model.configuration,
        assessment: Option.some({
          _tag: "Failed",
          failure: {
            code: "assessment_failed",
            message: "assessment failed",
            retryable: true,
          },
        }),
      }))
      const failed = yield* installer.install(model.configuration.id).pipe(Effect.either)
      return {
        tags: [missing, assessing, doesNotFit, failed].map((outcome) => outcome._tag),
        retained: yield* retained.get,
        admissions,
      }
    }))

    expect(outcomes.tags).toEqual(["Left", "Left", "Left", "Left"])
    expect(outcomes.retained).toEqual([])
    expect(outcomes.admissions).toBe(0)
  })
})
