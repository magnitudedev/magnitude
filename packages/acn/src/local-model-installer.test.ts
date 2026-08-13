import { Effect, Option, Ref, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelDownloadIdSchema,
  ModelServingConfigurationIdSchema,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import { makeLocalModelInstaller } from "./local-model-installer"
import type { LocalModelPackagesApi } from "./local-model-packages"
import type {
  ResolvedLocalModelConfiguration,
  LocalModelConfigurationResolverApi,
} from "./local-model-configuration-resolver"
import { localModelTargetIdentity } from "./local-model-configuration-resolver"

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
        localModelTargetIdentity(resolved.configuration.bundle),
        resolved,
      ]]),
    }))),
    changes: Stream.never,
    catalogReady: Effect.succeed(true),
    resolve: () => Ref.get(current),
  })

  it("admits an eligible derived configuration without persisting it", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const resolution = yield* Ref.make(Option.some(fitsResolution))
      const admittedBundles: unknown[] = []
      const packages: LocalModelPackagesApi = {
        initialized: Effect.succeed(true),
        snapshot: Effect.succeed({
          revision: 0,
          state: { inventory: { _tag: "Ready" }, entries: [], downloads: [] },
        }),
        changes: Stream.never,
        installedPackageIds: Effect.succeed(new Set()),
        admitBundle: (bundle) => Effect.sync(() => {
          admittedBundles.push(bundle)
          return admittedBundles.length === 1
            ? {
                _tag: "DownloadAdmitted" as const,
                downloadId: ModelDownloadIdSchema.make("download-a"),
              }
            : { _tag: "AlreadyInstalled" as const }
        }),
        cancelDownload: () => Effect.void,
        acknowledgeFailure: () => Effect.void,
        removeBundlePackages: () => Effect.void,
      }
      const installer = yield* makeLocalModelInstaller(packages, {
        exclusive: (operation) => operation,
      }, resolverFrom(resolution))
      const first = yield* installer.install(model.configuration.id)
      return {
        first,
        admittedBundles,
      }
    }))
    expect(result.first.providerModelId).toBe("configuration-a")
    expect(result.first).toMatchObject({
      _tag: "DownloadAdmitted",
      downloadId: "download-a",
    })
    expect(result.admittedBundles).toEqual([model.configuration.bundle])
  })

  it("rejects a configuration without a current Fits resolution", async () => {
    const outcomes = await Effect.runPromise(Effect.gen(function* () {
      let admissions = 0
      const packages = {
        initialized: Effect.succeed(true),
        snapshot: Effect.succeed({
          revision: 0,
          state: { inventory: { _tag: "Ready" as const }, entries: [], downloads: [] },
        }),
        changes: Stream.never,
        installedPackageIds: Effect.succeed(new Set<string>()),
        admitBundle: () => Effect.sync(() => {
          admissions += 1
          return { _tag: "AlreadyInstalled" as const }
        }),
        cancelDownload: () => Effect.void,
        acknowledgeFailure: () => Effect.void,
        removeBundlePackages: () => Effect.void,
      } satisfies LocalModelPackagesApi
      const resolution = yield* Ref.make<Option.Option<ResolvedLocalModelConfiguration>>(Option.none())
      const installer = yield* makeLocalModelInstaller(packages, {
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
          totalRequiredBytes: 0,
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
        admissions,
      }
    }))

    expect(outcomes.tags).toEqual(["Left", "Left", "Left", "Left"])
    expect(outcomes.admissions).toBe(0)
  })
})
