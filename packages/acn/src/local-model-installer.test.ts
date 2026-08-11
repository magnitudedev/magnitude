import { Effect, Option, Ref, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelServingConfigurationIdSchema,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import { ModelStateSchema } from "@magnitudedev/storage"
import type { IcnCatalogService } from "@magnitudedev/icn"
import type { RecommendableModel as NativeRecommendableModel } from "@magnitudedev/icn-protocol/schemas"
import { makeRetainedModelConfigurations } from "./retained-model-configurations"
import { makeLocalModelInstaller } from "./local-model-installer"
import { makeTestModelState } from "./model-state.test-support"
import type { LocalModelPackagesApi } from "./local-model-packages"

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
  it("materializes before admission and reinstalls from retained state after catalog removal", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const initial = yield* Schema.decodeUnknown(ModelStateSchema)({
          configurations: [],
          configurationRecoveryCompleted: true,
        })
      const state = yield* makeTestModelState(initial)
      const retained = makeRetainedModelConfigurations(state)
      const catalogModels = yield* Ref.make<readonly NativeRecommendableModel[]>([
        model as unknown as NativeRecommendableModel,
      ])
      const ready = yield* Ref.make(true)
      const catalog: IcnCatalogService = {
        get: Ref.get(catalogModels).pipe(Effect.map((models) => ({
          revision: 1,
          state: { models, diagnostics: [] },
        }))) as unknown as IcnCatalogService["get"],
        changes: Stream.never,
        ready: Ref.get(ready),
        refresh: Effect.void,
      }
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
      const installer = yield* makeLocalModelInstaller(retained, packages, catalog, {
        exclusive: (operation) => operation,
      }, {
        state: Effect.succeed(new Map()),
        changes: Stream.never,
      })
      const first = yield* installer.install(model.configuration.id)
      yield* Ref.set(catalogModels, [])
      yield* Ref.set(ready, false)
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
})
