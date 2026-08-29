import { describe, expect, it } from "vitest"
import { Effect, Layer, Stream, SubscriptionRef } from "effect"
import { IcnModels } from "@magnitudedev/icn"
import { LocalModelAssessmentReady, LocalModelAssessor } from "./local-model-assessor"
import { LocalModelCatalogAdapterLive } from "./local-model-catalog-adapter"
import {
  LocalModelConfigurationResolver,
  LocalModelConfigurationResolverLive,
} from "./local-model-configuration-resolver"
import { LocalModelPackages } from "./local-model-packages"

describe("LocalModelConfigurationResolver", () => {
  it("serves retained state without rereading source snapshots", async () => {
    let modelReads = 0
    let assessmentReads = 0
    let packageReads = 0

    const dependencies = Layer.mergeAll(
      Layer.succeed(IcnModels, IcnModels.of({
        get: Effect.sync(() => {
          modelReads += 1
          return {
            revision: 1,
            state: {
              revision: 1,
              reconciliationComplete: true,
              models: [],
              diagnostics: [],
            },
          }
        }),
        changes: Stream.never,
        initialized: Effect.succeed(true),
        refresh: Effect.void,
      })),
      Layer.succeed(LocalModelAssessor, LocalModelAssessor.of({
        snapshot: Effect.sync(() => {
          assessmentReads += 1
          return {
            assessments: [],
            lifecycle: new LocalModelAssessmentReady({
              cycle: {
                startedAtMs: 0,
                durationMs: 0,
                completedTargets: 0,
                totalTargets: 0,
              },
            }),
          }
        }),
        changes: Stream.never,
      })),
      Layer.effect(LocalModelPackages, Effect.gen(function* () {
        const state = yield* SubscriptionRef.make({
          inventory: { _tag: "Ready" as const },
          entries: [],
          downloads: [],
        })
        return LocalModelPackages.of({
          initialized: Effect.succeed(true),
          state: SubscriptionRef.get(state).pipe(Effect.tap(() => Effect.sync(() => {
            packageReads += 1
          }))),
          changes: state.changes,
          installedPackageIds: Effect.succeed(new Set()),
          refresh: Effect.void,
        })
      })),
    )

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const resolver = yield* LocalModelConfigurationResolver
      const readsAfterInitialization = { modelReads, assessmentReads, packageReads }
      yield* Effect.all(Array.from({ length: 100 }, () => resolver.get))
      expect({ modelReads, assessmentReads, packageReads }).toEqual(readsAfterInitialization)
    }).pipe(
      Effect.provide(LocalModelConfigurationResolverLive.pipe(
        Layer.provide(LocalModelCatalogAdapterLive.pipe(Layer.provideMerge(dependencies))),
        Layer.provide(dependencies),
      )),
    )))
  })
})
