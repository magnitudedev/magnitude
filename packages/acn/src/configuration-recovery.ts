import { Cause, Effect, Exit, Layer, Queue, Stream } from "effect"
import {
  servableModelBundlePackageIds,
  type ModelServingConfiguration,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnInstalledModels } from "@magnitudedev/icn"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import { RetainedModelConfigurations } from "./retained-model-configurations"
import { LocalModelConfigurationCoordinator } from "./local-model-configuration-coordinator"

export const installedCatalogConfigurations = (
  catalog: readonly RecommendableModel[],
  installedPackageIds: ReadonlySet<string>,
): readonly ModelServingConfiguration[] => catalog.flatMap(({ configuration }) =>
  servableModelBundlePackageIds(configuration.bundle).every((id) => installedPackageIds.has(id))
    ? [configuration]
    : [])

export const ConfigurationRecoveryLive: Layer.Layer<
  never,
  never,
  RetainedModelConfigurations | IcnCatalog | IcnInstalledModels | LocalModelConfigurationCoordinator
> = Layer.scopedDiscard(Effect.gen(function* () {
  const retained = yield* RetainedModelConfigurations
  const catalog = yield* IcnCatalog
  const installed = yield* IcnInstalledModels
  const coordinator = yield* LocalModelConfigurationCoordinator
  const changes = Stream.mergeAll([
    retained.changes.pipe(Stream.map(() => undefined)),
    catalog.changes.pipe(Stream.map(() => undefined)),
    installed.changes.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" })

  const reconcile = Effect.gen(function* () {
    if (yield* retained.recoveryCompleted) return
    if (!(yield* catalog.ready) || !(yield* installed.initialized)) return
    const catalogSnapshot = yield* catalog.get
    const installedSnapshot = yield* installed.get
    if (!installedSnapshot.state.reconciliationComplete) return
    const models = yield* Effect.forEach(
      catalogSnapshot.state.models,
      recommendableModelFromIcn,
    )
    const configurations = installedCatalogConfigurations(
      models,
      new Set(installedSnapshot.state.packages.map(({ package: modelPackage }) => modelPackage.id)),
    )
    const latestCatalog = yield* catalog.get
    const latestInstalled = yield* installed.get
    if (
      latestCatalog.revision !== catalogSnapshot.revision
      || latestInstalled.revision !== installedSnapshot.revision
      || latestInstalled.state.revision !== installedSnapshot.state.revision
    ) return
    yield* retained.completeRecovery(configurations)
  })

  const invalidations = yield* Queue.unbounded<void>()
  yield* Queue.offer(invalidations, undefined)
  yield* changes.pipe(
    Stream.runForEach(() => Queue.offer(invalidations, undefined)),
    Effect.forkScoped,
  )
  yield* Effect.gen(function* () {
    while (!(yield* retained.recoveryCompleted)) {
      yield* Queue.take(invalidations)
      yield* Queue.takeAll(invalidations)
      const outcome = yield* coordinator.exclusive(reconcile).pipe(Effect.exit)
      if (Exit.isFailure(outcome)) {
        yield* Effect.logWarning(
          "Unable to complete retained model configuration recovery",
        ).pipe(Effect.annotateLogs({ cause: Cause.pretty(outcome.cause) }))
        yield* Effect.sleep("1 second")
        yield* Queue.offer(invalidations, undefined)
      }
    }
  }).pipe(Effect.forkScoped)
}))
