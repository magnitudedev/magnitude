import { Cause, Context, Duration, Effect, Layer } from "effect"
import { dataEquivalence } from "@magnitudedev/utils/data-equivalence"
import {
  CatalogInstallationsResponse,
  CatalogModelsResponse,
  LocalModelsResponse,
  ModelAssessmentsSnapshot,
  type DrafterPruning,
  type LocalModelRemoval,
  type LocalModelsConfiguration,
} from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, type IcnClientService } from "../client.js"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"
import { IcnEvents, refreshOnIcnEvents } from "../events/index.js"

type CatalogReadError = Effect.Effect.Error<ReturnType<IcnClientService["catalog"]["listCatalogModels"]>>
type LocalModelsReadError = Effect.Effect.Error<ReturnType<IcnClientService["localModels"]["listLocalModels"]>>
type LocalModelsRefreshError = Effect.Effect.Error<ReturnType<IcnClientService["localModels"]["refreshLocalModels"]>>
export type LocalModelFilesRemovalError = Effect.Effect.Error<ReturnType<IcnClientService["localModels"]["removeLocalModelFiles"]>>
export type LocalModelsConfigureError = Effect.Effect.Error<ReturnType<IcnClientService["localModels"]["configureLocalModels"]>>
export type DrafterPruningError = Effect.Effect.Error<ReturnType<IcnClientService["localModels"]["pruneLocalDrafters"]>>
type CatalogInstallationsReadError = Effect.Effect.Error<ReturnType<IcnClientService["catalog"]["listCatalogInstallations"]>>
type AssessmentsReadError = Effect.Effect.Error<ReturnType<IcnClientService["models"]["getModelAssessments"]>>

export interface IcnCatalogService
  extends IcnObservedState<typeof CatalogModelsResponse.Type, CatalogReadError> {}
export class IcnCatalog extends Context.Tag("@magnitudedev/icn/IcnCatalog")<IcnCatalog, IcnCatalogService>() {}

/**
 * Custom and unsupported local models ICN found in the configured locations and declarations.
 * `configure` replaces the whole local-models configuration (the same snapshot twice is a no-op);
 * `removeFiles` deletes Magnitude's copy of an added model and never touches other locations;
 * `pruneDrafters` deletes Magnitude's drafter packages that no model uses any more.
 */
export interface IcnLocalModelsService
  extends IcnObservedState<typeof LocalModelsResponse.Type, LocalModelsReadError> {
  readonly reconcile: Effect.Effect<void, LocalModelsRefreshError | LocalModelsReadError | Cause.TimeoutException>
  readonly configure: (
    configuration: LocalModelsConfiguration,
  ) => Effect.Effect<void, LocalModelsConfigureError | LocalModelsReadError>
  readonly removeFiles: (modelId: string) => Effect.Effect<LocalModelRemoval, LocalModelFilesRemovalError>
  readonly pruneDrafters: Effect.Effect<DrafterPruning, DrafterPruningError>
}
export class IcnLocalModels extends Context.Tag("@magnitudedev/icn/IcnLocalModels")<IcnLocalModels, IcnLocalModelsService>() {}

export interface IcnCatalogInstallationsService
  extends IcnObservedState<typeof CatalogInstallationsResponse.Type, CatalogInstallationsReadError> {}
export class IcnCatalogInstallations extends Context.Tag("@magnitudedev/icn/IcnCatalogInstallations")<
  IcnCatalogInstallations,
  IcnCatalogInstallationsService
>() {}

export interface IcnModelAssessmentsService
  extends IcnObservedState<typeof ModelAssessmentsSnapshot.Type, AssessmentsReadError> {}
export class IcnModelAssessments extends Context.Tag("@magnitudedev/icn/IcnModelAssessments")<
  IcnModelAssessments,
  IcnModelAssessmentsService
>() {}

export interface IcnModelDomainOptions { readonly retryInterval?: Duration.DurationInput }
export interface IcnLocalModelsOptions extends IcnModelDomainOptions {
  readonly reconciliationTimeout?: Duration.DurationInput
}

export const makeIcnCatalog = (
  options: IcnModelDomainOptions = {},
): Layer.Layer<IcnCatalog, CatalogReadError, IcnClient | IcnEvents> => Layer.scoped(IcnCatalog, Effect.gen(function* () {
  const client = yield* IcnClient
  const invalidations = yield* (yield* IcnEvents).subscribe
  const read = client.catalog.listCatalogModels({})
  const retryInterval = options.retryInterval ?? "1 second"
  const initial = yield* read
  const observed = yield* makeIcnObservedState(initial, read, dataEquivalence(), {
    initiallyInitialized: true,
  })
  yield* refreshOnIcnEvents(invalidations, new Set(["catalog"]), observed.refresh, "ICN catalog", retryInterval)
    .pipe(Effect.forkScoped)
  return IcnCatalog.of(observed)
}))

export const makeIcnLocalModels = (
  options: IcnLocalModelsOptions = {},
): Layer.Layer<IcnLocalModels, LocalModelsReadError, IcnClient | IcnEvents> => Layer.scoped(IcnLocalModels, Effect.gen(function* () {
  const client = yield* IcnClient
  const invalidations = yield* (yield* IcnEvents).subscribe
  const read = client.localModels.listLocalModels({})
  const retryInterval = options.retryInterval ?? "1 second"
  const initial = yield* read
  const observed = yield* makeIcnObservedState(initial, read, dataEquivalence(), {
    initiallyInitialized: true,
  })
  yield* refreshOnIcnEvents(invalidations, new Set(["local-models"]), observed.refresh, "ICN local models", retryInterval)
    .pipe(Effect.forkScoped)
  return IcnLocalModels.of({
    ...observed,
    reconcile: client.localModels.refreshLocalModels({}).pipe(
      Effect.timeout(options.reconciliationTimeout ?? "2 minutes"),
      Effect.asVoid,
      Effect.zipRight(observed.refresh),
    ),
    configure: (configuration) => client.localModels.configureLocalModels({ payload: configuration }).pipe(
      Effect.zipRight(observed.refresh),
    ),
    removeFiles: (modelId) => client.localModels.removeLocalModelFiles({ path: { model_id: modelId } }).pipe(
      Effect.tap(() => observed.refresh.pipe(Effect.ignore)),
    ),
    pruneDrafters: Effect.suspend(() => client.localModels.pruneLocalDrafters({})).pipe(
      Effect.tap(({ removed }) => removed.length > 0 ? observed.refresh.pipe(Effect.ignore) : Effect.void),
    ),
  })
}))

export const makeIcnModelAssessments = (
  options: IcnModelDomainOptions = {},
): Layer.Layer<IcnModelAssessments, AssessmentsReadError, IcnClient | IcnEvents> =>
  Layer.scoped(IcnModelAssessments, Effect.gen(function* () {
    const client = yield* IcnClient
    const invalidations = yield* (yield* IcnEvents).subscribe
    const read = client.models.getModelAssessments({})
    const retryInterval = options.retryInterval ?? "1 second"
    const initial = yield* read
    const observed = yield* makeIcnObservedState(
      initial,
      read,
      dataEquivalence(),
      { initiallyInitialized: true },
    )
    yield* refreshOnIcnEvents(
      invalidations,
      new Set(["model-assessments"]),
      observed.refresh,
      "ICN model assessments",
      retryInterval,
    ).pipe(Effect.forkScoped)
    return IcnModelAssessments.of(observed)
  }))

export const makeIcnCatalogInstallations = (
  options: IcnModelDomainOptions = {},
): Layer.Layer<IcnCatalogInstallations, CatalogInstallationsReadError, IcnClient | IcnEvents> =>
  Layer.scoped(IcnCatalogInstallations, Effect.gen(function* () {
    const client = yield* IcnClient
    const invalidations = yield* (yield* IcnEvents).subscribe
    const read = client.catalog.listCatalogInstallations({})
    const retryInterval = options.retryInterval ?? "1 second"
    const initial = yield* read
    const observed = yield* makeIcnObservedState(
      initial,
      read,
      dataEquivalence(),
      { initiallyInitialized: true },
    )
    yield* refreshOnIcnEvents(
      invalidations,
      new Set(["catalog-installations"]),
      observed.refresh,
      "ICN catalog installations",
      retryInterval,
    ).pipe(Effect.forkScoped)
    return IcnCatalogInstallations.of(observed)
  }))
