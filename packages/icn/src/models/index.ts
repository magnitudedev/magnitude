import { Cause, Context, Duration, Effect, Layer, Schema } from "effect"
import {
  CatalogInstallationsResponse,
  CatalogModelsResponse,
  DiscoveredModelsResponse,
  ModelAssessmentsSnapshot,
} from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, type IcnClientService } from "../client.js"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"
import { IcnEvents, refreshOnIcnEvents } from "../events/index.js"

type CatalogReadError = Effect.Effect.Error<ReturnType<IcnClientService["catalog"]["listCatalogModels"]>>
type DiscoveryReadError = Effect.Effect.Error<ReturnType<IcnClientService["discovery"]["listDiscoveredModels"]>>
type DiscoveryRefreshError = Effect.Effect.Error<ReturnType<IcnClientService["discovery"]["refreshDiscoveredModels"]>>
type CatalogInstallationsReadError = Effect.Effect.Error<ReturnType<IcnClientService["catalog"]["listCatalogInstallations"]>>
type AssessmentsReadError = Effect.Effect.Error<ReturnType<IcnClientService["models"]["getModelAssessments"]>>

export interface IcnCatalogService
  extends IcnObservedState<typeof CatalogModelsResponse.Type, CatalogReadError> {}
export class IcnCatalog extends Context.Tag("@magnitudedev/icn/IcnCatalog")<IcnCatalog, IcnCatalogService>() {}

export interface IcnDiscoveryService
  extends IcnObservedState<typeof DiscoveredModelsResponse.Type, DiscoveryReadError> {
  readonly reconcile: Effect.Effect<void, DiscoveryRefreshError | DiscoveryReadError | Cause.TimeoutException>
}
export class IcnDiscovery extends Context.Tag("@magnitudedev/icn/IcnDiscovery")<IcnDiscovery, IcnDiscoveryService>() {}

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
export interface IcnDiscoveryOptions extends IcnModelDomainOptions {
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
  const observed = yield* makeIcnObservedState(initial, read, Schema.equivalence(CatalogModelsResponse), {
    initiallyInitialized: true,
  })
  yield* refreshOnIcnEvents(invalidations, new Set(["catalog"]), observed.refresh, "ICN catalog", retryInterval)
    .pipe(Effect.forkScoped)
  return IcnCatalog.of(observed)
}))

export const makeIcnDiscovery = (
  options: IcnDiscoveryOptions = {},
): Layer.Layer<IcnDiscovery, DiscoveryReadError | Cause.TimeoutException, IcnClient | IcnEvents> => Layer.scoped(IcnDiscovery, Effect.gen(function* () {
  const client = yield* IcnClient
  const invalidations = yield* (yield* IcnEvents).subscribe
  const read = client.discovery.listDiscoveredModels({})
  const retryInterval = options.retryInterval ?? "1 second"
  const initial = yield* read
  const observed = yield* makeIcnObservedState(initial, read, Schema.equivalence(DiscoveredModelsResponse), {
    initiallyInitialized: true,
  })
  yield* refreshOnIcnEvents(invalidations, new Set(["discovery"]), observed.refresh, "ICN discovery", retryInterval)
    .pipe(Effect.forkScoped)
  return IcnDiscovery.of({
    ...observed,
    reconcile: client.discovery.refreshDiscoveredModels({}).pipe(
      Effect.timeout(options.reconciliationTimeout ?? "2 minutes"),
      Effect.asVoid,
      Effect.zipRight(observed.refresh),
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
      Schema.equivalence(ModelAssessmentsSnapshot),
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
      Schema.equivalence(CatalogInstallationsResponse),
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
