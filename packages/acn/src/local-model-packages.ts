import {
  Context,
  Effect,
  Layer,
  Option,
  Schema,
  Stream,
  SubscriptionRef,
} from "effect"
import {
  InstalledCatalogAttributionSchema,
  LocalModelMutationFailed,
  ModelPackagesStateSchema,
  servableModelBundlePackages,
  type ModelPackage,
  type ModelPackageEntry,
  type ModelPackageId,
  type ModelPackageInstallationOrigin,
  type ModelPackagesState,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import {
  IcnDownloads,
  IcnInstalledModels,
  IcnModels,
} from "@magnitudedev/icn"
import {
  modelDownloadFromIcn,
  modelPackageFromIcn,
  packageInspectionFromIcn,
  catalogModelDefinitionFromIcn,
} from "./local-model-icn-adapter"

const packagesInCatalog = (
  catalog: readonly RecommendableModel[],
): readonly ModelPackage[] => {
  const packages = new Map<ModelPackageId, ModelPackage>()
  for (const recommendable of catalog) {
    for (const modelPackage of servableModelBundlePackages(recommendable.configuration.bundle)) {
      packages.set(modelPackage.id, modelPackage)
    }
  }
  return [...packages.values()]
}

export type PackageAcquisition =
  | { readonly _tag: "NotInstalled" }
  | { readonly _tag: "Installed"; readonly path: string; readonly origin: ModelPackageInstallationOrigin }

export const projectPackageAcquisition = (
  acquisition: PackageAcquisition,
): ModelPackageEntry["localState"] => {
  switch (acquisition._tag) {
    case "Installed":
      return { _tag: "Installed", path: acquisition.path, origin: acquisition.origin }
    case "NotInstalled":
      return { _tag: "NotInstalled" }
  }
}

export const packageAcquisition = (
  modelPackage: ModelPackage,
  installedPackages: ReadonlyMap<ModelPackageId, {
    readonly path: string
    readonly origin: ModelPackageInstallationOrigin
  }>,
): PackageAcquisition => {
  const installed = installedPackages.get(modelPackage.id)
  if (installed !== undefined) return { _tag: "Installed", ...installed }
  return { _tag: "NotInstalled" }
}

export interface LocalModelPackagesApi {
  readonly initialized: Effect.Effect<boolean>
  readonly state: Effect.Effect<ModelPackagesState>
  readonly changes: Stream.Stream<ModelPackagesState>
  readonly installedPackageIds: Effect.Effect<ReadonlySet<string>>
  readonly refresh: Effect.Effect<void, LocalModelMutationFailed>
}

export class LocalModelPackages extends Context.Tag("LocalModelPackages")<
  LocalModelPackages,
  LocalModelPackagesApi
>() {}

export const LocalModelPackagesLive: Layer.Layer<
  LocalModelPackages,
  never,
  IcnModels | IcnDownloads | IcnInstalledModels
> = Layer.scoped(LocalModelPackages, Effect.gen(function* () {
  const models = yield* IcnModels
  const installed = yield* IcnInstalledModels
  const downloads = yield* IcnDownloads
  const current = yield* SubscriptionRef.make<ModelPackagesState>({
    inventory: { _tag: "Initializing" },
    entries: [],
    downloads: [],
  })
  const equivalent = Schema.equivalence(ModelPackagesStateSchema)
  const publish = (next: ModelPackagesState) => Effect.gen(function* () {
    const previous = yield* SubscriptionRef.get(current)
    if (!equivalent(previous, next)) yield* SubscriptionRef.set(current, next)
  })
  const projectionLock = yield* Effect.makeSemaphore(1)
  const observedCompletions = new Set<string>()

  const projectCurrent = projectionLock.withPermits(1)(Effect.gen(function* () {
    const catalogModels = yield* Effect.forEach(
      (yield* models.get).state.models,
      catalogModelDefinitionFromIcn,
    )
    const downloadsState = (yield* downloads.get).state
    const newlyCompleted = downloadsState.downloads.filter(({ id, state }) =>
      state._tag === "Completed" && !observedCompletions.has(id))
    if (newlyCompleted.length > 0) {
      // ICN publishes installed inventory before completing package work. Refreshing
      // here closes the independent-observer race without treating historical
      // completion as proof that files are still present.
      yield* installed.refresh
      for (const { id } of newlyCompleted) observedCompletions.add(id)
    }
    const installedState = (yield* installed.get).state
    const installedModels = yield* Effect.forEach(
      installedState.packages,
      (entry) => Effect.all({
        package: modelPackageFromIcn(entry.package),
        path: Effect.succeed(entry.path),
        origin: Effect.succeed(entry.origin),
        inspection: packageInspectionFromIcn(entry.inspection),
        catalogAttribution: Schema.validate(InstalledCatalogAttributionSchema)(
          entry.catalogAttribution,
        ),
      }),
    )
    const bundleDownloads = yield* Effect.forEach(
      downloadsState.downloads,
      modelDownloadFromIcn,
    )
    const catalogPackages = packagesInCatalog(catalogModels)
    const allPackages = new Map<ModelPackageId, ModelPackage>(
      catalogPackages.map((modelPackage) => [modelPackage.id, modelPackage]),
    )
    for (const item of installedModels) {
      allPackages.set(item.package.id, item.package)
    }
    const installedById = new Map(installedModels.map((item) => [item.package.id, item]))
    const installedPackages = new Map(installedModels.map((item) => [item.package.id, {
      path: item.path,
      origin: item.origin,
    }]))

    const entries: ModelPackageEntry[] = [...allPackages.values()]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((modelPackage) => {
        const installedEntry = installedById.get(modelPackage.id)
        const acquisition = packageAcquisition(
          modelPackage,
          installedPackages,
        )
        const localState = projectPackageAcquisition(acquisition)
        return {
          package: modelPackage,
          localState,
          inspection: installedEntry?.inspection ?? { _tag: "Pending" },
          catalogAttribution: installedEntry?.catalogAttribution
            ?? { _tag: "NotCatalogTarget" },
        }
      })
    yield* publish({
      inventory: installedState.reconciliationComplete
        ? { _tag: "Ready" }
        : { _tag: "Initializing" },
      entries,
      downloads: bundleDownloads,
    })
  }))
  const project = projectCurrent.pipe(
    Effect.catchAllCause((cause) =>
      SubscriptionRef.get(current).pipe(
        Effect.flatMap((state) => publish({
          ...state,
          inventory: {
            _tag: "Degraded",
            failure: {
              code: "local_model_inventory_unavailable",
              message: "Local model inventory could not be refreshed",
              retryable: true,
            },
          },
        })),
        Effect.zipRight(Effect.logWarning("Unable to project local model packages").pipe(
          Effect.annotateLogs({ cause: String(cause) }),
        )),
      ),
    ),
  )

  yield* project
  yield* Stream.mergeAll([
    models.changes.pipe(Stream.map(() => undefined)),
    installed.changes.pipe(Stream.map(() => undefined)),
    downloads.changes.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" }).pipe(
    Stream.runForEach(() => project),
    Effect.forkScoped,
  )

  return LocalModelPackages.of({
    initialized: installed.initialized,
    state: SubscriptionRef.get(current),
    changes: current.changes,
    installedPackageIds: installed.get.pipe(Effect.map(({ state }) =>
      new Set(state.packages.map(({ package: modelPackage }) => modelPackage.id)))),
    refresh: Effect.all([
      models.refresh,
      downloads.refresh,
      installed.refresh,
    ], { discard: true }).pipe(
      Effect.andThen(projectCurrent),
      Effect.catchAllCause((cause) => Effect.logWarning("Unable to refresh local model packages").pipe(
        Effect.annotateLogs({ cause: String(cause) }),
        Effect.andThen(Effect.fail(new LocalModelMutationFailed({
          code: "local_model_state_refresh_failed",
          message: "The local model state could not be refreshed",
          retryable: true,
        }))),
      )),
    ),
  })
}))
