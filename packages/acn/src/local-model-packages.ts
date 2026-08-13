import {
  Context,
  Effect,
  Layer,
  Option,
  Schema,
  Stream,
  type Equivalence,
} from "effect"
import {
  LocalModelMutationFailed,
  ModelDownloadIdSchema,
  ModelPackageIdSchema,
  ModelPackagesStateSchema,
  servableModelBundlePackages,
  type ModelDownloadId,
  type LocalInferenceError,
  type ModelPackage,
  type ModelPackageEntry,
  type ModelPackageId,
  type ModelPackageInstallationOrigin,
  type ModelPackagesState,
  type ServableModelBundle,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import {
  IcnCatalog,
  IcnClient,
  IcnDownloads,
  IcnInstalledModels,
} from "@magnitudedev/icn"
import { makeObservedState } from "./mirrored-state"
import {
  modelDownloadFromIcn,
  modelPackageFromIcn,
  servableModelBundleToIcn,
  packageInspectionFromIcn,
  recommendableModelFromIcn,
} from "./local-model-icn-adapter"

export const localModelPackageMutationFailure = <Failure>(operation: string, error: Failure) =>
  error instanceof LocalModelMutationFailed
    ? error
    : new LocalModelMutationFailed({
        code: operation,
        message: error instanceof Error ? error.message : String(error),
        retryable: true,
      })

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
  readonly snapshot: Effect.Effect<{ readonly revision: number; readonly state: ModelPackagesState }>
  readonly changes: Stream.Stream<{ readonly revision: number; readonly state: ModelPackagesState }>
  readonly installedPackageIds: Effect.Effect<ReadonlySet<string>>
  readonly admitBundle: (
    bundle: ServableModelBundle,
  ) => Effect.Effect<BundleInstallationAdmission, LocalInferenceError>
  readonly cancelDownload: (downloadId: ModelDownloadId) => Effect.Effect<void, LocalInferenceError>
  readonly acknowledgeFailure: (downloadId: ModelDownloadId) => Effect.Effect<void, LocalInferenceError>
  readonly removeBundlePackages: (
    bundle: ServableModelBundle,
    retainedPackageIds?: ReadonlySet<string>,
  ) => Effect.Effect<void, LocalInferenceError>
}

export type BundleInstallationAdmission =
  | { readonly _tag: "AlreadyInstalled" }
  | {
      readonly _tag: "DownloadAdmitted"
      readonly downloadId: ModelDownloadId
    }

export class LocalModelPackages extends Context.Tag("LocalModelPackages")<
  LocalModelPackages,
  LocalModelPackagesApi
>() {}

export const LocalModelPackagesLive: Layer.Layer<
  LocalModelPackages,
  never,
  IcnCatalog | IcnClient | IcnDownloads | IcnInstalledModels
> = Layer.scoped(LocalModelPackages, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const installed = yield* IcnInstalledModels
  const downloads = yield* IcnDownloads
  const client = yield* IcnClient
  const mirror = yield* makeObservedState<ModelPackagesState>({
    inventory: { _tag: "Initializing" },
    entries: [],
    downloads: [],
  })
  const equivalent: Equivalence.Equivalence<ModelPackagesState> =
    Schema.equivalence(ModelPackagesStateSchema)
  const projectionLock = yield* Effect.makeSemaphore(1)
  const observedCompletions = new Set<string>()

  const project = projectionLock.withPermits(1)(Effect.gen(function* () {
    const catalogModels = yield* Effect.forEach(
      (yield* catalog.get).state.models,
      recommendableModelFromIcn,
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
        }
      })
    yield* mirror.setIfChanged({
      inventory: installedState.reconciliationComplete
        ? { _tag: "Ready" }
        : { _tag: "Initializing" },
      entries,
      downloads: bundleDownloads,
    }, equivalent)
  })).pipe(
    Effect.catchAllCause((cause) =>
      mirror.get.pipe(
        Effect.flatMap(({ state }) => mirror.setIfChanged({
          ...state,
          inventory: {
            _tag: "Degraded",
            failure: {
              code: "local_model_inventory_unavailable",
              message: "Local model inventory could not be refreshed",
              retryable: true,
            },
          },
        }, equivalent)),
        Effect.zipRight(Effect.logWarning("Unable to project local model packages").pipe(
          Effect.annotateLogs({ cause: String(cause) }),
        )),
      ),
    ),
  )

  yield* project
  yield* Stream.mergeAll([
    catalog.changes.pipe(Stream.map(() => undefined)),
    installed.changes.pipe(Stream.map(() => undefined)),
    downloads.changes.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" }).pipe(
    Stream.runForEach(() => project),
    Effect.forkScoped,
  )

  return LocalModelPackages.of({
    initialized: installed.initialized,
    snapshot: mirror.get,
    changes: mirror.changes,
    installedPackageIds: installed.get.pipe(Effect.map(({ state }) =>
      new Set(state.packages.map(({ package: modelPackage }) => modelPackage.id)))),
    admitBundle: (bundle) => Effect.gen(function* () {
      const nativeBundle = yield* servableModelBundleToIcn(bundle)
      const response = yield* client.models.startModelDownload({
        payload: { bundle: nativeBundle },
      })
      yield* downloads.observeAdmission(response)
      yield* project
      return Option.match(response.download, {
        onNone: () => ({ _tag: "AlreadyInstalled" } as const),
        onSome: ({ id }) => ({
          _tag: "DownloadAdmitted" as const,
          downloadId: ModelDownloadIdSchema.make(id),
        }),
      }) satisfies BundleInstallationAdmission
    }).pipe(Effect.mapError((error) =>
      localModelPackageMutationFailure("start_model_download_failed", error))),
    cancelDownload: (downloadId) => Effect.gen(function* () {
      const cancelled = yield* client.models.cancelModelDownload({
        path: { download_id: downloadId },
      }).pipe(Effect.mapError((error) =>
        localModelPackageMutationFailure("cancel_model_download_failed", error)))
      yield* downloads.observeDownload(cancelled)
      yield* project
    }),
    acknowledgeFailure: (downloadId) => Effect.gen(function* () {
      const acknowledged = yield* client.models.acknowledgeModelDownloadFailure({
        path: { download_id: downloadId },
      }).pipe(Effect.mapError((error) => localModelPackageMutationFailure(
        "acknowledge_model_download_failure_failed",
        error,
      )))
      yield* downloads.observeDownload(acknowledged)
      yield* project
    }),
    removeBundlePackages: (bundle, retainedPackageIds = new Set()) => Effect.gen(function* () {
      const installedIds = yield* installed.get.pipe(Effect.map(({ state }) =>
        new Set(state.packages.map(({ package: modelPackage }) => modelPackage.id))))
      yield* Effect.forEach(
        servableModelBundlePackages(bundle).filter((modelPackage) =>
          installedIds.has(modelPackage.id) && !retainedPackageIds.has(modelPackage.id)),
        (modelPackage) => client.models.removeInstalledModel({
          path: { package_id: modelPackage.id },
        }).pipe(
          Effect.mapError((error) =>
            localModelPackageMutationFailure("remove_installed_model_failed", error)),
        ),
        { concurrency: 1, discard: true },
      )
      yield* installed.refresh.pipe(
        Effect.mapError((error) =>
          localModelPackageMutationFailure("refresh_installed_models_failed", error)),
      )
    }),
  })
}))
