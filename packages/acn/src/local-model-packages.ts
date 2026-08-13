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
  DownloadAttemptIdSchema,
  LocalModelMutationFailed,
  ModelPackageIdSchema,
  ModelPackagesStateSchema,
  type DownloadAttempt,
  type DownloadAttemptId,
  type LocalInferenceError,
  type ModelDownloadFailure,
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
import { RetainedModelConfigurations } from "./retained-model-configurations"
import {
  downloadAttemptFromIcn,
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
    for (const modelPackage of recommendable.configuration.bundle._tag === "Standalone"
      ? [recommendable.configuration.bundle.package]
      : [recommendable.configuration.bundle.target, recommendable.configuration.bundle.draft]) {
      packages.set(modelPackage.id, modelPackage)
    }
  }
  return [...packages.values()]
}

const latestAttempt = (
  attempts: readonly DownloadAttempt[],
  packageId: ModelPackageId,
): Option.Option<DownloadAttempt> => {
  for (let index = attempts.length - 1; index >= 0; index--) {
    const attempt = attempts[index]
    if (attempt?.packageId === packageId) return Option.some(attempt)
  }
  return Option.none()
}

export type PackageAcquisition =
  | { readonly _tag: "NotInstalled" }
  | { readonly _tag: "Downloading"; readonly attempt: Extract<DownloadAttempt, {
      readonly _tag: "Pending" | "Downloading"
    }> }
  | { readonly _tag: "Failed"; readonly attemptId: DownloadAttemptId; readonly completedBytes: number
      readonly totalBytes: number; readonly failure: ModelDownloadFailure
      readonly acknowledged: boolean }
  | { readonly _tag: "Cancelled"; readonly attemptId: DownloadAttemptId }
  | { readonly _tag: "Publishing"; readonly attemptId: DownloadAttemptId
      readonly totalBytes: number }
  | { readonly _tag: "Installed"; readonly path: string; readonly origin: ModelPackageInstallationOrigin }

export const projectPackageAcquisition = (
  acquisition: PackageAcquisition,
): ModelPackageEntry["localState"] => {
  switch (acquisition._tag) {
    case "Installed":
      return { _tag: "Installed", path: acquisition.path, origin: acquisition.origin }
    case "Downloading": {
      const { attempt } = acquisition
      return {
        _tag: "Downloading",
        attemptId: attempt.id,
        stage: attempt._tag === "Downloading" ? attempt.stage : "queued",
        completedBytes: attempt._tag === "Downloading" ? attempt.completedBytes : 0,
        totalBytes: attempt._tag === "Downloading" ? attempt.totalBytes : 0,
        bytesPerSecond: attempt._tag === "Downloading"
          ? attempt.bytesPerSecond
          : Option.none(),
      }
    }
    case "Publishing":
      return {
        _tag: "Downloading",
        attemptId: acquisition.attemptId,
        stage: "publishing",
        completedBytes: acquisition.totalBytes,
        totalBytes: acquisition.totalBytes,
        bytesPerSecond: Option.none(),
      }
    case "Failed":
      return acquisition.acknowledged
        ? { _tag: "NotInstalled" }
        : {
            _tag: "DownloadFailed",
            attemptId: acquisition.attemptId,
            completedBytes: acquisition.completedBytes,
            totalBytes: acquisition.totalBytes,
            failure: acquisition.failure,
          }
    case "Cancelled":
      return {
        _tag: "DownloadCancelled",
        attemptId: acquisition.attemptId,
      }
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
  attempts: readonly DownloadAttempt[],
): PackageAcquisition => {
  const installed = installedPackages.get(modelPackage.id)
  if (installed !== undefined) return { _tag: "Installed", ...installed }
  const current = latestAttempt(attempts, modelPackage.id)
  if (Option.isNone(current)) return { _tag: "NotInstalled" }
  const attempt = current.value
  if (attempt._tag === "Pending" || attempt._tag === "Downloading") {
    return { _tag: "Downloading", attempt }
  }
  if (attempt._tag === "Cancelled") {
    return { _tag: "Cancelled", attemptId: attempt.id }
  }
  if (attempt._tag === "Failed") {
    return {
      _tag: "Failed",
      attemptId: attempt.id,
      completedBytes: attempt.completedBytes,
      totalBytes: attempt.totalBytes,
      failure: attempt.failure,
      acknowledged: attempt.acknowledged,
    }
  }
  return {
    _tag: "Publishing",
    attemptId: attempt.id,
    totalBytes: modelPackage.files.reduce((total, file) => total + file.sizeBytes, 0),
  }
}

export interface LocalModelPackagesApi {
  readonly initialized: Effect.Effect<boolean>
  readonly snapshot: Effect.Effect<{ readonly revision: number; readonly state: ModelPackagesState }>
  readonly changes: Stream.Stream<{ readonly revision: number; readonly state: ModelPackagesState }>
  readonly installedPackageIds: Effect.Effect<ReadonlySet<string>>
  readonly admitBundle: (
    bundle: ServableModelBundle,
  ) => Effect.Effect<BundleInstallationAdmission, LocalInferenceError>
  readonly cancelAttempts: (
    attemptIds: Extract<BundleInstallationAdmission, { readonly _tag: "DownloadAdmitted" }>["attemptIds"],
  ) => Effect.Effect<void, LocalInferenceError>
  readonly acknowledgeFailures: (
    attemptIds: readonly DownloadAttemptId[],
  ) => Effect.Effect<void, LocalInferenceError>
  readonly removeBundlePackages: (
    bundle: ServableModelBundle,
    retainedPackageIds?: ReadonlySet<string>,
  ) => Effect.Effect<void, LocalInferenceError>
}

export type BundleInstallationAdmission =
  | { readonly _tag: "AlreadyInstalled" }
  | {
      readonly _tag: "DownloadAdmitted"
      readonly attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]]
    }

export class LocalModelPackages extends Context.Tag("LocalModelPackages")<
  LocalModelPackages,
  LocalModelPackagesApi
>() {}

export const LocalModelPackagesLive: Layer.Layer<
  LocalModelPackages,
  never,
  IcnCatalog | IcnClient | IcnDownloads | IcnInstalledModels | RetainedModelConfigurations
> = Layer.scoped(LocalModelPackages, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const installed = yield* IcnInstalledModels
  const downloads = yield* IcnDownloads
  const client = yield* IcnClient
  const retained = yield* RetainedModelConfigurations
  const mirror = yield* makeObservedState<ModelPackagesState>({
    inventory: { _tag: "Initializing" },
    entries: [],
  })
  const equivalent: Equivalence.Equivalence<ModelPackagesState> =
    Schema.equivalence(ModelPackagesStateSchema)

  const project = Effect.gen(function* () {
    const catalogModels = yield* Effect.forEach(
      (yield* catalog.get).state.models,
      recommendableModelFromIcn,
    )
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
    const attempts = yield* Effect.forEach(
      (yield* downloads.get).state.attempts,
      downloadAttemptFromIcn,
    )
    const catalogPackages = packagesInCatalog(catalogModels)
    const allPackages = new Map<ModelPackageId, ModelPackage>(
      catalogPackages.map((modelPackage) => [modelPackage.id, modelPackage]),
    )
    for (const configuration of yield* retained.get) {
      const referenced = configuration.bundle._tag === "Standalone"
        ? [configuration.bundle.package]
        : [
            configuration.bundle.target,
            configuration.bundle.draft,
          ]
      for (const modelPackage of referenced) {
        allPackages.set(ModelPackageIdSchema.make(modelPackage.id), modelPackage)
      }
    }
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
          attempts,
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
    }, equivalent)
  }).pipe(
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
    retained.changes.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" }).pipe(
    Stream.runForEach(() => project),
    Effect.forkScoped,
  )

  const bundlePackages = (bundle: ServableModelBundle) =>
    bundle._tag === "Standalone" ? [bundle.package] : [bundle.target, bundle.draft]

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
      yield* Effect.forEach(response.attempts, downloads.observeAttempt, { discard: true })
      yield* project
      const attemptIds = response.attempts.map((attempt) => DownloadAttemptIdSchema.make(attempt.id))
      const [first, ...rest] = attemptIds
      if (first === undefined) {
        return { _tag: "AlreadyInstalled" } satisfies BundleInstallationAdmission
      }
      return {
        _tag: "DownloadAdmitted",
        attemptIds: [first, ...rest],
      } satisfies BundleInstallationAdmission
    }).pipe(Effect.mapError((error) =>
      localModelPackageMutationFailure("start_model_download_failed", error))),
    cancelAttempts: (attemptIds) => Effect.gen(function* () {
      yield* Effect.forEach(attemptIds, (attemptId) => client.models.cancelModelDownload({
        path: { attempt_id: attemptId },
      }).pipe(
        Effect.mapError((error) =>
          localModelPackageMutationFailure("cancel_model_download_failed", error)),
      ), { concurrency: "unbounded", discard: true })
      yield* downloads.refresh.pipe(
        Effect.mapError((error) =>
          localModelPackageMutationFailure("refresh_model_downloads_failed", error)),
      )
      yield* project
    }),
    acknowledgeFailures: (attemptIds) => Effect.gen(function* () {
      const acknowledged = yield* Effect.forEach(
        attemptIds,
        (attemptId) => client.models.acknowledgeModelDownloadFailure({
          path: { attempt_id: attemptId },
        }).pipe(
          Effect.mapError((error) => localModelPackageMutationFailure(
            "acknowledge_model_download_failure_failed",
            error,
          )),
        ),
        { concurrency: "unbounded" },
      )
      yield* Effect.forEach(acknowledged, downloads.observeAttempt, { discard: true })
      yield* downloads.refresh.pipe(
        Effect.mapError((error) => localModelPackageMutationFailure(
          "refresh_model_downloads_failed",
          error,
        )),
      )
      yield* project
    }),
    removeBundlePackages: (bundle, retainedPackageIds = new Set()) => Effect.gen(function* () {
      const installedIds = yield* installed.get.pipe(Effect.map(({ state }) =>
        new Set(state.packages.map(({ package: modelPackage }) => modelPackage.id))))
      yield* Effect.forEach(
        bundlePackages(bundle).filter((modelPackage) =>
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
