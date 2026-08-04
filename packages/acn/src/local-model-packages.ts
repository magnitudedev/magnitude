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
  ModelOfferingTargetIdSchema,
  ModelPackageIdSchema,
  ModelPackagesStateSchema,
  type DownloadAttempt,
  type DownloadAttemptId,
  type LocalInferenceError,
  type ModelDownloadAdmission,
  type ModelPackage,
  type ModelPackageEntry,
  type ModelPackageId,
  type ModelPackagesState,
  type ModelOfferingTarget,
  type ModelOfferingTargetId,
  type RecommendableModel,
  modelOfferingTargetPackageIds,
} from "@magnitudedev/acn-protocol"
import {
  IcnCatalog,
  IcnClient,
  IcnDownloads,
  IcnInstalledModels,
} from "@magnitudedev/icn"
import { MagnitudeStorage } from "@magnitudedev/storage"
import { makeObservedState } from "./mirrored-state"
import {
  downloadAttemptFromIcn,
  modelPackageFromIcn,
  modelPackageToIcn,
  packageInspectionFromIcn,
  recommendableModelFromIcn,
} from "./local-model-icn-adapter"

const mutationFailure = <Failure>(operation: string, error: Failure) =>
  new LocalModelMutationFailed({
    code: operation,
    message: error instanceof Error ? error.message : String(error),
    retryable: true,
  })

const packagesInCatalog = (
  catalog: readonly RecommendableModel[],
): readonly {
  readonly package: ModelPackage
  readonly targetId: Option.Option<ModelOfferingTargetId>
}[] => {
  const packages = new Map<
    ModelPackageId,
    { readonly package: ModelPackage; readonly targetId: Option.Option<ModelOfferingTargetId> }
  >()
  for (const recommendable of catalog) {
    for (const modelPackage of recommendable.target._tag === "Package"
      ? [recommendable.target.package]
      : [recommendable.target.target, recommendable.target.draft]) {
      packages.set(modelPackage.id, {
        package: modelPackage,
        targetId: recommendable.target._tag === "Package"
          ? Option.some(recommendable.targetId)
          : Option.none(),
      })
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

type PackageAcquisition =
  | { readonly _tag: "NotInstalled" }
  | { readonly _tag: "Downloading"; readonly attempt: Extract<DownloadAttempt, {
      readonly _tag: "Pending" | "Downloading"
    }> }
  | { readonly _tag: "Failed"; readonly attemptId: DownloadAttemptId; readonly completedBytes: number
      readonly totalBytes: number; readonly failure: LocalModelMutationFailed }
  | { readonly _tag: "Cancelled"; readonly attemptId: DownloadAttemptId }
  | { readonly _tag: "Installed"; readonly path: string }

const packageAcquisition = (
  modelPackage: ModelPackage,
  installedPackages: ReadonlyMap<ModelPackageId, string>,
  attempts: readonly DownloadAttempt[],
): PackageAcquisition => {
  const installedPath = installedPackages.get(modelPackage.id)
  if (installedPath !== undefined) return { _tag: "Installed", path: installedPath }
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
      failure: new LocalModelMutationFailed(attempt.failure),
    }
  }
  return { _tag: "NotInstalled" }
}

export interface LocalModelPackagesApi {
  readonly snapshot: Effect.Effect<{ readonly revision: number; readonly state: ModelPackagesState }>
  readonly changes: Stream.Stream<{ readonly revision: number; readonly state: ModelPackagesState }>
  readonly installedPackageIds: Effect.Effect<ReadonlySet<string>>
  readonly admitTarget: (
    target: ModelOfferingTarget,
  ) => Effect.Effect<ModelDownloadAdmission["attemptIds"], LocalInferenceError>
  readonly cancelAttempts: (
    attemptIds: ModelDownloadAdmission["attemptIds"],
  ) => Effect.Effect<void, LocalInferenceError>
  readonly dismissTargetFailure: (
    target: ModelOfferingTarget,
  ) => Effect.Effect<void, LocalInferenceError>
  readonly removeTargetPackages: (
    target: ModelOfferingTarget,
    retainedPackageIds?: ReadonlySet<string>,
  ) => Effect.Effect<void, LocalInferenceError>
}

export class LocalModelPackages extends Context.Tag("LocalModelPackages")<
  LocalModelPackages,
  LocalModelPackagesApi
>() {}

export const LocalModelPackagesLive: Layer.Layer<
  LocalModelPackages,
  never,
  IcnCatalog | IcnClient | IcnDownloads | IcnInstalledModels | MagnitudeStorage
> = Layer.scoped(LocalModelPackages, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const installed = yield* IcnInstalledModels
  const downloads = yield* IcnDownloads
  const client = yield* IcnClient
  const storage = yield* MagnitudeStorage
  const mirror = yield* makeObservedState<ModelPackagesState>({ entries: [] })
  const equivalent: Equivalence.Equivalence<ModelPackagesState> =
    Schema.equivalence(ModelPackagesStateSchema)

  const project = Effect.gen(function* () {
    const catalogModels = yield* Effect.forEach(
      (yield* catalog.get).state.models,
      recommendableModelFromIcn,
    )
    const installedModels = yield* Effect.forEach(
      (yield* installed.get).state.packages,
      (entry) => Effect.all({
        targetId: Effect.succeed(ModelOfferingTargetIdSchema.make(String(entry.targetId))),
        package: modelPackageFromIcn(entry.package),
        path: Effect.succeed(entry.path),
        inspection: packageInspectionFromIcn(entry.inspection),
      }),
    )
    const attempts = yield* Effect.forEach(
      (yield* downloads.get).state.attempts,
      downloadAttemptFromIcn,
    )
    const config = yield* storage.config.load()
    const dismissed = new Set(config.models?.dismissedDownloadFailures ?? [])
    const catalogPackages = packagesInCatalog(catalogModels)
    const allPackages = new Map<ModelPackageId, ModelPackage>(
      catalogPackages.map(({ package: modelPackage }) => [modelPackage.id, modelPackage]),
    )
    const targetIds = new Map(catalogPackages.flatMap(({ package: modelPackage, targetId }) =>
      Option.match(targetId, {
        onNone: () => [],
        onSome: (id) => [[modelPackage.id, id] as const],
      })))
    for (const offering of config.models?.localProviderOfferings ?? []) {
      const referenced = offering.configuration.target._tag === "Package"
        ? [offering.configuration.target.package]
        : [
            offering.configuration.target.target,
            offering.configuration.target.draft,
          ]
      for (const modelPackage of referenced) {
        allPackages.set(ModelPackageIdSchema.make(modelPackage.id), modelPackage)
      }
    }
    for (const item of installedModels) {
      allPackages.set(item.package.id, item.package)
      targetIds.set(item.package.id, item.targetId)
    }
    const installedById = new Map(installedModels.map((item) => [item.package.id, item]))
    const installedPaths = new Map(installedModels.map((item) => [item.package.id, item.path]))

    const entries: ModelPackageEntry[] = [...allPackages.values()]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((modelPackage) => {
        const installedEntry = installedById.get(modelPackage.id)
        const acquisition = packageAcquisition(
          modelPackage,
          installedPaths,
          attempts,
        )
        const localState: ModelPackageEntry["localState"] = (() => {
          switch (acquisition._tag) {
            case "Installed":
              return { _tag: "Installed", path: acquisition.path }
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
            case "Failed":
              return dismissed.has(modelPackage.id)
                ? { _tag: "NotInstalled" }
                : {
                    _tag: "DownloadFailed",
                    attemptId: acquisition.attemptId,
                    completedBytes: acquisition.completedBytes,
                    totalBytes: acquisition.totalBytes,
                    failure: {
                      code: acquisition.failure.code,
                      message: acquisition.failure.message,
                      retryable: acquisition.failure.retryable,
                    },
                  }
            case "Cancelled":
              return {
                _tag: "DownloadCancelled",
                attemptId: acquisition.attemptId,
              }
            case "NotInstalled":
              return { _tag: "NotInstalled" }
          }
        })()
        return {
          package: modelPackage,
          targetId: Option.fromNullable(targetIds.get(modelPackage.id)),
          localState,
          inspection: installedEntry?.inspection ?? { _tag: "Pending" },
        }
      })
    yield* mirror.setIfChanged({ entries }, equivalent)
  }).pipe(
    Effect.catchAllCause((cause) =>
      Effect.logWarning("Unable to project local model packages").pipe(
        Effect.annotateLogs({ cause: String(cause) }),
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

  const startDownload = (modelPackage: ModelPackage) => Effect.gen(function* () {
    const nativePackage = yield* modelPackageToIcn(modelPackage)
    const { attempt } = yield* client.models.startModelDownload({
      payload: { package: nativePackage },
    })
    yield* downloads.observeAttempt(attempt)
    yield* storage.config.clearDismissedDownloadFailure(
      ModelPackageIdSchema.make(modelPackage.id),
    )
    return DownloadAttemptIdSchema.make(attempt.id)
  }).pipe(Effect.mapError((error) => mutationFailure("start_model_download_failed", error)))

  const targetPackages = (target: ModelOfferingTarget) =>
    target._tag === "Package" ? [target.package] : [target.target, target.draft]

  return LocalModelPackages.of({
    snapshot: mirror.get,
    changes: mirror.changes,
    installedPackageIds: installed.get.pipe(Effect.map(({ state }) =>
      new Set(state.packages.map(({ package: modelPackage }) => modelPackage.id)))),
    admitTarget: (target) => Effect.gen(function* () {
      const attemptIds = yield* Effect.forEach(
        targetPackages(target),
        startDownload,
        { concurrency: "unbounded" },
      )
      yield* project
      const [first, ...rest] = attemptIds
      if (first === undefined) {
        return yield* Effect.die("A model target must contain at least one package")
      }
      return [first, ...rest]
    }),
    cancelAttempts: (attemptIds) => Effect.gen(function* () {
      yield* Effect.forEach(attemptIds, (attemptId) => client.models.cancelModelDownload({
        path: { attempt_id: attemptId },
      }).pipe(
        Effect.mapError((error) => mutationFailure("cancel_model_download_failed", error)),
      ), { concurrency: "unbounded", discard: true })
      yield* downloads.refresh.pipe(
        Effect.mapError((error) => mutationFailure("refresh_model_downloads_failed", error)),
      )
      yield* project
    }),
    dismissTargetFailure: (target) => Effect.forEach(
      targetPackages(target),
      (modelPackage) => storage.config.dismissDownloadFailure(modelPackage.id),
      { concurrency: "unbounded", discard: true },
    ).pipe(
      Effect.tap(() => project),
      Effect.mapError((error) => mutationFailure("dismiss_model_download_failure_failed", error)),
    ),
    removeTargetPackages: (target, retainedPackageIds = new Set()) => Effect.gen(function* () {
      const installedIds = yield* installed.get.pipe(Effect.map(({ state }) =>
        new Set(state.packages.map(({ package: modelPackage }) => modelPackage.id))))
      yield* Effect.forEach(
        targetPackages(target).filter((modelPackage) =>
          installedIds.has(modelPackage.id) && !retainedPackageIds.has(modelPackage.id)),
        (modelPackage) => client.models.removeInstalledModel({
          path: { package_id: modelPackage.id },
        }).pipe(
          Effect.mapError((error) => mutationFailure("remove_installed_model_failed", error)),
        ),
        { concurrency: 1, discard: true },
      )
      yield* installed.refresh.pipe(
        Effect.mapError((error) => mutationFailure("refresh_installed_models_failed", error)),
      )
    }),
  })
}))
