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
  type ModelPackage,
  type ModelPackageEntry,
  type ModelPackageId,
  type ModelPackagesState,
  type ModelOfferingTarget,
  type ModelOfferingTargetId,
  type RecommendableModel,
  modelOfferingTargetPackageIds,
} from "@magnitudedev/protocol"
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

const mutationFailure = (operation: string, error: unknown) =>
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

export interface LocalModelPackagesApi {
  readonly snapshot: Effect.Effect<{ readonly revision: number; readonly state: ModelPackagesState }>
  readonly changes: Stream.Stream<{ readonly revision: number; readonly state: ModelPackagesState }>
  readonly installedPackageIds: Effect.Effect<ReadonlySet<string>>
  readonly downloadTarget: (
    target: ModelOfferingTarget,
  ) => Effect.Effect<void, LocalInferenceError>
  readonly cancelTargetDownload: (
    target: ModelOfferingTarget,
  ) => Effect.Effect<void, LocalInferenceError>
  readonly dismissTargetFailure: (
    target: ModelOfferingTarget,
  ) => Effect.Effect<void, LocalInferenceError>
  readonly removeTargetPackages: (
    target: ModelOfferingTarget,
    retainedPackageIds?: ReadonlySet<string>,
  ) => Effect.Effect<void, LocalInferenceError>
  readonly refresh: Effect.Effect<void>
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

    const entries: ModelPackageEntry[] = [...allPackages.values()]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((modelPackage) => {
        const installedEntry = installedById.get(modelPackage.id)
        const latest = latestAttempt(attempts, modelPackage.id)
        const localState: ModelPackageEntry["localState"] = installedEntry
          ? { _tag: "Installed", path: installedEntry.path }
          : Option.match(latest, {
              onNone: () => ({ _tag: "NotInstalled" as const }),
              onSome: (attempt): ModelPackageEntry["localState"] =>
                attempt._tag === "Pending" || attempt._tag === "Downloading"
                  ? ({
                _tag: "Downloading" as const,
                attemptId: attempt.id,
                completedBytes: attempt._tag === "Downloading" ? attempt.completedBytes : 0,
                totalBytes: attempt._tag === "Downloading" ? attempt.totalBytes : 0,
                    })
                  : ({ _tag: "NotInstalled" as const }),
            })
        return {
          package: modelPackage,
          targetId: Option.fromNullable(targetIds.get(modelPackage.id)),
          localState,
          inspection: installedEntry?.inspection ?? { _tag: "Pending" },
          lastDownloadFailure: dismissed.has(modelPackage.id)
            ? Option.none()
            : Option.flatMap(latest, (attempt) =>
                attempt._tag === "Failed"
                  ? Option.some({
                      completedBytes: attempt.completedBytes,
                      totalBytes: attempt.totalBytes,
                      failure: attempt.failure,
                    })
                  : Option.none()),
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

  const refresh = Effect.all(
    [catalog.refresh, installed.refresh, downloads.refresh],
    { concurrency: "unbounded", discard: true },
  ).pipe(Effect.ignore, Effect.andThen(project))

  const startDownload = (modelPackage: ModelPackage) => Effect.gen(function* () {
    const nativePackage = yield* modelPackageToIcn(modelPackage)
    return yield* client.models.startModelDownload({ payload: { package: nativePackage } }).pipe(
      Effect.map(({ attempt }) => DownloadAttemptIdSchema.make(attempt.id)),
      Effect.tap(() => storage.config.clearDismissedDownloadFailure(
        ModelPackageIdSchema.make(modelPackage.id),
      )),
      Effect.tap(() => downloads.refresh),
    )
  }).pipe(Effect.mapError((error) => mutationFailure("start_model_download_failed", error)))

  const targetPackages = (target: ModelOfferingTarget) =>
    target._tag === "Package" ? [target.package] : [target.target, target.draft]

  return LocalModelPackages.of({
    snapshot: mirror.get,
    changes: mirror.changes,
    installedPackageIds: installed.get.pipe(Effect.map(({ state }) =>
      new Set(state.packages.map(({ package: modelPackage }) => modelPackage.id)))),
    downloadTarget: (target) => Effect.gen(function* () {
      const entries = (yield* mirror.get).state.entries
      yield* Effect.forEach(targetPackages(target), (modelPackage) => {
        const entry = entries.find((candidate) => candidate.package.id === modelPackage.id)
        return entry?.localState._tag === "Installed" || entry?.localState._tag === "Downloading"
          ? Effect.void
          : startDownload(modelPackage).pipe(Effect.asVoid)
      }, { concurrency: "unbounded", discard: true })
    }),
    cancelTargetDownload: (target) => Effect.gen(function* () {
      const entries = (yield* mirror.get).state.entries
      const attempts = targetPackages(target).flatMap((modelPackage) => {
        const state = entries.find((entry) => entry.package.id === modelPackage.id)?.localState
        return state?._tag === "Downloading" ? [state.attemptId] : []
      })
      yield* Effect.forEach(attempts, (attemptId) => client.models.cancelModelDownload({
        path: { attempt_id: attemptId },
      }).pipe(
        Effect.mapError((error) => mutationFailure("cancel_model_download_failed", error)),
      ), { concurrency: "unbounded", discard: true })
      yield* downloads.refresh.pipe(
        Effect.mapError((error) => mutationFailure("refresh_model_downloads_failed", error)),
      )
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
    refresh,
  })
}))
