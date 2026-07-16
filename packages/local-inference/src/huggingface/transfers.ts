import { createHash, randomUUID } from "node:crypto"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import { Effect, Fiber, Option, Ref, Schema, Scope, Stream, SubscriptionRef } from "effect"
import { ModelArtifactKey, ModelOriginRepositoryId, ModelOriginRevisionId, SourceFileKey } from "../model-files"
import { normalizeFileSystemFailure, sha256File } from "../model-files/platform"
import { DownloadableArtifactId, HuggingFaceRevision, ModelTransferId, type ModelTransferId as TransferId } from "./identity"
import {
  ModelTransferStatus,
  TransferExecutionError,
  TransferNotFound,
  TransferPlanningError,
  TransferRegistryError,
  TransferStateError,
  type ModelTransferRegistryApi,
  type ModelTransferRegistryOptions,
  type ModelTransferSnapshot,
  type TransferFailureOperation,
  type TransferFailureReason,
  type VerifiedTransferFile,
  type VerifiedTransferPlan,
} from "./transfer-contracts"
import { PersistedTransferJson, SafeRelativePath } from "./transfer-schema"

export * from "./transfer-contracts"

interface ActiveTransfer { readonly plan: VerifiedTransferPlan; readonly snapshot: SubscriptionRef.SubscriptionRef<ModelTransferSnapshot>; readonly fiber: Ref.Ref<Option.Option<Fiber.RuntimeFiber<void, never>>> }

export const makeModelTransferRegistry = (options: ModelTransferRegistryOptions): Effect.Effect<ModelTransferRegistryApi, never, FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | Scope.Scope> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const http = yield* HttpClient.HttpClient
  const scope = yield* Scope.Scope
  const stagingRoot = path.resolve(options.stagingRoot)
  const stateRoot = path.resolve(Option.getOrElse(options.stateRoot, () => path.join(stagingRoot, ".state")))
  const transfers = yield* Ref.make<ReadonlyMap<TransferId, ActiveTransfer>>(new Map())
  const diagnostics = yield* Ref.make<readonly TransferRegistryError[]>([])
  const persistenceLock = yield* Effect.makeSemaphore(1)
  const lifecycleLock = yield* Effect.makeSemaphore(1)
  yield* fs.makeDirectory(stateRoot, { recursive: true }).pipe(Effect.catchAll((error) => Ref.update(diagnostics, (current) => [...current, new TransferRegistryError({ operation: "restore", reason: normalizeFileSystemFailure(error), path: stateRoot })])))

  const persist = (active: ActiveTransfer): Effect.Effect<void, TransferRegistryError> => persistenceLock.withPermits(1)(Effect.gen(function* () {
    const snapshot = yield* SubscriptionRef.get(active.snapshot)
    const encoded = yield* Schema.encode(PersistedTransferJson)({ version: 1, plan: active.plan, snapshot }).pipe(Effect.mapError(() => new TransferRegistryError({ operation: "persist", reason: "invalid-record", path: stateRoot })))
    const target = path.join(stateRoot, `${snapshot.id}.json`)
    const temporary = `${target}.${randomUUID()}.tmp`
    yield* Effect.acquireUseRelease(
      fs.writeFileString(temporary, encoded, { mode: 0o600 }).pipe(Effect.mapError((error) => new TransferRegistryError({ operation: "persist", reason: normalizeFileSystemFailure(error), path: temporary }))),
      () => fs.rename(temporary, target).pipe(Effect.mapError((error) => new TransferRegistryError({ operation: "persist", reason: normalizeFileSystemFailure(error), path: target }))),
      () => fs.remove(temporary, { force: true }).pipe(Effect.ignore),
    )
  }))
  const persistObserved = (active: ActiveTransfer) => persist(active).pipe(Effect.tapError((error) => Ref.update(diagnostics, (current) => [...current, error])))
  const setSnapshot = (active: ActiveTransfer, snapshot: ModelTransferSnapshot) => SubscriptionRef.set(active.snapshot, snapshot).pipe(Effect.zipRight(persistObserved(active)))
  const transition = (
    active: ActiveTransfer,
    status: ModelTransferStatus,
    completedBytes: number,
  ) => SubscriptionRef.get(active.snapshot).pipe(Effect.flatMap((current) => setSnapshot(active, {
    id: current.id,
    artifactId: current.artifactId,
    totalBytes: current.totalBytes,
    status,
    completedBytes,
  })))
  const executionFailure = (id: TransferId, operation: TransferFailureOperation, reason: TransferFailureReason, pathValue: Option.Option<string> = Option.none(), status: Option.Option<number> = Option.none()) => new TransferExecutionError({ transferId: id, operation, reason, path: pathValue, status })
  const recordProgress = (active: ActiveTransfer, completedBytes: number) => SubscriptionRef.update(active.snapshot, (snapshot) => ({
    id: snapshot.id,
    artifactId: snapshot.artifactId,
    status: snapshot.status,
    totalBytes: snapshot.totalBytes,
    completedBytes,
  }))

  const execute = (active: ActiveTransfer): Effect.Effect<void, never> => Effect.gen(function* () {
    const initial = yield* SubscriptionRef.get(active.snapshot)
    const id = initial.id
    yield* transition(active, ModelTransferStatus.CheckingSpace(), initial.completedBytes).pipe(Effect.mapError(() => executionFailure(id, "persist", "persistence-failed")))
    yield* fs.makeDirectory(path.join(stagingRoot, id), { recursive: true }).pipe(Effect.mapError((error) => executionFailure(id, "download", normalizeFileSystemFailure(error), Option.some(stagingRoot))))
    let completedBytes = 0
    for (const file of active.plan.files) {
      const candidate = path.resolve(stagingRoot, id, file.path)
      const root = path.resolve(stagingRoot, id)
      const relative = path.relative(root, candidate)
      if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) return yield* executionFailure(id, "download", "unsafe-path", Option.some(file.path))
      const existing = yield* fs.stat(candidate).pipe(Effect.option)
      if (Option.isSome(existing) && existing.value.type === "File") completedBytes += Math.min(Number(existing.value.size), file.sizeBytes)
    }
    const available = yield* options.capacity.availableBytes(stagingRoot).pipe(Effect.mapError(() => executionFailure(id, "space", "capacity-unavailable")))
    if (available < active.plan.totalBytes - completedBytes + options.reserveBytes) return yield* executionFailure(id, "space", "insufficient-space")
    for (const file of active.plan.files) {
      const target = path.resolve(stagingRoot, id, file.path)
      yield* fs.makeDirectory(path.dirname(target), { recursive: true }).pipe(Effect.mapError((error) => executionFailure(id, "download", normalizeFileSystemFailure(error), Option.some(file.path))))
      const existing = yield* fs.stat(target).pipe(Effect.option)
      let offset = Option.isSome(existing) && existing.value.type === "File" ? Number(existing.value.size) : 0
      if (offset > file.sizeBytes) { yield* fs.truncate(target, 0).pipe(Effect.mapError((error) => executionFailure(id, "download", normalizeFileSystemFailure(error), Option.some(file.path)))); offset = 0 }
      yield* transition(active, ModelTransferStatus.Downloading({ currentFile: file.path }), completedBytes).pipe(Effect.mapError(() => executionFailure(id, "persist", "persistence-failed")))
      if (offset < file.sizeBytes) {
        let request = HttpClientRequest.get(options.hub.downloadUrl(active.plan.repository, active.plan.commit, file.path).toString())
        if (offset > 0) request = HttpClientRequest.setHeader(request, "range", `bytes=${offset}-`)
        const response = yield* http.execute(request).pipe(Effect.mapError(() => executionFailure(id, "download", "transport", Option.some(file.path))))
        if (response.status < 200 || response.status >= 300) return yield* executionFailure(id, "download", "http-rejected", Option.some(file.path), Option.some(response.status))
        if (offset > 0 && response.status !== 206) { yield* fs.truncate(target, 0).pipe(Effect.mapError((error) => executionFailure(id, "download", normalizeFileSystemFailure(error), Option.some(file.path)))); completedBytes -= offset; offset = 0 }
        yield* response.stream.pipe(
          Stream.mapError(() => executionFailure(id, "download", "transport", Option.some(file.path))),
          Stream.tap((bytes) => {
            completedBytes += bytes.byteLength
            return recordProgress(active, completedBytes)
          }),
          Stream.run(fs.sink(target, { flag: offset > 0 ? "a" : "w" })),
          Effect.mapError((error) => error._tag === "TransferExecutionError" ? error : executionFailure(id, "download", normalizeFileSystemFailure(error), Option.some(file.path))),
        )
      }
      const info = yield* fs.stat(target).pipe(Effect.mapError((error) => executionFailure(id, "verify", normalizeFileSystemFailure(error), Option.some(file.path))))
      if (Number(info.size) !== file.sizeBytes) return yield* executionFailure(id, "verify", "size-mismatch", Option.some(file.path))
      yield* transition(active, ModelTransferStatus.Verifying({ currentFile: file.path }), completedBytes).pipe(Effect.mapError(() => executionFailure(id, "persist", "persistence-failed")))
      const digest = yield* sha256File(target).pipe(Effect.provideService(FileSystem.FileSystem, fs), Effect.mapError((error) => executionFailure(id, "verify", normalizeFileSystemFailure(error), Option.some(file.path))))
      if (digest !== file.sha256) { yield* fs.truncate(target, 0).pipe(Effect.ignore); return yield* executionFailure(id, "verify", "digest-mismatch", Option.some(file.path)) }
    }
    yield* transition(active, ModelTransferStatus.Publishing(), completedBytes).pipe(Effect.mapError(() => executionFailure(id, "persist", "persistence-failed")))
    const transferRoot = path.resolve(stagingRoot, id)
    const artifactKey = yield* Schema.decodeUnknown(ModelArtifactKey)(active.plan.artifactId).pipe(Effect.mapError(() => executionFailure(id, "publish", "destination-rejected")))
    const publicationFiles = yield* Effect.forEach(active.plan.files, (file) => Schema.decodeUnknown(SourceFileKey)(file.path).pipe(Effect.map((key) => ({ key, stagedPath: path.join(transferRoot, file.path), publishedRelativePath: file.path, role: file.role, shardIndex: file.shardIndex, sizeBytes: file.sizeBytes, sha256: file.sha256 })), Effect.mapError(() => executionFailure(id, "publish", "destination-rejected", Option.some(file.path)))))
    const publicationRelationships = yield* Effect.forEach(active.plan.relationships, (relationship) => Effect.all({ from: Schema.decodeUnknown(SourceFileKey)(relationship.fromPath), to: Schema.decodeUnknown(SourceFileKey)(relationship.toPath) }).pipe(Effect.map(({ from, to }) => ({ kind: "projector-for" as const, from, to })), Effect.mapError(() => executionFailure(id, "publish", "destination-rejected"))))
    const origin = yield* Effect.all({ repository: Schema.decodeUnknown(ModelOriginRepositoryId)(active.plan.repository), revision: Schema.decodeUnknown(ModelOriginRevisionId)(active.plan.commit) }).pipe(Effect.mapError(() => executionFailure(id, "publish", "destination-rejected")))
    const modelFileId = yield* options.destination.publish({ artifactKey, files: publicationFiles, relationships: publicationRelationships, origin: Option.some({ kind: "huggingface", ...origin }) }).pipe(Effect.mapError(() => executionFailure(id, "publish", "destination-rejected")))
    yield* transition(active, ModelTransferStatus.Ready({ modelFileId }), active.plan.totalBytes).pipe(Effect.mapError(() => executionFailure(id, "persist", "persistence-failed")))
    yield* fs.remove(transferRoot, { recursive: true, force: true }).pipe(Effect.ignore)
  }).pipe(
    Effect.catchTag("TransferExecutionError", (failure) => SubscriptionRef.get(active.snapshot).pipe(Effect.flatMap((snapshot) => transition(active, ModelTransferStatus.Failed({ failure: { operation: failure.operation, reason: failure.reason, path: failure.path, status: failure.status } }), snapshot.completedBytes)), Effect.ignore)),
    Effect.onInterrupt(() => SubscriptionRef.get(active.snapshot).pipe(Effect.flatMap((snapshot) => transition(active, ModelTransferStatus.Cancelled(), snapshot.completedBytes)), Effect.ignore)),
    Effect.ensuring(Ref.set(active.fiber, Option.none())),
  )
  const launch = (active: ActiveTransfer) => execute(active).pipe(Effect.forkScoped, Effect.provideService(Scope.Scope, scope), Effect.tap((fiber) => Ref.set(active.fiber, Option.some(fiber))), Effect.asVoid)
  const makeActive = (plan: VerifiedTransferPlan, snapshot: ModelTransferSnapshot) => Effect.gen(function* () { return { plan, snapshot: yield* SubscriptionRef.make(snapshot), fiber: yield* Ref.make<Option.Option<Fiber.RuntimeFiber<void, never>>>(Option.none()) } })

  const stateDirectory = yield* fs.readDirectory(stateRoot).pipe(Effect.either)
  const stateFiles = stateDirectory._tag === "Right" ? stateDirectory.right : []
  if (stateDirectory._tag === "Left") yield* Ref.update(diagnostics, (current) => [...current, new TransferRegistryError({ operation: "restore", reason: normalizeFileSystemFailure(stateDirectory.left), path: stateRoot })])
  for (const name of stateFiles.filter((value) => value.endsWith(".json")).sort()) {
    const recordPath = path.join(stateRoot, name)
    const decoded = yield* fs.readFileString(recordPath).pipe(Effect.flatMap(Schema.decode(PersistedTransferJson)), Effect.either)
    if (decoded._tag === "Left") { yield* Ref.update(diagnostics, (current) => [...current, new TransferRegistryError({ operation: "restore", reason: "invalid-record", path: recordPath })]); continue }
    const interrupted = ["CheckingSpace", "Downloading", "Verifying", "Publishing"].includes(decoded.right.snapshot.status._tag)
    const snapshot: ModelTransferSnapshot = interrupted
      ? { id: decoded.right.snapshot.id, artifactId: decoded.right.snapshot.artifactId, status: ModelTransferStatus.Paused(), completedBytes: decoded.right.snapshot.completedBytes, totalBytes: decoded.right.snapshot.totalBytes }
      : decoded.right.snapshot
    const active = yield* makeActive(decoded.right.plan, snapshot)
    yield* Ref.update(transfers, (current) => new Map(current).set(snapshot.id, active))
  }

  const get = (id: TransferId) => Ref.get(transfers).pipe(
    Effect.flatMap((current) => Option.match(Option.fromNullable(current.get(id)), {
      onNone: () => Effect.fail(new TransferNotFound({ id })),
      onSome: Effect.succeed,
    })),
  )
  return {
    plan: (request) => Effect.gen(function* () {
      if (request.files.length === 0) return yield* new TransferPlanningError({ operation: "validate", reason: "empty-files", repository: request.repository, path: Option.none() })
      if (new Set(request.files.map(({ path }) => path)).size !== request.files.length) return yield* new TransferPlanningError({ operation: "validate", reason: "duplicate-file", repository: request.repository, path: Option.none() })
      for (const file of request.files) if (Option.isNone(yield* Schema.decodeUnknown(SafeRelativePath)(file.path).pipe(Effect.option))) return yield* new TransferPlanningError({ operation: "validate", reason: "unsafe-path", repository: request.repository, path: Option.some(file.path) })
      if (request.relationships.some((relationship) => !request.files.some(({ path }) => path === relationship.fromPath) || !request.files.some(({ path }) => path === relationship.toPath))) return yield* new TransferPlanningError({ operation: "validate", reason: "invalid-relationship", repository: request.repository, path: Option.none() })
      const resolved = yield* options.hub.resolveRevision(request.repository, request.revision).pipe(Effect.mapError(() => new TransferPlanningError({ operation: "resolve-revision", reason: "revision-unavailable", repository: request.repository, path: Option.none() })))
      const pinnedRevision = yield* Schema.decodeUnknown(HuggingFaceRevision)(resolved.commit).pipe(Effect.mapError(() => new TransferPlanningError({ operation: "list-files", reason: "listing-unavailable", repository: request.repository, path: Option.none() })))
      const listing = yield* options.hub.listFiles(request.repository, pinnedRevision).pipe(Effect.mapError(() => new TransferPlanningError({ operation: "list-files", reason: "listing-unavailable", repository: request.repository, path: Option.none() })))
      const files: VerifiedTransferFile[] = []
      for (const requested of request.files) {
        const remote = Option.fromNullable(listing.find((entry) => entry.type === "file" && entry.path === requested.path))
        if (Option.isNone(remote)) return yield* new TransferPlanningError({ operation: "validate", reason: "file-missing", repository: request.repository, path: Option.some(requested.path) })
        if (Option.isNone(remote.value.lfs)) return yield* new TransferPlanningError({ operation: "validate", reason: "digest-unavailable", repository: request.repository, path: Option.some(requested.path) })
        files.push({ path: requested.path, role: requested.role, shardIndex: requested.shardIndex, sizeBytes: remote.value.lfs.value.sizeBytes, sha256: remote.value.lfs.value.sha256 })
      }
      const artifactId = DownloadableArtifactId.make(`${request.repository}@${resolved.commit}:${createHash("sha256").update(files.map((file) => `${file.path}:${file.sha256}`).join("\0")).digest("hex")}`)
      return { artifactId, repository: request.repository, commit: resolved.commit, files, relationships: request.relationships, totalBytes: files.reduce((sum, file) => sum + file.sizeBytes, 0) }
    }),
    start: (plan) => lifecycleLock.withPermits(1)(Effect.gen(function* () {
      const current = yield* Ref.get(transfers)
      const duplicate = Option.fromNullable([...current.values()].find((active) => active.plan.artifactId === plan.artifactId))
      if (Option.isSome(duplicate)) return (yield* SubscriptionRef.get(duplicate.value.snapshot)).id
      const id = ModelTransferId.make(randomUUID())
      const active = yield* makeActive(plan, { id, artifactId: plan.artifactId, status: ModelTransferStatus.Planned(), completedBytes: 0, totalBytes: plan.totalBytes })
      yield* Ref.update(transfers, (entries) => new Map(entries).set(id, active))
      yield* persistObserved(active).pipe(Effect.tapError(() => Ref.update(transfers, (entries) => { const next = new Map(entries); next.delete(id); return next })))
      yield* launch(active)
      return id
    })),
    observe: (id) => Stream.unwrap(get(id).pipe(Effect.map((active) => active.snapshot.changes))),
    list: Ref.get(transfers).pipe(Effect.flatMap((current) => Effect.forEach([...current.values()], ({ snapshot }) => SubscriptionRef.get(snapshot)))),
    cancel: (id) => lifecycleLock.withPermits(1)(Effect.gen(function* () {
      const active = yield* get(id)
      const snapshot = yield* SubscriptionRef.get(active.snapshot)
      if (["Ready", "Cancelled"].includes(snapshot.status._tag)) return yield* new TransferStateError({ id, operation: "cancel", state: snapshot.status._tag })
      const fiber = yield* Ref.get(active.fiber)
      if (Option.isSome(fiber)) yield* Fiber.interrupt(fiber.value)
      else yield* transition(active, ModelTransferStatus.Cancelled(), snapshot.completedBytes).pipe(Effect.mapError(() => new TransferStateError({ id, operation: "cancel", state: snapshot.status._tag })))
    })),
    resume: (id) => lifecycleLock.withPermits(1)(Effect.gen(function* () {
      const active = yield* get(id)
      const snapshot = yield* SubscriptionRef.get(active.snapshot)
      if (!["Failed", "Cancelled", "Paused"].includes(snapshot.status._tag)) return yield* new TransferStateError({ id, operation: "resume", state: snapshot.status._tag })
      yield* launch(active)
    })),
    recoveryDiagnostics: Ref.get(diagnostics),
  }
})
