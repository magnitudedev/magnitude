import { createHash } from "node:crypto"
import * as FileSystem from "@effect/platform/FileSystem"
import { Chunk, Effect, Option, Stream, SynchronizedRef } from "effect"
import { makeContentId, makeModelFileId, makeModelFilePartId, type ModelFileId } from "./identity"
import type { InspectedModelArtifact, ModelFileDiscoveryRefresh, ModelFileFormat, ModelFileRecord, ModelFileRegistryApi, ModelFileSnapshot, ModelFileSourceRegistration, ResolvedModelFiles, SourceDiscoveryIssue, SourceFileSet } from "./types"
import { ModelFileDeleteError, ModelFileNotFound, ModelFileResolveError } from "./types"

const resolveError = (id: ModelFileId, reason: ModelFileResolveError["reason"], part: Option.Option<SourceFileSet["entries"][number]["key"]> = Option.none()) => new ModelFileResolveError({ id, reason, part })

interface RegistryEntry {
  readonly record: ModelFileRecord
  readonly registration: ModelFileSourceRegistration
  readonly setId: SourceFileSet["id"]
  readonly artifact: InspectedModelArtifact
}
interface CachedInspection { readonly version: string; readonly artifacts: readonly InspectedModelArtifact[] }
interface RegistryState {
  readonly snapshot: Option.Option<ModelFileSnapshot>
  readonly entries: ReadonlyMap<ModelFileId, RegistryEntry>
  readonly formatCache: ReadonlyMap<string, CachedInspection>
}
export interface ModelFileRegistryOptions { readonly sources: readonly ModelFileSourceRegistration[]; readonly formats: readonly ModelFileFormat[] }

const setVersion = (set: SourceFileSet): string => {
  const hash = createHash("sha256")
  for (const entry of set.entries) hash.update(`${entry.key.length}:${entry.key}${entry.sizeBytes}:${Option.getOrElse(entry.modifiedAtMillis, () => -1)}:${Option.getOrElse(entry.sha256, () => "")}`)
  return hash.digest("hex")
}

const buildRecord = (registration: ModelFileSourceRegistration, format: ModelFileFormat, artifact: InspectedModelArtifact): ModelFileRecord => {
  const source = registration.source
  const digests = artifact.parts.flatMap(({ entry }) => Option.toArray(entry.sha256))
  return {
    id: makeModelFileId(source.id, artifact.key), sourceId: source.id,
    contentId: digests.length === artifact.parts.length ? makeContentId(digests) : Option.none(),
    displayName: artifact.displayName, format: format.id,
    sizeBytes: artifact.parts.reduce((sum, { entry }) => sum + entry.sizeBytes, 0),
    files: artifact.parts.map(({ entry, role }) => ({ id: makeModelFilePartId(source.id, entry.key), role, sizeBytes: entry.sizeBytes, sha256: entry.sha256 })),
    metadata: artifact.metadata, ownership: source.ownership, operations: { delete: registration._tag === "Deletable" }, warnings: artifact.warnings,
  }
}

export const makeModelFileRegistry = (options: ModelFileRegistryOptions): Effect.Effect<ModelFileRegistryApi, never, FileSystem.FileSystem> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const state = yield* SynchronizedRef.make<RegistryState>({
    snapshot: Option.none(),
    entries: new Map(),
    formatCache: new Map(),
  })

  const inspectFresh = (
    refresh: ModelFileDiscoveryRefresh,
  ): Effect.Effect<ModelFileSnapshot> => SynchronizedRef.modifyEffect(state, (previous) => Effect.gen(function* () {
    const previousCache = refresh === "full" ? new Map<string, CachedInspection>() : previous.formatCache
    const cache = new Map<string, CachedInspection>()
    const entries = new Map<ModelFileId, RegistryEntry>()
    const issues: SourceDiscoveryIssue[] = []
    for (const registration of options.sources) {
      const source = registration.source
      const discovered = yield* Stream.runCollect(source.discover({ refresh })).pipe(Effect.either)
      if (discovered._tag === "Left") {
        issues.push({ sourceId: source.id, code: "unreadable", message: `${discovered.left.operation}: ${discovered.left.reason}`, sourceKey: Option.none() })
        continue
      }
      for (const event of Chunk.toReadonlyArray(discovered.right)) {
        if (event._tag === "Issue") { issues.push(event.issue); continue }
        for (const format of options.formats) {
          const cacheKey = `${source.id}\0${event.set.id}\0${format.id}`
          const version = setVersion(event.set)
          const cached = Option.fromNullable(previousCache.get(cacheKey))
          const inspected = Option.match(cached, {
            onNone: () => format.inspect(event.set),
            onSome: (entry) => entry.version === version ? Effect.succeed(entry.artifacts) : format.inspect(event.set),
          })
          const result = yield* inspected.pipe(Effect.either)
          if (result._tag === "Left") {
            issues.push({ sourceId: source.id, code: "unreadable", message: `${result.left.operation}: ${result.left.reason}`, sourceKey: Option.some(result.left.file) })
            continue
          }
          cache.set(cacheKey, { version, artifacts: result.right })
          for (const artifact of result.right) {
            const record = buildRecord(registration, format, artifact)
            entries.set(record.id, { record, registration, setId: event.set.id, artifact })
          }
        }
      }
    }
    const snapshot: ModelFileSnapshot = {
      records: [...entries.values()].map(({ record }) => record).sort((left, right) => left.displayName.localeCompare(right.displayName) || String(left.id).localeCompare(String(right.id))),
      issues, capturedAt: new Date(),
    }
    const next: RegistryState = {
      snapshot: Option.some(snapshot),
      entries,
      formatCache: cache,
    }
    return [snapshot, next] as const
  }))

  return {
    inspect: (refresh) => refresh === "cached"
      ? SynchronizedRef.get(state).pipe(Effect.flatMap((current) => Option.match(current.snapshot, { onNone: () => inspectFresh("changed"), onSome: Effect.succeed })))
      : inspectFresh(refresh),
    get: (id) => SynchronizedRef.get(state).pipe(Effect.flatMap((current) => Option.match(
      Option.fromNullable(current.entries.get(id)),
      { onNone: () => Effect.fail(new ModelFileNotFound({ id })), onSome: ({ record }) => Effect.succeed(record) },
    ))),
    resolve: (id) => Effect.gen(function* () {
      const registered = yield* Option.match(Option.fromNullable((yield* SynchronizedRef.get(state)).entries.get(id)), {
        onNone: () => Effect.fail(resolveError(id, "not-found")),
        onSome: Effect.succeed,
      })
      const current = yield* registered.registration.source.resolve(registered.setId).pipe(Effect.mapError(() => resolveError(id, "source-unavailable")))
      const filesByKey = new Map(current.set.entries.map((entry) => [entry.key, entry]))
      const resolvedParts = yield* Effect.forEach(registered.artifact.parts, (part) => Option.match(
        Option.fromNullable(filesByKey.get(part.entry.key)),
        { onNone: () => Effect.fail(resolveError(id, "part-missing", Option.some(part.entry.key))), onSome: (file) => Effect.succeed({ part, file }) },
      ))
      for (const { file, part: artifactPart } of resolvedParts) {
        const actual = yield* fs.stat(file.path).pipe(Effect.mapError(() => resolveError(id, "unreadable", Option.some(file.key))))
        const expected = artifactPart.entry
        const modified = Option.map(actual.mtime, (mtime) => mtime.getTime())
        const timestampChanged = Option.exists(expected.modifiedAtMillis, (expectedMillis) => !Option.contains(modified, expectedMillis))
        if (Number(actual.size) !== expected.sizeBytes || timestampChanged) return yield* resolveError(id, "changed", Option.some(expected.key))
      }
      const paths = resolvedParts.map(({ part, file }) => ({ part, path: file.path }))
      const primary = yield* Option.match(Option.orElse(
        Option.fromNullable(paths.find(({ part }) => part.role === "primary")),
        () => Option.fromNullable(paths[0]),
      ), {
        onNone: () => Effect.fail(resolveError(id, "part-missing")),
        onSome: Effect.succeed,
      })
      const projector = Option.fromNullable(paths.find(({ part }) => part.role === "projector"))
      return {
        record: registered.record,
        primaryPath: primary.path,
        shardPaths: paths.filter(({ part }) => part.role === "shard").sort((left, right) => Option.getOrElse(left.part.entry.shardIndex, () => 0) - Option.getOrElse(right.part.entry.shardIndex, () => 0)).map(({ path }) => path),
        projectorPath: Option.map(projector, ({ path }) => path),
        auxiliaryPaths: paths.filter(({ part }) => part.role === "auxiliary").map(({ path }) => path),
        version: resolvedParts.map(({ file }) => ({ key: file.key, sizeBytes: file.sizeBytes, modifiedAtMillis: file.modifiedAtMillis })),
      } satisfies ResolvedModelFiles
    }),
    remove: (id) => SynchronizedRef.modifyEffect(state, (current) => Effect.gen(function* () {
      const entry = yield* Option.match(Option.fromNullable(current.entries.get(id)), {
        onNone: () => Effect.fail(new ModelFileDeleteError({ id, reason: "not-found" })),
        onSome: Effect.succeed,
      })
      if (entry.registration._tag !== "Deletable") return yield* new ModelFileDeleteError({ id, reason: "read-only" })
      yield* entry.registration.source.remove(id)
      const entries = new Map(current.entries)
      entries.delete(id)
      return [current.snapshot, {
        snapshot: Option.none(),
        entries,
        formatCache: current.formatCache,
      }] as const
    })).pipe(Effect.asVoid),
  }
})
