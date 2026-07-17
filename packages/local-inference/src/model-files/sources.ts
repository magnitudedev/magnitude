import { createHash, randomUUID } from "node:crypto"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Option, Schema, Stream } from "effect"
import {
  ModelArtifactKey, ModelFileId, ModelFileSourceId, ModelFileSourceKind, ModelOriginRepositoryId, ModelOriginRevisionId, Sha256Digest, SourceFileKey,
  makeModelFileId, makeSourceFileKey, makeSourceFileSetId, type SourceFileSetId,
} from "./identity"
import { normalizeFileSystemFailure, sha256File } from "./platform"
import type { DeletableModelFileSource, ModelFilePublication, ModelFileRole, ModelFileSource, SourceDiscoveryEvent as SourceDiscoveryEventType, SourceFileEntry, SourceFileRelationship, SourceFileSet, WritableModelFileSource } from "./types"
import { ModelFileDeleteError, ModelFilePublishError, ModelFileRole as ModelFileRoleSchema, SourceDiscoveryError, SourceDiscoveryEvent, SourceFileRelationshipKind, SourceFileSetNotFound, SourceUnavailable } from "./types"

const within = (path: Path.Path, root: string, candidate: string): boolean => {
  const relative = path.relative(root, candidate)
  return relative === "" || (relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
}

const sourceEntry = (absolutePath: string, relativePath: string, info: FileSystem.File.Info): SourceFileEntry => {
  return {
    key: makeSourceFileKey(relativePath), path: absolutePath, relativePath, sizeBytes: Number(info.size),
    modifiedAtMillis: Option.map(info.mtime, (mtime) => mtime.getTime()),
    sha256: Option.none(),
    declaredRole: Option.none(),
    shardIndex: Option.none(),
  }
}

export interface DirectoryModelSourceOptions {
  readonly id: ModelFileSourceId
  readonly label: Option.Option<string>
  readonly root: string
  readonly recursive: boolean
  readonly followSymlinks: boolean
  readonly maxDepth: number
  readonly ignore: Option.Option<(relativePath: string) => boolean>
}

export const makeDirectoryModelSource = (options: DirectoryModelSourceOptions): Effect.Effect<ModelFileSource, never, FileSystem.FileSystem | Path.Path> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const id = options.id
  const scan = Effect.gen(function* () {
    const root = path.resolve(options.root)
    const rootReal = yield* fs.realPath(root).pipe(Effect.mapError((error) => new SourceDiscoveryError({ sourceId: id, operation: "read-root", reason: normalizeFileSystemFailure(error), path: root })))
    const events: SourceDiscoveryEventType[] = []
    const discoveredFiles = new Set<string>()
    const visitedDirectories = new Set<string>([rootReal])

    const visit = (directory: string, depth: number): Effect.Effect<void, never> => Effect.gen(function* () {
      const names = yield* fs.readDirectory(directory).pipe(Effect.match({
        onFailure: (error) => {
          events.push(SourceDiscoveryEvent.Issue({ issue: { sourceId: id, code: "unreadable", message: `Cannot read directory (${normalizeFileSystemFailure(error)})`, sourceKey: Option.none() } }))
          return []
        },
        onSuccess: (value) => value.sort((left, right) => left.localeCompare(right)),
      }))
      const files: SourceFileEntry[] = []
      for (const name of names) {
        const absolute = path.join(directory, name)
        const relative = path.relative(root, absolute)
        if (Option.exists(options.ignore, (ignore) => ignore(relative))) continue
        const link = yield* fs.readLink(absolute).pipe(Effect.option)
        if (Option.isSome(link)) {
          if (!options.followSymlinks) continue
          const target = yield* fs.realPath(absolute).pipe(Effect.either)
          if (target._tag === "Left") {
            events.push(SourceDiscoveryEvent.Issue({ issue: { sourceId: id, code: "unreadable", message: `Cannot resolve symbolic link (${normalizeFileSystemFailure(target.left)})`, sourceKey: Option.some(makeSourceFileKey(relative)) } }))
            continue
          }
          if (!within(path, rootReal, target.right)) {
            events.push(SourceDiscoveryEvent.Issue({ issue: { sourceId: id, code: "unsafe_path", message: "Symbolic link escapes the declared source root", sourceKey: Option.some(makeSourceFileKey(relative)) } }))
            continue
          }
          const targetInfo = yield* fs.stat(target.right).pipe(Effect.either)
          if (targetInfo._tag === "Right" && targetInfo.right.type === "Directory" && options.recursive && depth < options.maxDepth) {
            if (!visitedDirectories.has(target.right)) {
              visitedDirectories.add(target.right)
              yield* visit(target.right, depth + 1)
            }
          } else if (targetInfo._tag === "Right" && targetInfo.right.type === "File" && !discoveredFiles.has(target.right)) {
            discoveredFiles.add(target.right)
            files.push(sourceEntry(target.right, relative, targetInfo.right))
          }
          continue
        }
        const info = yield* fs.stat(absolute).pipe(Effect.either)
        if (info._tag === "Left") {
          events.push(SourceDiscoveryEvent.Issue({ issue: { sourceId: id, code: "unreadable", message: `Cannot inspect entry (${normalizeFileSystemFailure(info.left)})`, sourceKey: Option.some(makeSourceFileKey(relative)) } }))
          continue
        }
        if (info.right.type === "Directory" && options.recursive && depth < options.maxDepth) yield* visit(absolute, depth + 1)
        else if (info.right.type === "File") {
          const canonical = yield* fs.realPath(absolute).pipe(Effect.either)
          if (canonical._tag === "Right" && !discoveredFiles.has(canonical.right)) {
            discoveredFiles.add(canonical.right)
            files.push(sourceEntry(canonical.right, relative, info.right))
          }
        }
      }
      if (files.length > 0) {
        const relativeDirectory = path.relative(root, directory) || "."
        events.push(SourceDiscoveryEvent.FileSet({ set: { id: makeSourceFileSetId(relativeDirectory), artifactKey: Option.none(), sourceId: id, entries: files, relationships: [], origin: Option.none() } }))
      }
    })

    yield* visit(root, 0)
    return events
  })

  return {
    id, kind: ModelFileSourceKind.make("directory"), label: Option.getOrElse(options.label, () => options.root), ownership: "external",
    discover: () => Stream.fromIterableEffect(scan),
    resolve: (setId) => scan.pipe(
      Effect.flatMap((events) => {
        const found = Option.fromNullable(events.find((event) => event._tag === "FileSet" && event.set.id === setId))
        return Option.match(found, {
          onNone: () => Effect.fail(new SourceFileSetNotFound({ id: setId })),
          onSome: (event) => event._tag === "FileSet"
            ? Effect.succeed({ set: event.set })
            : Effect.fail(new SourceFileSetNotFound({ id: setId })),
        })
      }),
      Effect.mapError((error) => error._tag === "SourceFileSetNotFound" ? error : new SourceUnavailable({ sourceId: id, setId, reason: "unreadable" })),
    ),
  }
})

const ManifestFile = Schema.Struct({
  key: SourceFileKey, path: Schema.String, role: ModelFileRoleSchema,
  shardIndex: Schema.optionalWith(Schema.NonNegativeInt, { as: "Option", exact: true }), sizeBytes: Schema.NonNegativeInt, sha256: Sha256Digest,
})
const ManifestRelationship = Schema.Struct({ kind: SourceFileRelationshipKind, from: SourceFileKey, to: SourceFileKey })
const OwnedManifest = Schema.Struct({
  version: Schema.Literal(1), artifactKey: ModelArtifactKey, files: Schema.Array(ManifestFile), relationships: Schema.Array(ManifestRelationship),
  origin: Schema.optionalWith(
    Schema.Struct({ kind: Schema.Literal("huggingface"), repository: ModelOriginRepositoryId, revision: ModelOriginRevisionId }),
    { as: "Option", exact: true },
  ),
})
type OwnedManifest = Schema.Schema.Type<typeof OwnedManifest>
const OwnedManifestJson = Schema.parseJson(OwnedManifest, { space: 2 })

export interface MagnitudeModelSourceOptions { readonly root: string; readonly id: Option.Option<ModelFileSourceId>; readonly label: Option.Option<string> }

export const makeMagnitudeModelSource = (options: MagnitudeModelSourceOptions): Effect.Effect<WritableModelFileSource & DeletableModelFileSource, never, FileSystem.FileSystem | Path.Path> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const id = Option.getOrElse(options.id, () => ModelFileSourceId.make("magnitude"))

  const readSet = (directory: string): Effect.Effect<SourceFileSet, SourceDiscoveryError> => Effect.gen(function* () {
    const manifestPath = path.join(directory, "manifest.json")
    const text = yield* fs.readFileString(manifestPath).pipe(Effect.mapError((error) => new SourceDiscoveryError({ sourceId: id, operation: "inspect-entry", reason: normalizeFileSystemFailure(error), path: manifestPath })))
    const manifest = yield* Schema.decode(OwnedManifestJson)(text).pipe(Effect.mapError(() => new SourceDiscoveryError({ sourceId: id, operation: "inspect-entry", reason: "invalid-data", path: manifestPath })))
    const entries = yield* Effect.forEach(manifest.files, (file) => Effect.gen(function* () {
      const absolute = path.resolve(directory, file.path)
      if (!within(path, directory, absolute)) return yield* new SourceDiscoveryError({ sourceId: id, operation: "inspect-entry", reason: "invalid-data", path: file.path })
      const info = yield* fs.stat(absolute).pipe(Effect.mapError((error) => new SourceDiscoveryError({ sourceId: id, operation: "inspect-entry", reason: normalizeFileSystemFailure(error), path: absolute })))
      if (info.type !== "File" || Number(info.size) !== file.sizeBytes) return yield* new SourceDiscoveryError({ sourceId: id, operation: "inspect-entry", reason: "invalid-data", path: absolute })
      return { key: file.key, path: absolute, relativePath: file.path, sizeBytes: file.sizeBytes, sha256: Option.some(file.sha256), declaredRole: Option.some(file.role), shardIndex: file.shardIndex, modifiedAtMillis: Option.map(info.mtime, (mtime) => mtime.getTime()) } satisfies SourceFileEntry
    }))
    return { id: makeSourceFileSetId(manifest.artifactKey), artifactKey: Option.some(manifest.artifactKey), sourceId: id, entries, relationships: manifest.relationships, origin: Option.map(manifest.origin, (origin) => ({ ...origin, revision: Option.some(origin.revision) })) }
  })

  const discover = Effect.gen(function* () {
    const root = path.resolve(options.root)
    yield* fs.makeDirectory(root, { recursive: true }).pipe(Effect.mapError((error) => new SourceDiscoveryError({ sourceId: id, operation: "read-root", reason: normalizeFileSystemFailure(error), path: root })))
    const names = yield* fs.readDirectory(root).pipe(Effect.mapError((error) => new SourceDiscoveryError({ sourceId: id, operation: "read-root", reason: normalizeFileSystemFailure(error), path: root })))
    return yield* Effect.forEach(names.filter((name) => !name.startsWith(".")).sort(), (name) => readSet(path.join(root, name)).pipe(Effect.match({
      onFailure: () => SourceDiscoveryEvent.Issue({ issue: { sourceId: id, code: "invalid_manifest", message: "Artifact manifest or published files are invalid", sourceKey: Option.none() } }),
      onSuccess: (set) => SourceDiscoveryEvent.FileSet({ set }),
    })))
  })

  const find = (setId: SourceFileSetId) => discover.pipe(Effect.map((events) => events.find((event) => event._tag === "FileSet" && event.set.id === setId)))
  const publishError = (publication: ModelFilePublication, operation: ModelFilePublishError["operation"], reason: ModelFilePublishError["reason"], failurePath: Option.Option<string> = Option.none()) => new ModelFilePublishError({ sourceId: id, artifactKey: publication.artifactKey, operation, reason, path: failurePath })

  return {
    id, kind: ModelFileSourceKind.make("magnitude"), label: Option.getOrElse(options.label, () => "Magnitude models"), ownership: "magnitude",
    discover: () => Stream.fromIterableEffect(discover),
    resolve: (setId) => find(setId).pipe(
      Effect.flatMap((event) => Option.match(Option.fromNullable(event), {
        onNone: () => Effect.fail(new SourceFileSetNotFound({ id: setId })),
        onSome: (found) => found._tag === "FileSet"
          ? Effect.succeed({ set: found.set })
          : Effect.fail(new SourceFileSetNotFound({ id: setId })),
      })),
      Effect.mapError((error) => error._tag === "SourceFileSetNotFound" ? error : new SourceUnavailable({ sourceId: id, setId, reason: "unreadable" })),
    ),
    publish: (publication: ModelFilePublication) => Effect.gen(function* () {
      const root = path.resolve(options.root)
      if (publication.files.length === 0) return yield* publishError(publication, "validate", "empty")
      if (publication.files.filter(({ role }) => role === "primary").length !== 1) return yield* publishError(publication, "validate", "primary-count")
      const keys = new Set(publication.files.map(({ key }) => key))
      if (keys.size !== publication.files.length) return yield* publishError(publication, "validate", "duplicate-key")
      if (publication.relationships.some(({ from, to }) => !keys.has(from) || !keys.has(to))) return yield* publishError(publication, "validate", "invalid-relationship")
      yield* fs.makeDirectory(root, { recursive: true }).pipe(Effect.mapError((error) => publishError(publication, "stage", normalizeFileSystemFailure(error), Option.some(root))))
      const destination = path.join(root, `artifact-${createHash("sha256").update(publication.artifactKey).digest("hex")}`)
      const temporary = path.join(root, `.publishing-${randomUUID()}`)
      yield* Effect.acquireUseRelease(
        fs.makeDirectory(temporary, { recursive: true }).pipe(Effect.as(temporary), Effect.mapError((error) => publishError(publication, "stage", normalizeFileSystemFailure(error), Option.some(temporary)))),
        () => Effect.gen(function* () {
          const manifestFiles: OwnedManifest["files"][number][] = []
          for (const file of publication.files) {
            const target = path.resolve(temporary, file.publishedRelativePath)
            if (!within(path, temporary, target)) return yield* publishError(publication, "validate", "unsafe-path", Option.some(file.publishedRelativePath))
            yield* fs.makeDirectory(path.dirname(target), { recursive: true }).pipe(Effect.mapError((error) => publishError(publication, "stage", normalizeFileSystemFailure(error), Option.some(target))))
            yield* fs.copyFile(file.stagedPath, target).pipe(Effect.mapError((error) => publishError(publication, "stage", normalizeFileSystemFailure(error), Option.some(file.stagedPath))))
            const info = yield* fs.stat(target).pipe(Effect.mapError((error) => publishError(publication, "verify", normalizeFileSystemFailure(error), Option.some(target))))
            if (Number(info.size) !== file.sizeBytes) return yield* publishError(publication, "verify", "size-mismatch", Option.some(file.publishedRelativePath))
            const digest = yield* sha256File(target).pipe(Effect.provideService(FileSystem.FileSystem, fs), Effect.mapError((error) => publishError(publication, "verify", normalizeFileSystemFailure(error), Option.some(target))))
            if (digest !== file.sha256) return yield* publishError(publication, "verify", "digest-mismatch", Option.some(file.publishedRelativePath))
            manifestFiles.push({ key: file.key, path: file.publishedRelativePath, role: file.role, shardIndex: file.shardIndex, sizeBytes: file.sizeBytes, sha256: file.sha256 })
          }
          const manifestOrigin = Option.map(publication.origin, ({ kind, repository, revision }) => ({ kind, repository, revision }))
          const encoded = yield* Schema.encode(OwnedManifestJson)({ version: 1, artifactKey: publication.artifactKey, files: manifestFiles, relationships: publication.relationships, origin: manifestOrigin }).pipe(Effect.mapError(() => publishError(publication, "validate", "invalid-data")))
          yield* fs.writeFileString(path.join(temporary, "manifest.json"), encoded, { mode: 0o600 }).pipe(Effect.mapError((error) => publishError(publication, "commit", normalizeFileSystemFailure(error), Option.some(temporary))))
          yield* fs.rename(temporary, destination).pipe(Effect.mapError((error) => publishError(publication, "commit", normalizeFileSystemFailure(error), Option.some(destination))))
        }),
        () => fs.remove(temporary, { recursive: true, force: true }).pipe(Effect.ignore),
      )
      return makeModelFileId(id, publication.artifactKey)
    }),
    remove: (modelId: ModelFileId) => Effect.gen(function* () {
      const events = yield* discover.pipe(Effect.mapError(() => new ModelFileDeleteError({ id: modelId, reason: "source-unavailable" })))
      const event = Option.fromNullable(events.find((candidate) => candidate._tag === "FileSet" && Option.exists(candidate.set.artifactKey, (key) => makeModelFileId(id, key) === modelId)))
      if (Option.isNone(event) || event.value._tag !== "FileSet" || Option.isNone(event.value.set.artifactKey)) return yield* new ModelFileDeleteError({ id: modelId, reason: "not-found" })
      const directory = path.join(path.resolve(options.root), `artifact-${createHash("sha256").update(event.value.set.artifactKey.value).digest("hex")}`)
      yield* fs.remove(directory, { recursive: true }).pipe(Effect.mapError((error) => new ModelFileDeleteError({ id: modelId, reason: normalizeFileSystemFailure(error) })))
    }),
  }
})
