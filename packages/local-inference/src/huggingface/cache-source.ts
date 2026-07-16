import { randomUUID } from "node:crypto"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Option, Schema, Stream } from "effect"
import {
  ModelArtifactKey,
  ModelFileDeleteError,
  ModelFileSourceKind,
  ModelOriginRepositoryId,
  ModelOriginRevisionId,
  SourceDiscoveryEvent,
  SourceDiscoveryError,
  SourceFileSetNotFound,
  SourceUnavailable,
  makeModelFileId,
  makeSourceFileKey,
  makeSourceFileSetId,
  type DeletableModelFileSource,
  type ModelFileId,
  type SourceDiscoveryEvent as SourceDiscoveryEventType,
  type SourceFileSet,
} from "../model-files"
import type { HuggingFaceManagedStoreOptions } from "./contracts"
import { normalizeFileSystemFailure } from "../model-files/platform"
import { blobPath, isWithin, remoteContentKey } from "./cache-paths"
import { makeHuggingFaceArtifactId } from "./artifact-identity"
import { HuggingFaceInstallationManifestJson, type HuggingFaceInstallationManifest } from "./installation-schema"

export interface HuggingFaceCacheSourceOptions {
  readonly store: HuggingFaceManagedStoreOptions
  readonly label: Option.Option<string>
}

interface InstalledSet { readonly manifestPath: string; readonly manifest: HuggingFaceInstallationManifest; readonly set: SourceFileSet }

export const makeHuggingFaceCacheSource = (options: HuggingFaceCacheSourceOptions): Effect.Effect<DeletableModelFileSource, never, FileSystem.FileSystem | Path.Path> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const id = options.store.sourceId
  const cacheRoot = path.resolve(options.store.cacheRoot)
  const installationRoot = path.resolve(options.store.installationRoot)
  yield* fs.makeDirectory(cacheRoot, { recursive: true }).pipe(Effect.ignore)
  const cacheRootReal = yield* fs.realPath(cacheRoot).pipe(Effect.orElseSucceed(() => cacheRoot))

  const readInstalled = (manifestPath: string): Effect.Effect<InstalledSet, unknown> => Effect.gen(function* () {
    const text = yield* fs.readFileString(manifestPath)
    const manifest = yield* Schema.decode(HuggingFaceInstallationManifestJson)(text)
    if (makeHuggingFaceArtifactId(manifest.artifact) !== manifest.artifact.id) return yield* Effect.fail("artifact-id-mismatch" as const)
    if (manifest.files.length !== manifest.artifact.files.length) return yield* Effect.fail("manifest-file-count-mismatch" as const)
    for (const artifactFile of manifest.artifact.files) {
      const cachedFile = manifest.files.find(({ path }) => path === artifactFile.path)
      if (!cachedFile || cachedFile.role !== artifactFile.role || cachedFile.sizeBytes !== artifactFile.sizeBytes || Option.getOrUndefined(cachedFile.shardIndex) !== Option.getOrUndefined(artifactFile.shardIndex) || remoteContentKey(cachedFile.content) !== remoteContentKey(artifactFile.content)) return yield* Effect.fail("manifest-file-mismatch" as const)
    }
    const entries = yield* Effect.forEach(manifest.files, (file) => Effect.gen(function* () {
      const absolute = path.resolve(cacheRoot, file.snapshotRelativePath)
      if (!isWithin(path, cacheRoot, absolute)) return yield* Effect.fail("manifest-path-escapes-cache-root" as const)
      const resolved = yield* fs.realPath(absolute)
      if (!isWithin(path, cacheRootReal, resolved)) return yield* Effect.fail("snapshot-target-escapes-cache-root" as const)
      const info = yield* fs.stat(resolved)
      if (info.type !== "File" || Number(info.size) !== file.sizeBytes) return yield* Effect.fail("cached-file-does-not-match-manifest" as const)
      return {
        key: makeSourceFileKey(`${manifest.artifact.id}:${file.path}`),
        path: resolved,
        relativePath: file.path,
        sizeBytes: file.sizeBytes,
        modifiedAtMillis: Option.map(info.mtime, (mtime) => mtime.getTime()),
        sha256: file.content._tag === "LfsSha256" ? Option.some(file.content.sha256) : Option.none(),
        declaredRole: Option.some(file.role),
        shardIndex: file.shardIndex,
      }
    }))
    const keys = new Map(manifest.files.map((file) => [file.path, makeSourceFileKey(`${manifest.artifact.id}:${file.path}`)]))
    if (manifest.artifact.relationships.some(({ fromPath, toPath }) => !keys.has(fromPath) || !keys.has(toPath))) return yield* Effect.fail("invalid-manifest-relationship" as const)
    return {
      manifestPath,
      manifest,
      set: {
        id: makeSourceFileSetId(manifest.artifact.id),
        artifactKey: Option.some(ModelArtifactKey.make(manifest.artifact.id)),
        sourceId: id,
        entries,
        relationships: manifest.artifact.relationships.map((relationship) => ({ kind: relationship.kind, from: keys.get(relationship.fromPath)!, to: keys.get(relationship.toPath)! })),
        origin: Option.some({ kind: "huggingface", repository: ModelOriginRepositoryId.make(manifest.artifact.repository), revision: Option.some(ModelOriginRevisionId.make(manifest.artifact.commit)) }),
      },
    }
  })

  const scan = Effect.gen(function* () {
    yield* fs.makeDirectory(installationRoot, { recursive: true })
    const names = yield* fs.readDirectory(installationRoot)
    return yield* Effect.forEach(names.filter((name) => name.endsWith(".json") && !name.startsWith(".")).sort(), (name) => {
      const manifestPath = path.join(installationRoot, name)
      return readInstalled(manifestPath).pipe(Effect.match({
        onFailure: (error) => ({ event: SourceDiscoveryEvent.Issue({ issue: { sourceId: id, code: "invalid_manifest", message: `Hugging Face installation manifest or cached files are invalid (${String(error).slice(0, 256)})`, sourceKey: Option.some(makeSourceFileKey(name)) } }), installed: Option.none<InstalledSet>() }),
        onSuccess: (installed) => ({ event: SourceDiscoveryEvent.FileSet({ set: installed.set }), installed: Option.some(installed) }),
      }))
    })
  })

  const findBySet = (setId: SourceFileSet["id"]) => scan.pipe(Effect.map((items) => items.flatMap(({ installed }) => Option.toArray(installed)).find(({ set }) => set.id === setId)))
  const findByModel = (modelId: ModelFileId) => scan.pipe(Effect.map((items) => items.flatMap(({ installed }) => Option.toArray(installed)).find(({ manifest }) => makeModelFileId(id, ModelArtifactKey.make(manifest.artifact.id)) === modelId)))

  return {
    id,
    kind: ModelFileSourceKind.make("huggingface-cache"),
    label: Option.getOrElse(options.label, () => "Hugging Face models"),
    ownership: "magnitude",
    discover: () => Stream.fromIterableEffect(scan.pipe(
      Effect.map((items) => items.map(({ event }) => event satisfies SourceDiscoveryEventType)),
      Effect.mapError(() => new SourceDiscoveryError({ sourceId: id, operation: "read-root", reason: "invalid-data", path: installationRoot })),
    )),
    resolve: (setId) => findBySet(setId).pipe(
      Effect.flatMap((found) => found ? Effect.succeed({ set: found.set }) : Effect.fail(new SourceFileSetNotFound({ id: setId }))),
      Effect.mapError((error) => error._tag === "SourceFileSetNotFound" ? error : new SourceUnavailable({ sourceId: id, setId, reason: "unreadable" })),
    ),
    remove: (modelId) => Effect.gen(function* () {
      const installed = yield* findByModel(modelId).pipe(Effect.mapError(() => new ModelFileDeleteError({ id: modelId, reason: "source-unavailable" })))
      if (!installed) return yield* new ModelFileDeleteError({ id: modelId, reason: "not-found" })
      const tombstone = path.join(installationRoot, `.cleanup-${randomUUID()}.json`)
      yield* fs.rename(installed.manifestPath, tombstone).pipe(Effect.mapError((error) => new ModelFileDeleteError({ id: modelId, reason: normalizeFileSystemFailure(error) })))
      const remaining = yield* scan.pipe(Effect.orElseSucceed(() => []))
      const referenced = new Set(remaining.flatMap(({ installed }) => Option.toArray(installed)).flatMap(({ manifest }) => manifest.files.map(({ content }) => `${content._tag}:${content._tag === "LfsSha256" ? content.sha256 : content._tag === "Xet" ? content.hash : content.oid}`)))
      for (const file of installed.manifest.files) {
        const pointer = path.resolve(cacheRoot, file.snapshotRelativePath)
        if (isWithin(path, cacheRoot, pointer)) yield* fs.remove(pointer, { force: true }).pipe(Effect.ignore)
        const reference = `${file.content._tag}:${file.content._tag === "LfsSha256" ? file.content.sha256 : file.content._tag === "Xet" ? file.content.hash : file.content.oid}`
        if (!referenced.has(reference)) yield* fs.remove(blobPath(path, cacheRoot, installed.manifest.artifact.repository, file), { force: true }).pipe(Effect.ignore)
      }
      yield* fs.remove(tombstone, { force: true }).pipe(Effect.ignore)
    }),
  }
})
