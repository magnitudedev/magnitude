import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Option, Schema, Stream } from "effect"
import { ModelFileSourceId, ModelFileSourceKind, ModelOriginRepositoryId, ModelOriginRevisionId, makeSourceFileKey, makeSourceFileSetId } from "../model-files/identity"
import { normalizeFileSystemFailure } from "../model-files/platform"
import type { ModelFileSource, SourceDiscoveryEvent, SourceFileEntry } from "../model-files/types"
import { SourceDiscoveryError, SourceDiscoveryEvent as SourceDiscoveryEventData, SourceFileSetNotFound, SourceUnavailable } from "../model-files/types"
import { HuggingFaceCommitId, HuggingFaceRepositoryId } from "./identity"

export interface HuggingFaceCacheSourceOptions { readonly root: string; readonly label: Option.Option<string> }

const repositoryFolder = Schema.String.pipe(
  Schema.filter((folder) => folder.startsWith("models--") && folder.slice(8).split("--").length === 2, { message: () => "Not a model repository cache folder" }),
  Schema.transform(HuggingFaceRepositoryId, { strict: true, decode: (folder) => folder.slice(8).replace("--", "/"), encode: (repository) => `models--${repository.replace("/", "--")}` }),
)

export const makeHuggingFaceCacheSource = (options: HuggingFaceCacheSourceOptions): Effect.Effect<ModelFileSource, never, FileSystem.FileSystem | Path.Path> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const id = ModelFileSourceId.make("huggingface-cache")
  const scan = Effect.gen(function* () {
    const root = path.resolve(options.root)
    const rootReal = yield* fs.realPath(root).pipe(Effect.mapError((error) => new SourceDiscoveryError({ sourceId: id, operation: "read-root", reason: normalizeFileSystemFailure(error), path: root })))
    const repositoryFolders = yield* fs.readDirectory(root).pipe(Effect.mapError((error) => new SourceDiscoveryError({ sourceId: id, operation: "read-root", reason: normalizeFileSystemFailure(error), path: root })))
    const events: SourceDiscoveryEvent[] = []
    for (const folder of repositoryFolders.sort()) {
      const repositoryOption = yield* Schema.decodeUnknown(repositoryFolder)(folder).pipe(Effect.option)
      if (Option.isNone(repositoryOption)) {
        if (folder.startsWith("models--")) events.push(SourceDiscoveryEventData.Issue({ issue: { sourceId: id, code: "unsupported_layout", message: "Invalid Hugging Face repository cache folder", sourceKey: Option.some(makeSourceFileKey(folder)) } }))
        continue
      }
      const repository = repositoryOption.value
      const snapshots = path.join(root, folder, "snapshots")
      const commitDirectory = yield* fs.readDirectory(snapshots).pipe(Effect.either)
      if (commitDirectory._tag === "Left") {
        events.push(SourceDiscoveryEventData.Issue({ issue: { sourceId: id, code: "unreadable", message: `Cannot read Hugging Face snapshots (${normalizeFileSystemFailure(commitDirectory.left)})`, sourceKey: Option.some(makeSourceFileKey(repository)) } }))
        continue
      }
      const commits = commitDirectory.right
      for (const commitFolder of commits.sort()) {
        const commitOption = yield* Schema.decodeUnknown(HuggingFaceCommitId)(commitFolder).pipe(Effect.option)
        if (Option.isNone(commitOption)) {
          events.push(SourceDiscoveryEventData.Issue({ issue: { sourceId: id, code: "invalid_manifest", message: "Invalid Hugging Face snapshot commit", sourceKey: Option.some(makeSourceFileKey(`${repository}@${commitFolder}`)) } }))
          continue
        }
        const commit = commitOption.value
        const snapshot = path.join(snapshots, commit)
        const entries: SourceFileEntry[] = []
        const walk = (directory: string): Effect.Effect<void, never> => Effect.gen(function* () {
          const directoryEntries = yield* fs.readDirectory(directory).pipe(Effect.either)
          if (directoryEntries._tag === "Left") {
            events.push(SourceDiscoveryEventData.Issue({ issue: { sourceId: id, code: "unreadable", message: `Cannot read Hugging Face snapshot directory (${normalizeFileSystemFailure(directoryEntries.left)})`, sourceKey: Option.some(makeSourceFileKey(path.relative(root, directory))) } }))
            return
          }
          const names = directoryEntries.right
          for (const name of names.sort()) {
            const candidate = path.join(directory, name)
            const info = yield* fs.stat(candidate).pipe(Effect.either)
            if (info._tag === "Left") {
              events.push(SourceDiscoveryEventData.Issue({ issue: { sourceId: id, code: "unreadable", message: `Cannot inspect Hugging Face cache entry (${normalizeFileSystemFailure(info.left)})`, sourceKey: Option.some(makeSourceFileKey(path.relative(root, candidate))) } }))
              continue
            }
            if (info.right.type === "Directory") { yield* walk(candidate); continue }
            if (info.right.type !== "File" && info.right.type !== "SymbolicLink") continue
            const target = yield* fs.realPath(candidate).pipe(Effect.either)
            if (target._tag === "Left") {
              events.push(SourceDiscoveryEventData.Issue({ issue: { sourceId: id, code: "unreadable", message: `Cannot resolve Hugging Face cache entry (${normalizeFileSystemFailure(target.left)})`, sourceKey: Option.some(makeSourceFileKey(path.relative(root, candidate))) } }))
              continue
            }
            const relativeToRoot = path.relative(rootReal, target.right)
            if (relativeToRoot === ".." || relativeToRoot.startsWith(`..${path.sep}`) || path.isAbsolute(relativeToRoot)) {
              events.push(SourceDiscoveryEventData.Issue({ issue: { sourceId: id, code: "unsafe_path", message: "Hugging Face cache link escapes the cache root", sourceKey: Option.some(makeSourceFileKey(path.relative(root, candidate))) } }))
              continue
            }
            const targetInfo = yield* fs.stat(target.right).pipe(Effect.either)
            if (targetInfo._tag === "Left" || targetInfo.right.type !== "File") {
              const message = targetInfo._tag === "Left" ? `Cannot inspect Hugging Face cache target (${normalizeFileSystemFailure(targetInfo.left)})` : "Hugging Face cache target is not a file"
              events.push(SourceDiscoveryEventData.Issue({ issue: { sourceId: id, code: "unreadable", message, sourceKey: Option.some(makeSourceFileKey(path.relative(root, candidate))) } }))
              continue
            }
            const relativePath = path.relative(snapshot, candidate)
            entries.push({ key: makeSourceFileKey(`${repository}@${commit}:${relativePath}`), path: target.right, relativePath, sizeBytes: Number(targetInfo.right.size), modifiedAtMillis: Option.map(targetInfo.right.mtime, (mtime) => mtime.getTime()), sha256: Option.none(), declaredRole: Option.none(), shardIndex: Option.none() })
          }
        })
        yield* walk(snapshot)
        if (entries.length > 0) events.push(SourceDiscoveryEventData.FileSet({ set: { id: makeSourceFileSetId(`${repository}@${commit}`), artifactKey: Option.none(), sourceId: id, entries, relationships: [], origin: Option.some({ kind: "huggingface", repository: ModelOriginRepositoryId.make(repository), revision: Option.some(ModelOriginRevisionId.make(commit)) }) } }))
      }
    }
    return events
  })
  return {
    id, kind: ModelFileSourceKind.make("huggingface-cache"), label: Option.getOrElse(options.label, () => "Hugging Face cache"), ownership: "external",
    discover: () => Stream.fromIterableEffect(scan),
    resolve: (setId) => scan.pipe(
      Effect.flatMap((events) => {
        const event = Option.fromNullable(events.find((candidate) => candidate._tag === "FileSet" && candidate.set.id === setId))
        return Option.match(event, {
          onNone: () => Effect.fail(new SourceFileSetNotFound({ id: setId })),
          onSome: (found) => found._tag === "FileSet"
            ? Effect.succeed({ set: found.set })
            : Effect.fail(new SourceFileSetNotFound({ id: setId })),
        })
      }),
      Effect.mapError((error) => error._tag === "SourceFileSetNotFound" ? error : new SourceUnavailable({ sourceId: id, setId, reason: "unreadable" })),
    ),
  }
})
