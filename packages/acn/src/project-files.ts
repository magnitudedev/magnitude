import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { lstat } from "node:fs/promises"
import { createHash } from "node:crypto"
import {
  InvalidProjectFilePath,
  ProjectFileAccessDenied,
  ProjectFileAlreadyExists,
  ProjectFileConflict,
  ProjectFileNotFound,
  ProjectFileOperationFailed,
  ProjectFileError as ProjectFileErrorSchema,
  ProjectFileRevisionSchema,
  ProjectRelativePathSchema,
  type ProjectDirectoryListing,
  type ProjectEntryMove,
  type ProjectFileError,
  type ProjectFileSnapshot,
  type ProjectFileTextSnapshot,
  type ProjectFileRevision,
  type ProjectFilesChange,
  type ProjectId,
  type ProjectRelativePath,
} from "@magnitudedev/acn-protocol"
import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { ProjectRegistry } from "./project-registry"

const MAX_TEXT_BYTES = 5 * 1024 * 1024
const MAX_IMAGE_BYTES = 10 * 1024 * 1024
const imageTypes = new Map<string, "image/png" | "image/jpeg" | "image/gif" | "image/webp">([
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".gif", "image/gif"],
  [".webp", "image/webp"],
] as const)

export interface ProjectFilesApi {
  readonly listDirectory: (projectId: ProjectId, directory: ProjectRelativePath) => Effect.Effect<ProjectDirectoryListing, ProjectFileError>
  readonly watchChanges: (projectId: ProjectId) => Stream.Stream<ProjectFilesChange, ProjectFileError>
  readonly readFile: (projectId: ProjectId, path: ProjectRelativePath) => Effect.Effect<ProjectFileSnapshot, ProjectFileError>
  readonly writeFile: (input: { readonly projectId: ProjectId; readonly path: ProjectRelativePath; readonly content: string; readonly expectedRevision: ProjectFileRevision }) => Effect.Effect<ProjectFileTextSnapshot, ProjectFileError>
  readonly deleteFile: (input: { readonly projectId: ProjectId; readonly path: ProjectRelativePath; readonly expectedRevision: ProjectFileRevision }) => Effect.Effect<void, ProjectFileError>
  readonly moveEntry: (input: { readonly projectId: ProjectId; readonly sourcePath: ProjectRelativePath; readonly destinationDirectory: ProjectRelativePath }) => Effect.Effect<ProjectEntryMove, ProjectFileError>
}

export class ProjectFiles extends Context.Tag("ProjectFiles")<ProjectFiles, ProjectFilesApi>() {}

const revision = (bytes: Uint8Array) => ProjectFileRevisionSchema.make(
  createHash("sha256").update(bytes).digest("hex"),
)

const isMissingFileError = (error: unknown): boolean =>
  typeof error === "object" && error !== null && (
    ("code" in error && error.code === "ENOENT")
    || ("reason" in error && error.reason === "NotFound")
  )

const operationFailed = (operation: string, path: ProjectRelativePath) => (cause: unknown) => {
  if (Schema.is(ProjectFileErrorSchema)(cause)) return cause
  const reason = cause instanceof Error ? cause.message : String(cause)
  if (isMissingFileError(cause)) return new ProjectFileNotFound({ path })
  return new ProjectFileOperationFailed({ operation, path, reason })
}

const newlineKind = (content: string): ProjectFileTextSnapshot["newline"] => {
  const crlf = content.includes("\r\n")
  const lf = content.replaceAll("\r\n", "").includes("\n")
  return crlf && lf ? "mixed" : crlf ? "crlf" : lf ? "lf" : "none"
}

const parentDirectory = (path: ProjectRelativePath): ProjectRelativePath => {
  const separator = path.lastIndexOf("/")
  return ProjectRelativePathSchema.make(separator === -1 ? "" : path.slice(0, separator))
}

export const ProjectFilesLive = Layer.effect(
  ProjectFiles,
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const paths = yield* Path.Path
    const registry = yield* ProjectRegistry
    const mutations = yield* Effect.makeSemaphore(1)

    const resolve = Effect.fn("acn.project-files.resolve")(function* (
      projectId: ProjectId,
      relativePath: ProjectRelativePath,
      requireExisting = true,
    ) {
      const root = yield* registry.resolveSourceDirectory(projectId).pipe(
        Effect.mapError((error) => error._tag === "ProjectNotFound"
          ? error
          : new ProjectFileOperationFailed({ operation: "resolve project", path: relativePath, reason: String(error) })),
      )
      const canonicalRoot = yield* fs.realPath(root).pipe(
        Effect.mapError(operationFailed("resolve project source", relativePath)),
      )
      const target = paths.resolve(canonicalRoot, relativePath || ".")
      const lexical = paths.relative(canonicalRoot, target)
      if (lexical === ".." || lexical.startsWith(`..${paths.sep}`) || paths.resolve(target) === paths.resolve(canonicalRoot, "..")) {
        return yield* new InvalidProjectFilePath({ path: relativePath, reason: "Path escapes the project source" })
      }
      let cursor = canonicalRoot
      for (const segment of relativePath === "" ? [] : relativePath.split("/")) {
        cursor = paths.join(cursor, segment)
        const info = yield* Effect.tryPromise({
          try: () => lstat(cursor),
          catch: operationFailed("inspect project path", relativePath),
        }).pipe(Effect.catchIf(
          (error) => error._tag === "ProjectFileNotFound" && !requireExisting && cursor === target,
          () => Effect.void,
        ))
        if (info?.isSymbolicLink()) {
          return yield* new ProjectFileAccessDenied({ path: relativePath, reason: "Symbolic links are not accessible from the project browser" })
        }
      }
      if (requireExisting) {
        const realTarget = yield* fs.realPath(target).pipe(Effect.mapError(operationFailed("resolve project path", relativePath)))
        const canonicalRelative = paths.relative(canonicalRoot, realTarget)
        if (canonicalRelative === ".." || canonicalRelative.startsWith(`..${paths.sep}`)) {
          return yield* new ProjectFileAccessDenied({ path: relativePath, reason: "Resolved path escapes the project source" })
        }
      }
      return target
    })

    const readText = Effect.fn("acn.project-files.read-text")(function* (
      path: ProjectRelativePath,
      absolute: string,
      bytes?: Uint8Array,
    ) {
      const data = bytes ?? (yield* fs.readFile(absolute).pipe(Effect.mapError(operationFailed("read file", path))))
      const content = yield* Effect.try({
        try: () => new TextDecoder("utf-8", { fatal: true }).decode(data),
        catch: () => new ProjectFileAccessDenied({ path, reason: "File is not valid UTF-8 text" }),
      })
      return {
        _tag: "text" as const,
        path,
        content,
        revision: revision(data),
        size: data.byteLength,
        newline: newlineKind(content),
      }
    })

    const readFile = Effect.fn("acn.project-files.read-file")(function* (projectId: ProjectId, path: ProjectRelativePath) {
      if (path === "") return yield* new InvalidProjectFilePath({ path, reason: "A file path cannot be empty" })
      const absolute = yield* resolve(projectId, path)
      const info = yield* fs.stat(absolute).pipe(Effect.mapError(operationFailed("inspect file", path)))
      if (info.type !== "File") return yield* new ProjectFileAccessDenied({ path, reason: "Path is not a regular file" })
      const size = Number(info.size)
      const mediaType = imageTypes.get(paths.extname(path).toLowerCase())
      if (mediaType !== undefined) {
        if (size > MAX_IMAGE_BYTES) return { _tag: "unsupported" as const, path, reason: "too_large" as const, size }
        const bytes = yield* fs.readFile(absolute).pipe(Effect.mapError(operationFailed("read image", path)))
        return { _tag: "image" as const, path, mediaType, data: Buffer.from(bytes).toString("base64"), revision: revision(bytes), size }
      }
      if (size > MAX_TEXT_BYTES) return { _tag: "unsupported" as const, path, reason: "too_large" as const, size }
      const bytes = yield* fs.readFile(absolute).pipe(Effect.mapError(operationFailed("read file", path)))
      if (bytes.includes(0)) return { _tag: "unsupported" as const, path, reason: "binary" as const, size }
      return yield* readText(path, absolute, bytes).pipe(
        Effect.catchTag("ProjectFileAccessDenied", () => Effect.succeed({ _tag: "unsupported" as const, path, reason: "binary" as const, size })),
      )
    })

    const listDirectory = Effect.fn("acn.project-files.list-directory")(function* (projectId: ProjectId, directory: ProjectRelativePath) {
      const absolute = yield* resolve(projectId, directory)
      const info = yield* fs.stat(absolute).pipe(Effect.mapError(operationFailed("inspect directory", directory)))
      if (info.type !== "Directory") return yield* new ProjectFileAccessDenied({ path: directory, reason: "Path is not a directory" })
      const names = yield* fs.readDirectory(absolute).pipe(Effect.mapError(operationFailed("list directory", directory)))
      const entries = yield* Effect.forEach(names.filter((name) => name !== ".git"), (name) => Effect.gen(function* () {
        const child = ProjectRelativePathSchema.make(directory === "" ? name : `${directory}/${name}`)
        const childAbsolute = paths.join(absolute, name)
        const stat = yield* Effect.tryPromise({ try: () => lstat(childAbsolute), catch: operationFailed("inspect directory entry", child) })
        if (stat.isSymbolicLink() || (!stat.isDirectory() && !stat.isFile())) return Option.none()
        return Option.some({ name, path: child, kind: stat.isDirectory() ? "directory" as const : "file" as const, size: stat.isFile() ? Option.some(stat.size) : Option.none() })
      }), { concurrency: 16 })
      return {
        projectId,
        directory,
        entries: entries.flatMap(Option.match({ onNone: () => [], onSome: (entry) => [entry] })).sort((a, b) => a.kind === b.kind ? a.name.localeCompare(b.name) : a.kind === "directory" ? -1 : 1),
      }
    })

    const watchChanges = (projectId: ProjectId) => Stream.unwrap(
      resolve(projectId, ProjectRelativePathSchema.make("")).pipe(
        Effect.map((root) => fs.watch(root, { recursive: true }).pipe(
          Stream.filter((event) => event.path !== ".git" && !event.path.startsWith(".git/")),
          Stream.debounce("50 millis"),
          Stream.map(() => ({ projectId })),
          Stream.mapError(operationFailed("watch project files", ProjectRelativePathSchema.make(""))),
        )),
      ),
    )

    const writeFile = (input: { readonly projectId: ProjectId; readonly path: ProjectRelativePath; readonly content: string; readonly expectedRevision: ProjectFileRevision }) => mutations.withPermits(1)(Effect.gen(function* () {
      const current = yield* readFile(input.projectId, input.path)
      if (current._tag !== "text") return yield* new ProjectFileAccessDenied({ path: input.path, reason: "Only text files can be edited" })
      if (current.revision !== input.expectedRevision) return yield* new ProjectFileConflict({ path: input.path, current })
      const absolute = yield* resolve(input.projectId, input.path)
      const bytes = new TextEncoder().encode(input.content)
      if (bytes.byteLength > MAX_TEXT_BYTES) return yield* new ProjectFileAccessDenied({ path: input.path, reason: "File exceeds the 5 MiB editing limit" })
      const info = yield* fs.stat(absolute).pipe(Effect.mapError(operationFailed("inspect file before save", input.path)))
      const temp = paths.join(paths.dirname(absolute), `.magnitude-${paths.basename(absolute)}-${crypto.randomUUID()}.tmp`)
      yield* Effect.scoped(fs.open(temp, { flag: "wx", mode: info.mode }).pipe(
        Effect.flatMap((file) => file.writeAll(bytes).pipe(Effect.zipRight(file.sync))),
      )).pipe(
        Effect.zipRight(Effect.gen(function* () {
          const latest = yield* readFile(input.projectId, input.path)
          if (latest._tag !== "text") return yield* new ProjectFileAccessDenied({ path: input.path, reason: "Only text files can be edited" })
          if (latest.revision !== input.expectedRevision) return yield* new ProjectFileConflict({ path: input.path, current: latest })
          yield* fs.rename(temp, absolute)
        })),
        Effect.ensuring(fs.remove(temp, { force: true }).pipe(Effect.ignore)),
        Effect.mapError(operationFailed("save file", input.path)),
      )
      return yield* readText(input.path, absolute)
    }))

    const deleteFile = (input: { readonly projectId: ProjectId; readonly path: ProjectRelativePath; readonly expectedRevision: ProjectFileRevision }) => mutations.withPermits(1)(Effect.gen(function* () {
      const current = yield* readFile(input.projectId, input.path)
      if (current._tag === "unsupported") {
        return yield* new ProjectFileAccessDenied({ path: input.path, reason: "Unsupported files cannot be deleted from the project browser" })
      }
      if (current.revision !== input.expectedRevision) {
        if (current._tag !== "text") {
          return yield* new ProjectFileAccessDenied({ path: input.path, reason: "The file changed on disk" })
        }
        return yield* new ProjectFileConflict({ path: input.path, current })
      }
      const absolute = yield* resolve(input.projectId, input.path)
      const latest = yield* readFile(input.projectId, input.path)
      if (latest.revision !== input.expectedRevision) {
        if (latest._tag !== "text") {
          return yield* new ProjectFileAccessDenied({ path: input.path, reason: "The file changed on disk" })
        }
        return yield* new ProjectFileConflict({ path: input.path, current: latest })
      }
      yield* fs.remove(absolute).pipe(Effect.mapError(operationFailed("delete file", input.path)))
    }))

    const moveEntry = (input: {
      readonly projectId: ProjectId
      readonly sourcePath: ProjectRelativePath
      readonly destinationDirectory: ProjectRelativePath
    }) => mutations.withPermits(1)(Effect.gen(function* () {
      if (input.sourcePath === "") {
        return yield* new InvalidProjectFilePath({
          path: input.sourcePath,
          reason: "The project root cannot be moved",
        })
      }

      if (parentDirectory(input.sourcePath) === input.destinationDirectory) {
        return yield* new ProjectFileAccessDenied({
          path: input.sourcePath,
          reason: "The entry is already in that directory",
        })
      }

      const sourceAbsolute = yield* resolve(input.projectId, input.sourcePath)
      const sourceInfo = yield* fs.stat(sourceAbsolute).pipe(
        Effect.mapError(operationFailed("inspect move source", input.sourcePath)),
      )
      if (sourceInfo.type !== "Directory" && sourceInfo.type !== "File") {
        return yield* new ProjectFileAccessDenied({
          path: input.sourcePath,
          reason: "Only regular files and directories can be moved",
        })
      }

      if (
        sourceInfo.type === "Directory"
        && (input.destinationDirectory === input.sourcePath
          || input.destinationDirectory.startsWith(`${input.sourcePath}/`))
      ) {
        return yield* new ProjectFileAccessDenied({
          path: input.sourcePath,
          reason: "A directory cannot be moved into itself or one of its descendants",
        })
      }

      const destinationDirectoryAbsolute = yield* resolve(input.projectId, input.destinationDirectory)
      const destinationDirectoryInfo = yield* fs.stat(destinationDirectoryAbsolute).pipe(
        Effect.mapError(operationFailed("inspect move destination", input.destinationDirectory)),
      )
      if (destinationDirectoryInfo.type !== "Directory") {
        return yield* new ProjectFileAccessDenied({
          path: input.destinationDirectory,
          reason: "The move destination is not a directory",
        })
      }

      const basename = paths.basename(input.sourcePath)
      const destinationPath = ProjectRelativePathSchema.make(
        input.destinationDirectory === ""
          ? basename
          : `${input.destinationDirectory}/${basename}`,
      )
      const destinationAbsolute = paths.join(destinationDirectoryAbsolute, basename)
      return yield* Effect.uninterruptible(Effect.gen(function* () {
        const destinationExists = yield* Effect.tryPromise({
          try: async () => {
            try {
              await lstat(destinationAbsolute)
              return true
            } catch (error) {
              if (isMissingFileError(error)) return false
              throw error
            }
          },
          catch: operationFailed("inspect move destination", destinationPath),
        })
        if (destinationExists) return yield* new ProjectFileAlreadyExists({ path: destinationPath })

        yield* fs.rename(sourceAbsolute, destinationAbsolute).pipe(
          Effect.mapError(operationFailed("move project entry", input.sourcePath)),
        )
        return {
          sourcePath: input.sourcePath,
          destinationPath,
          kind: sourceInfo.type === "Directory" ? "directory" as const : "file" as const,
        }
      }))
    }))

    return ProjectFiles.of({ listDirectory, watchChanges, readFile, writeFile, deleteFile, moveEntry })
  }),
)
