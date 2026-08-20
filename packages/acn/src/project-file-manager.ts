import {
  FileContentHashSchema,
  InvalidProjectFilePath,
  ProjectFileAccessDenied,
  ProjectFileAlreadyExists,
  ProjectFileChanged,
  ProjectFileNotFound,
  ProjectFileTooLarge,
  RelativePathSchema,
  type DirectoryAccessDenied,
  type DirectoryNotFound,
  type DirectoryPath,
  type FileContentHash,
  type FileSystemUnavailable,
  type PathNotDirectory,
  type ProjectDirectoryListing,
  type ProjectEntryMove,
  type ProjectFileSnapshot,
  type ProjectFileTextSnapshot,
  type ProjectFilesChange,
  type ProjectId,
  type ProjectNotFound,
  type ProjectStoreUnavailable,
  type RelativePath,
} from "@magnitudedev/acn-protocol"
import { Context, Effect, HashMap, Layer, Option, Stream, SynchronizedRef } from "effect"
import {
  FileSystemManager,
  type OpenedDirectory,
  type ResolveContainedPathError,
} from "./file-system-manager"
import { ProjectStore } from "./project-store"

const MAX_TEXT_BYTES = 5 * 1024 * 1024
const MAX_IMAGE_BYTES = 10 * 1024 * 1024
const imageTypes = new Map<string, "image/png" | "image/jpeg" | "image/gif" | "image/webp">([
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".gif", "image/gif"],
  [".webp", "image/webp"],
])

export type ProjectFileAccessError =
  | ProjectNotFound
  | ProjectStoreUnavailable
  | DirectoryNotFound
  | DirectoryAccessDenied
  | PathNotDirectory
  | FileSystemUnavailable
  | InvalidProjectFilePath
  | ProjectFileNotFound
  | ProjectFileAccessDenied

export interface ProjectFileManager {
  readonly listDirectory: (
    projectId: ProjectId,
    directory: RelativePath,
  ) => Effect.Effect<ProjectDirectoryListing, ProjectFileAccessError>
  readonly watchChanges: (
    projectId: ProjectId,
  ) => Stream.Stream<ProjectFilesChange, ProjectFileAccessError>
  readonly readFile: (
    projectId: ProjectId,
    path: RelativePath,
  ) => Effect.Effect<ProjectFileSnapshot, ProjectFileAccessError>
  readonly writeFile: (input: {
    readonly projectId: ProjectId
    readonly path: RelativePath
    readonly content: string
    readonly expectedContentHash: FileContentHash
  }) => Effect.Effect<
    ProjectFileTextSnapshot,
    ProjectFileAccessError | ProjectFileChanged | ProjectFileTooLarge
  >
  readonly deleteFile: (input: {
    readonly projectId: ProjectId
    readonly path: RelativePath
    readonly expectedContentHash: FileContentHash
  }) => Effect.Effect<void, ProjectFileAccessError | ProjectFileChanged>
  readonly moveEntry: (input: {
    readonly projectId: ProjectId
    readonly sourcePath: RelativePath
    readonly destinationDirectory: RelativePath
  }) => Effect.Effect<ProjectEntryMove, ProjectFileAccessError | ProjectFileAlreadyExists>
}

export const ProjectFileManager = Context.GenericTag<ProjectFileManager>("acn/ProjectFileManager")

const contentHash = (bytes: Uint8Array): Effect.Effect<FileContentHash> =>
  Effect.promise(() => crypto.subtle.digest("SHA-256", new Uint8Array(bytes))).pipe(
    Effect.map((digest) => FileContentHashSchema.make(
      Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(""),
    )),
  )

const newlineKind = (content: string): ProjectFileTextSnapshot["newline"] => {
  const crlf = content.includes("\r\n")
  const lf = content.replaceAll("\r\n", "").includes("\n")
  return crlf && lf ? "mixed" : crlf ? "crlf" : lf ? "lf" : "none"
}

const extensionOf = (path: RelativePath): string => {
  const name = path.slice(path.lastIndexOf("/") + 1)
  const dot = name.lastIndexOf(".")
  return dot <= 0 ? "" : name.slice(dot).toLowerCase()
}

const parentDirectory = (path: RelativePath): RelativePath => {
  const separator = path.lastIndexOf("/")
  return RelativePathSchema.make(separator === -1 ? "" : path.slice(0, separator))
}

const basenameOf = (path: RelativePath): string => path.slice(path.lastIndexOf("/") + 1)

const childPath = (directory: RelativePath, name: string): RelativePath =>
  RelativePathSchema.make(directory === "" ? name : `${directory}/${name}`)

/**
 * The single translation from the generic contained-filesystem vocabulary to
 * project-file errors. `accessKind` names what a FileAccessDenied means at
 * this call site: containment rejection ("symlink") or host permissions.
 */
const toProjectFileErrors = (
  path: RelativePath,
  accessKind: "symlink" | "permission_denied",
) => <A, R>(
  effect: Effect.Effect<A, ResolveContainedPathError, R>,
): Effect.Effect<
  A,
  ProjectFileNotFound | ProjectFileAccessDenied | FileSystemUnavailable,
  R
> =>
  Effect.catchTags(effect, {
    FileNotFound: () => Effect.fail(new ProjectFileNotFound({ path })),
    DirectoryNotFound: () => Effect.fail(new ProjectFileNotFound({ path })),
    FileAccessDenied: () => Effect.fail(new ProjectFileAccessDenied({ path, kind: accessKind })),
    PathNotFile: () => Effect.fail(new ProjectFileAccessDenied({ path, kind: "not_regular_file" })),
    PathNotDirectory: () => Effect.fail(new ProjectFileAccessDenied({ path, kind: "not_directory" })),
  })

export const ProjectFileManagerLive: Layer.Layer<
  ProjectFileManager,
  never,
  ProjectStore | FileSystemManager
> = Layer.effect(
  ProjectFileManager,
  Effect.gen(function* () {
    const store = yield* ProjectStore
    const fileSystem = yield* FileSystemManager
    const locks = yield* SynchronizedRef.make(HashMap.empty<DirectoryPath, Effect.Semaphore>())

    const lockFor = (cwd: DirectoryPath): Effect.Effect<Effect.Semaphore> =>
      SynchronizedRef.modifyEffect(locks, (map) =>
        Option.match(HashMap.get(map, cwd), {
          onSome: (existing) => Effect.succeed([existing, map] as const),
          onNone: () => Effect.makeSemaphore(1).pipe(
            Effect.map((created) => [created, HashMap.set(map, cwd, created)] as const),
          ),
        }))

    const openProject = (projectId: ProjectId) => store.get(projectId).pipe(
      Effect.flatMap((project) => fileSystem.openDirectory(project.cwd)),
    )

    /** Mutations coordinate per Project cwd; unrelated Projects never block each other. */
    const withProjectLock = <A, E, R>(
      projectId: ProjectId,
      body: (open: OpenedDirectory) => Effect.Effect<A, E, R>,
    ): Effect.Effect<A, E | ProjectFileAccessError, R> =>
      openProject(projectId).pipe(
        Effect.flatMap((open) => lockFor(open.cwd).pipe(
          Effect.flatMap((lock) => lock.withPermits(1)(body(open))),
        )),
      )

    const readSnapshot = Effect.fn("acn.project-file-manager.read-snapshot")(function* (
      open: OpenedDirectory,
      path: RelativePath,
    ) {
      if (path === "") {
        return yield* new InvalidProjectFilePath({ path, kind: "empty_path" })
      }
      const kind = yield* open.stat(path)
      switch (kind._tag) {
        case "missing": return yield* new ProjectFileNotFound({ path })
        case "symlink": return yield* new ProjectFileAccessDenied({ path, kind: "symlink" })
        case "directory":
        case "other":
          return yield* new ProjectFileAccessDenied({ path, kind: "not_regular_file" })
        case "file": break
      }
      const mediaType = imageTypes.get(extensionOf(path))
      if (mediaType !== undefined) {
        if (kind.size > MAX_IMAGE_BYTES) {
          return { _tag: "unsupported", path, reason: "too_large", size: kind.size } as const
        }
        const bytes = yield* open.readFile(path).pipe(
          toProjectFileErrors(path, "permission_denied"),
        )
        return {
          _tag: "image",
          path,
          mediaType,
          data: Buffer.from(bytes).toString("base64"),
          contentHash: yield* contentHash(bytes),
          size: bytes.byteLength,
        } as const
      }
      if (kind.size > MAX_TEXT_BYTES) {
        return { _tag: "unsupported", path, reason: "too_large", size: kind.size } as const
      }
      const bytes = yield* open.readFile(path).pipe(toProjectFileErrors(path, "permission_denied"))
      if (bytes.includes(0)) {
        return { _tag: "unsupported", path, reason: "binary", size: bytes.byteLength } as const
      }
      const decoded = Effect.try(() => new TextDecoder("utf-8", { fatal: true }).decode(bytes))
      const content = yield* decoded.pipe(Effect.option)
      if (Option.isNone(content)) {
        return { _tag: "unsupported", path, reason: "binary", size: bytes.byteLength } as const
      }
      return {
        _tag: "text",
        path,
        content: content.value,
        contentHash: yield* contentHash(bytes),
        size: bytes.byteLength,
        newline: newlineKind(content.value),
      } as const
    })

    return ProjectFileManager.of({
      listDirectory: Effect.fn("acn.project-file-manager.list-directory")(function* (
        projectId,
        directory,
      ) {
        const open = yield* openProject(projectId)
        const entries = yield* open.listDirectory(directory).pipe(
          toProjectFileErrors(directory, "symlink"),
        )
        return {
          projectId,
          directory,
          entries: entries
            .filter((entry) => entry.name !== ".git")
            .map((entry) => ({
              name: entry.name,
              path: childPath(directory, entry.name),
              kind: entry.kind,
              size: entry.size,
            }))
            .sort((left, right) => left.kind === right.kind
              ? left.name.localeCompare(right.name)
              : left.kind === "directory" ? -1 : 1),
        }
      }),

      watchChanges: (projectId) => Stream.unwrap(
        openProject(projectId).pipe(
          Effect.map((open) => {
            const root = RelativePathSchema.make("")
            return open.watch(root, { recursive: true }).pipe(
              Stream.filter((event) => event.path !== ".git" && !event.path.startsWith(".git/")),
              Stream.debounce("50 millis"),
              Stream.map(() => ({ projectId })),
              Stream.mapError((error): ProjectFileAccessError => {
                switch (error._tag) {
                  case "FileNotFound": return new ProjectFileNotFound({ path: root })
                  case "FileAccessDenied":
                    return new ProjectFileAccessDenied({ path: root, kind: "permission_denied" })
                  case "PathNotFile":
                    return new ProjectFileAccessDenied({ path: root, kind: "not_directory" })
                  default:
                    return error
                }
              }),
            )
          }),
        ),
      ),

      readFile: Effect.fn("acn.project-file-manager.read-file")(function* (projectId, path) {
        const open = yield* openProject(projectId)
        return yield* readSnapshot(open, path)
      }),

      writeFile: Effect.fn("acn.project-file-manager.write-file")(function* (input) {
        return yield* withProjectLock(input.projectId, (open) => Effect.gen(function* () {
          const current = yield* readSnapshot(open, input.path)
          if (current._tag !== "text") {
            return yield* new ProjectFileAccessDenied({ path: input.path, kind: "not_text" })
          }
          if (current.contentHash !== input.expectedContentHash) {
            return yield* new ProjectFileChanged({ path: input.path, current })
          }
          const bytes = new TextEncoder().encode(input.content)
          if (bytes.byteLength > MAX_TEXT_BYTES) {
            return yield* new ProjectFileTooLarge({ path: input.path, size: bytes.byteLength })
          }
          const mode = yield* open.stat(input.path)
          // Stale-write protection re-checks immediately before the rename commit.
          const guard = readSnapshot(open, input.path).pipe(
            Effect.flatMap((latest): Effect.Effect<
              void,
              ProjectFileChanged | ProjectFileAccessDenied
            > => latest._tag !== "text"
              ? Effect.fail(new ProjectFileAccessDenied({ path: input.path, kind: "not_text" }))
              : latest.contentHash === input.expectedContentHash
                ? Effect.void
                : Effect.fail(new ProjectFileChanged({ path: input.path, current: latest }))),
          )
          yield* Effect.catchTags(open.writeFileAtomic(input.path, bytes, {
            ...(mode._tag === "file" ? { mode: mode.mode } : {}),
            guard,
          }), {
            FileNotFound: () => Effect.fail(new ProjectFileNotFound({ path: input.path })),
            DirectoryNotFound: () => Effect.fail(new ProjectFileNotFound({ path: input.path })),
            FileAccessDenied: () => Effect.fail(new ProjectFileAccessDenied({
              path: input.path,
              kind: "permission_denied",
            })),
            PathNotFile: () => Effect.fail(new ProjectFileAccessDenied({
              path: input.path,
              kind: "not_regular_file",
            })),
            PathNotDirectory: () => Effect.fail(new ProjectFileAccessDenied({
              path: input.path,
              kind: "not_directory",
            })),
          })
          return {
            _tag: "text",
            path: input.path,
            content: input.content,
            contentHash: yield* contentHash(bytes),
            size: bytes.byteLength,
            newline: newlineKind(input.content),
          } as const
        }))
      }),

      deleteFile: Effect.fn("acn.project-file-manager.delete-file")(function* (input) {
        return yield* withProjectLock(input.projectId, (open) => Effect.gen(function* () {
          const current = yield* readSnapshot(open, input.path)
          if (current._tag === "unsupported") {
            return yield* new ProjectFileAccessDenied({
              path: input.path,
              kind: "not_regular_file",
            })
          }
          if (current.contentHash !== input.expectedContentHash) {
            if (current._tag === "text") {
              return yield* new ProjectFileChanged({ path: input.path, current })
            }
            return yield* new ProjectFileAccessDenied({
              path: input.path,
              kind: "changed_on_disk",
            })
          }
          yield* open.removeFile(input.path).pipe(
            toProjectFileErrors(input.path, "permission_denied"),
          )
        }))
      }),

      moveEntry: Effect.fn("acn.project-file-manager.move-entry")(function* (input) {
        return yield* withProjectLock(input.projectId, (open) => Effect.gen(function* () {
          if (input.sourcePath === "") {
            return yield* new InvalidProjectFilePath({
              path: input.sourcePath,
              kind: "root_immovable",
            })
          }
          if (parentDirectory(input.sourcePath) === input.destinationDirectory) {
            return yield* new ProjectFileAccessDenied({
              path: input.sourcePath,
              kind: "already_in_destination",
            })
          }
          const source = yield* open.stat(input.sourcePath)
          if (source._tag === "missing") {
            return yield* new ProjectFileNotFound({ path: input.sourcePath })
          }
          if (source._tag === "symlink") {
            return yield* new ProjectFileAccessDenied({ path: input.sourcePath, kind: "symlink" })
          }
          if (source._tag === "other") {
            return yield* new ProjectFileAccessDenied({
              path: input.sourcePath,
              kind: "not_regular_file",
            })
          }
          if (
            source._tag === "directory"
            && (input.destinationDirectory === input.sourcePath
              || input.destinationDirectory.startsWith(`${input.sourcePath}/`))
          ) {
            return yield* new ProjectFileAccessDenied({
              path: input.sourcePath,
              kind: "self_move",
            })
          }
          const destinationPath = childPath(
            input.destinationDirectory,
            basenameOf(input.sourcePath),
          )
          yield* Effect.catchTags(open.moveEntry(input.sourcePath, destinationPath), {
            FileAlreadyExists: () =>
              Effect.fail(new ProjectFileAlreadyExists({ path: destinationPath })),
            FileNotFound: () => Effect.fail(new ProjectFileNotFound({ path: input.sourcePath })),
            DirectoryNotFound: () =>
              Effect.fail(new ProjectFileNotFound({ path: input.destinationDirectory })),
            FileAccessDenied: () => Effect.fail(new ProjectFileAccessDenied({
              path: input.sourcePath,
              kind: "permission_denied",
            })),
            PathNotFile: () => Effect.fail(new ProjectFileAccessDenied({
              path: input.sourcePath,
              kind: "not_regular_file",
            })),
            PathNotDirectory: () => Effect.fail(new ProjectFileAccessDenied({
              path: input.destinationDirectory,
              kind: "not_directory",
            })),
          })
          return {
            sourcePath: input.sourcePath,
            destinationPath,
            kind: source._tag,
          }
        }))
      }),
    })
  }),
)
