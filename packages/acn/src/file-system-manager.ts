import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import type { PlatformError } from "@effect/platform/Error"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import {
  AbsolutePathSchema,
  DirectoryAccessDenied,
  DirectoryNotFound,
  DirectoryPathSchema,
  FileAccessDenied,
  FileAlreadyExists,
  FileNotFound,
  FileSystemUnavailable,
  InvalidDirectoryPath,
  PathNotDirectory,
  PathNotFile,
  RelativePathSchema,
  RevealFailed,
  RevealUnsupported,
  type AbsolutePath,
  type DirectoryInspection,
  type DirectoryPath,
  type RelativePath,
} from "@magnitudedev/acn-protocol"
import { Array as Arr, Context, Effect, Layer, Option, Ref, Schedule, Schema, Stream } from "effect"

/** Host observation of an arbitrary absolute path (follows symlinks). */
export type PathInspection =
  | { readonly _tag: "file"; readonly size: number }
  | { readonly _tag: "directory" }
  | { readonly _tag: "other" }
  | { readonly _tag: "missing" }
  | { readonly _tag: "access_denied" }
  | { readonly _tag: "unavailable" }

/** Symlink-aware classification of a contained path (never follows symlinks). */
export type PathMetadata =
  | { readonly _tag: "file"; readonly size: number; readonly mode: number }
  | { readonly _tag: "directory" }
  | { readonly _tag: "symlink" }
  | { readonly _tag: "other" }
  | { readonly _tag: "missing" }

export interface DirectoryEntry {
  readonly name: string
  readonly kind: "file" | "directory"
  readonly size: Option.Option<number>
}

export type FileWatchEvent = { readonly _tag: "created" | "changed" | "removed" }

export type SubdirectoryListing =
  | { readonly _tag: "listed"; readonly names: ReadonlyArray<string> }
  | { readonly _tag: "unavailable" }

export type PathRequirement = "file" | "directory" | "parent_directory" | "none"

export type ResolveContainedPathError =
  | FileNotFound
  | DirectoryNotFound
  | FileAccessDenied
  | PathNotFile
  | PathNotDirectory
  | FileSystemUnavailable

/**
 * One verified directory root. The single containment implementation in the
 * ACN: every operation takes a `RelativePath` (the brand already excludes
 * absolute paths, traversal, NULs, and separator ambiguity) and resolves it
 * internally. Symlinks are rejected uniformly inside an opened root.
 */
export interface OpenedDirectory {
  readonly cwd: DirectoryPath
  readonly resolve: (
    path: RelativePath,
    requirement: PathRequirement,
  ) => Effect.Effect<AbsolutePath, ResolveContainedPathError>
  readonly stat: (path: RelativePath) => Effect.Effect<PathMetadata, FileSystemUnavailable>
  readonly listDirectory: (
    path: RelativePath,
  ) => Effect.Effect<ReadonlyArray<DirectoryEntry>, ResolveContainedPathError>
  /**
   * Bounded breadth-first file discovery: skips dot entries, never descends
   * symlinks, and skips unreadable subtrees.
   */
  readonly walkFiles: (options: {
    readonly limit: number
  }) => Effect.Effect<ReadonlyArray<RelativePath>>
  readonly readFile: (
    path: RelativePath,
  ) => Effect.Effect<Uint8Array, ResolveContainedPathError>
  /**
   * Temp-sibling atomic write: exclusive temp with the requested mode, write,
   * fsync, run `guard`, rename over the target. Guard failure or any error
   * removes the temp and writes nothing.
   */
  readonly writeFileAtomic: <E = never>(
    path: RelativePath,
    bytes: Uint8Array,
    options?: {
      readonly mode?: number
      readonly guard?: Effect.Effect<void, E>
    },
  ) => Effect.Effect<void, E | ResolveContainedPathError>
  readonly removeFile: (path: RelativePath) => Effect.Effect<void, ResolveContainedPathError>
  /** `destination` is the complete target path. Never overwrites. */
  readonly moveEntry: (
    source: RelativePath,
    destination: RelativePath,
  ) => Effect.Effect<void, ResolveContainedPathError | FileAlreadyExists>
  readonly watch: (
    path: RelativePath,
    options?: { readonly recursive?: boolean },
  ) => Stream.Stream<{ readonly path: string }, ResolveContainedPathError>
}

export interface FileSystemManager {
  /** Lexical normalization of a directory input. No filesystem I/O. */
  readonly normalizeDirectory: (path: string) => Effect.Effect<DirectoryPath, InvalidDirectoryPath>
  /**
   * Lexical resolution for session file operations: optionally expands
   * "~"/"~/...", keeps absolute input, resolves relative input against base.
   */
  readonly resolveHostPath: (
    base: DirectoryPath,
    path: string,
    options?: { readonly expandHome?: boolean },
  ) => Effect.Effect<AbsolutePath, InvalidDirectoryPath>
  readonly inspectDirectory: (path: DirectoryPath) => Effect.Effect<DirectoryInspection>
  readonly inspectPath: (path: AbsolutePath) => Effect.Effect<PathInspection>
  readonly openDirectory: (path: DirectoryPath) => Effect.Effect<
    OpenedDirectory,
    DirectoryNotFound | DirectoryAccessDenied | PathNotDirectory | FileSystemUnavailable
  >
  readonly readHostFile: (path: AbsolutePath) => Effect.Effect<
    Uint8Array,
    FileNotFound | FileAccessDenied | PathNotFile | FileSystemUnavailable
  >
  /**
   * Native watcher with a per-subscription 500ms polling fallback — a
   * client-demanded stream, not an ambient poller.
   */
  readonly watchHostFile: (path: AbsolutePath) => Stream.Stream<FileWatchEvent>
  readonly listHostSubdirectories: (parent: AbsolutePath) => Effect.Effect<SubdirectoryListing>
  readonly revealDirectory: (
    path: DirectoryPath,
  ) => Effect.Effect<void, RevealUnsupported | RevealFailed>
}

export const FileSystemManager = Context.GenericTag<FileSystemManager>("acn/FileSystemManager")

const decodeDirectoryPath = Schema.decodeUnknown(DirectoryPathSchema)

export const FileSystemManagerLive: Layer.Layer<
  FileSystemManager,
  never,
  FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
> = Layer.effect(
  FileSystemManager,
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const paths = yield* Path.Path
    const executor = yield* CommandExecutor.CommandExecutor
    const homeDirectory = process.env.HOME ?? process.env.USERPROFILE ?? null

    const unavailable = (path: string) => (_error: PlatformError) =>
      new FileSystemUnavailable({ path })

    const fileErrors = (path: string) => <A, R>(
      effect: Effect.Effect<A, PlatformError, R>,
    ): Effect.Effect<A, FileNotFound | FileAccessDenied | FileSystemUnavailable, R> =>
      Effect.catchTags(effect, {
        SystemError: (error) =>
          error.reason === "NotFound"
            ? Effect.fail(new FileNotFound({ path }))
            : error.reason === "PermissionDenied"
              ? Effect.fail(new FileAccessDenied({ path }))
              : Effect.fail(unavailable(path)(error)),
        BadArgument: (error) => Effect.fail(unavailable(path)(error)),
      })

    /** Symlink-aware kind classification: readLink success means symlink. */
    const classifyContained = (absolute: string): Effect.Effect<PathMetadata, PlatformError> =>
      fs.readLink(absolute).pipe(
        Effect.map((): PathMetadata => ({ _tag: "symlink" })),
        Effect.catchAll(() => fs.stat(absolute).pipe(
          Effect.map((info): PathMetadata => info.type === "File"
            ? { _tag: "file", size: Number(info.size), mode: info.mode }
            : info.type === "Directory"
              ? { _tag: "directory" }
              : { _tag: "other" }),
          Effect.catchTag("SystemError", (error) => error.reason === "NotFound"
            ? Effect.succeed<PathMetadata>({ _tag: "missing" })
            : Effect.fail(error)),
        )),
      )

    const inspectPath = (path: AbsolutePath): Effect.Effect<PathInspection> =>
      fs.stat(path).pipe(
        Effect.map((info): PathInspection => info.type === "File"
          ? { _tag: "file", size: Number(info.size) }
          : info.type === "Directory"
            ? { _tag: "directory" }
            : { _tag: "other" }),
        Effect.catchTag("SystemError", (error) => Effect.succeed<PathInspection>(
          error.reason === "NotFound"
            ? { _tag: "missing" }
            : error.reason === "PermissionDenied"
              ? { _tag: "access_denied" }
              : { _tag: "unavailable" },
        )),
        Effect.catchTag("BadArgument", () => Effect.succeed<PathInspection>({ _tag: "unavailable" })),
      )

    const inspectDirectory = (cwd: DirectoryPath): Effect.Effect<DirectoryInspection> =>
      inspectPath(AbsolutePathSchema.make(cwd)).pipe(
        Effect.map((inspection): DirectoryInspection => {
          switch (inspection._tag) {
            case "directory": return { _tag: "available" }
            case "missing": return { _tag: "missing" }
            case "access_denied": return { _tag: "access_denied" }
            case "unavailable": return { _tag: "unavailable" }
            case "file":
            case "other":
              return { _tag: "not_directory" }
          }
        }),
      )

    const expandHome = (value: string): string => {
      if (homeDirectory === null) return value
      if (value === "~") return homeDirectory
      if (value.startsWith("~/")) return paths.join(homeDirectory, value.slice(2))
      return value
    }

    const resolveHostPath = (
      base: DirectoryPath,
      value: string,
      options?: { readonly expandHome?: boolean },
    ): Effect.Effect<AbsolutePath, InvalidDirectoryPath> => {
      const expanded = options?.expandHome === true ? expandHome(value) : value
      const resolved = paths.isAbsolute(expanded)
        ? paths.resolve(expanded)
        : paths.resolve(base, expanded)
      return Schema.decodeUnknown(AbsolutePathSchema)(resolved).pipe(
        Effect.mapError(() => new InvalidDirectoryPath({ path: value })),
      )
    }

    const makeOpenedDirectory = (cwd: DirectoryPath): OpenedDirectory => {
      const stat = (path: RelativePath): Effect.Effect<PathMetadata, FileSystemUnavailable> =>
        classifyContained(paths.join(cwd, path)).pipe(
          Effect.mapError((error) => unavailable(path)(error)),
        )

      const resolve = Effect.fn("acn.file-system-manager.resolve-contained")(function* (
        path: RelativePath,
        requirement: PathRequirement,
      ) {
        // The root itself was verified by openDirectory through a
        // symlink-following observation; containment governs entries beneath it.
        if (path === "") {
          const root = AbsolutePathSchema.make(cwd)
          const inspection = yield* inspectPath(root)
          switch (inspection._tag) {
            case "directory": return root
            case "missing": return yield* new DirectoryNotFound({ path })
            case "access_denied": return yield* new FileAccessDenied({ path })
            case "unavailable": return yield* new FileSystemUnavailable({ path })
            case "file":
            case "other":
              return yield* new PathNotDirectory({ path })
          }
        }

        const absolute = paths.join(cwd, path)
        const segments = path.split("/")

        // Ancestor walk: an existing symlink prefix escapes the root.
        let prefix = cwd as string
        for (const segment of segments.slice(0, -1)) {
          prefix = paths.join(prefix, segment)
          const kind = yield* classifyContained(prefix).pipe(
            Effect.mapError((error) => unavailable(path)(error)),
          )
          if (kind._tag === "symlink") return yield* new FileAccessDenied({ path })
          if (kind._tag === "missing") break
        }

        const target = yield* classifyContained(absolute).pipe(
          Effect.mapError((error) => unavailable(path)(error)),
        )
        if (target._tag === "symlink") return yield* new FileAccessDenied({ path })
        switch (requirement) {
          case "file": {
            if (target._tag === "missing") return yield* new FileNotFound({ path })
            if (target._tag !== "file") return yield* new PathNotFile({ path })
            break
          }
          case "directory": {
            if (target._tag === "missing") return yield* new DirectoryNotFound({ path })
            if (target._tag !== "directory") return yield* new PathNotDirectory({ path })
            break
          }
          case "parent_directory": {
            const parent = yield* classifyContained(paths.dirname(absolute)).pipe(
              Effect.mapError((error) => unavailable(path)(error)),
            )
            if (parent._tag === "missing") return yield* new DirectoryNotFound({ path })
            if (parent._tag !== "directory") return yield* new PathNotDirectory({ path })
            break
          }
          case "none":
            break
        }
        return AbsolutePathSchema.make(absolute)
      })

      const listDirectory = Effect.fn("acn.file-system-manager.list-directory")(function* (
        path: RelativePath,
      ) {
        const absolute = yield* resolve(path, "directory")
        const names = yield* Effect.catchTags(fs.readDirectory(absolute), {
          SystemError: (error) =>
            error.reason === "PermissionDenied"
              ? Effect.fail(new FileAccessDenied({ path }))
              : Effect.fail(unavailable(path)(error)),
          BadArgument: (error) => Effect.fail(unavailable(path)(error)),
        })
        const entries = yield* Effect.forEach(names, (name) =>
          classifyContained(paths.join(absolute, name)).pipe(
            Effect.map((kind): Option.Option<DirectoryEntry> => kind._tag === "file"
              ? Option.some({ name, kind: "file", size: Option.some(kind.size) })
              : kind._tag === "directory"
                ? Option.some({ name, kind: "directory", size: Option.none() })
                : Option.none()),
            Effect.catchAll(() => Effect.succeed(Option.none<DirectoryEntry>())),
          ), { concurrency: 16 })
        return Arr.getSomes(entries)
      })

      const walkFiles = Effect.fn("acn.file-system-manager.walk-files")(function* (options: {
        readonly limit: number
      }) {
        const found: RelativePath[] = []
        const queue: RelativePath[] = [RelativePathSchema.make("")]
        while (queue.length > 0 && found.length < options.limit) {
          const directory = queue.shift()
          if (directory === undefined) break
          const entries = yield* listDirectory(directory).pipe(
            Effect.catchAll(() => Effect.succeed<ReadonlyArray<DirectoryEntry>>([])),
          )
          for (const entry of entries) {
            if (entry.name.startsWith(".")) continue
            const child = RelativePathSchema.make(
              directory === "" ? entry.name : `${directory}/${entry.name}`,
            )
            if (entry.kind === "directory") queue.push(child)
            else {
              found.push(child)
              if (found.length >= options.limit) break
            }
          }
        }
        return found
      })

      const readFile = Effect.fn("acn.file-system-manager.read-contained-file")(function* (
        path: RelativePath,
      ) {
        const absolute = yield* resolve(path, "file")
        return yield* fs.readFile(absolute).pipe(fileErrors(path))
      })

      const writeFileAtomic = <E = never>(
        path: RelativePath,
        bytes: Uint8Array,
        options?: { readonly mode?: number; readonly guard?: Effect.Effect<void, E> },
      ): Effect.Effect<void, E | ResolveContainedPathError> =>
        Effect.gen(function* () {
          const absolute = yield* resolve(path, "parent_directory")
          const temp = paths.join(
            paths.dirname(absolute),
            `.magnitude-${paths.basename(absolute)}-${crypto.randomUUID()}.tmp`,
          )
          const stage = Effect.scoped(fs.open(temp, {
            flag: "wx",
            ...(options?.mode !== undefined ? { mode: options.mode } : {}),
          }).pipe(
            Effect.flatMap((file) => file.writeAll(bytes).pipe(Effect.zipRight(file.sync))),
          )).pipe(fileErrors(path))
          const commit = fs.rename(temp, absolute).pipe(fileErrors(path))
          yield* stage.pipe(
            Effect.zipRight(options?.guard ?? Effect.void),
            Effect.zipRight(commit),
            Effect.ensuring(fs.remove(temp, { force: true }).pipe(Effect.ignore)),
          )
        })

      const removeFile = Effect.fn("acn.file-system-manager.remove-contained-file")(function* (
        path: RelativePath,
      ) {
        const absolute = yield* resolve(path, "file")
        yield* fs.remove(absolute).pipe(fileErrors(path))
      })

      const moveEntry = Effect.fn("acn.file-system-manager.move-contained-entry")(function* (
        source: RelativePath,
        destination: RelativePath,
      ) {
        const sourceAbsolute = yield* resolve(source, "none")
        const sourceKind = yield* stat(source)
        if (sourceKind._tag === "missing") return yield* new FileNotFound({ path: source })
        if (sourceKind._tag !== "file" && sourceKind._tag !== "directory") {
          return yield* new FileAccessDenied({ path: source })
        }
        const destinationAbsolute = yield* resolve(destination, "parent_directory")
        yield* Effect.uninterruptible(Effect.gen(function* () {
          const existing = yield* stat(destination)
          if (existing._tag !== "missing") {
            return yield* new FileAlreadyExists({ path: destination })
          }
          yield* fs.rename(sourceAbsolute, destinationAbsolute).pipe(fileErrors(source))
        }))
      })

      const watch = (
        path: RelativePath,
        options?: { readonly recursive?: boolean },
      ): Stream.Stream<{ readonly path: string }, ResolveContainedPathError> =>
        Stream.unwrap(resolve(path, "directory").pipe(
          Effect.map((absolute) => fs.watch(absolute, {
            recursive: options?.recursive === true,
          }).pipe(
            Stream.map((event) => ({ path: event.path })),
            Stream.mapError((error) => unavailable(path)(error)),
          )),
        ))

      return {
        cwd,
        resolve,
        stat,
        listDirectory,
        walkFiles,
        readFile,
        writeFileAtomic,
        removeFile,
        moveEntry,
        watch,
      }
    }

    const watchHostFile = (path: AbsolutePath): Stream.Stream<FileWatchEvent> => {
      const native = fs.watch(path).pipe(
        Stream.map((event): FileWatchEvent => event._tag === "Create"
          ? { _tag: "created" }
          : event._tag === "Update"
            ? { _tag: "changed" }
            : { _tag: "removed" }),
      )
      const polling = Stream.unwrap(Effect.gen(function* () {
        const previous = yield* Ref.make(Option.none<{
          readonly size: number
          readonly mtimeMs: number
        }>())
        return Stream.repeatEffectWithSchedule(
          Effect.gen(function* () {
            const info = yield* fs.stat(path).pipe(Effect.option)
            if (Option.isNone(info)) {
              const before = yield* Ref.getAndSet(previous, Option.none())
              return Option.map(before, (): FileWatchEvent => ({ _tag: "removed" }))
            }
            const current = {
              size: Number(info.value.size),
              mtimeMs: Option.match(info.value.mtime, {
                onNone: () => 0,
                onSome: (mtime) => mtime.getTime(),
              }),
            }
            const before = yield* Ref.getAndSet(previous, Option.some(current))
            return Option.match(before, {
              onNone: () => Option.some<FileWatchEvent>({ _tag: "created" }),
              onSome: (last) => last.size !== current.size || last.mtimeMs !== current.mtimeMs
                ? Option.some<FileWatchEvent>({ _tag: "changed" })
                : Option.none<FileWatchEvent>(),
            })
          }),
          Schedule.spaced("500 millis"),
        ).pipe(
          Stream.filterMap((event) => event),
        )
      }))
      return native.pipe(Stream.catchAll(() => polling))
    }

    const listHostSubdirectories = Effect.fn("acn.file-system-manager.list-host-subdirectories")(
      function* (parent: AbsolutePath) {
        const names = yield* fs.readDirectory(parent).pipe(Effect.option)
        if (Option.isNone(names)) return { _tag: "unavailable" } as const
        const kept = yield* Effect.forEach(
          names.value.filter((name) => !name.startsWith(".")),
          (name) => inspectPath(AbsolutePathSchema.make(paths.join(parent, name))).pipe(
            Effect.map((inspection) => inspection._tag === "directory"
              ? Option.some(name)
              : Option.none<string>()),
          ),
          { concurrency: 16 },
        )
        return { _tag: "listed", names: Arr.getSomes(kept) } as const
      },
    )

    return FileSystemManager.of({
      normalizeDirectory: (value) => {
        if (value.trim() === "") {
          return Effect.fail(new InvalidDirectoryPath({ path: value }))
        }
        return decodeDirectoryPath(paths.resolve(value)).pipe(
          Effect.mapError(() => new InvalidDirectoryPath({ path: value })),
        )
      },
      resolveHostPath,
      inspectDirectory,
      inspectPath,
      openDirectory: Effect.fn("acn.file-system-manager.open-directory")(function* (cwd) {
        const inspection = yield* inspectDirectory(cwd)
        switch (inspection._tag) {
          case "available": return makeOpenedDirectory(cwd)
          case "missing": return yield* new DirectoryNotFound({ path: cwd })
          case "access_denied": return yield* new DirectoryAccessDenied({ path: cwd })
          case "not_directory": return yield* new PathNotDirectory({ path: cwd })
          case "unavailable": return yield* new FileSystemUnavailable({ path: cwd })
        }
      }),
      readHostFile: Effect.fn("acn.file-system-manager.read-host-file")(function* (path) {
        const inspection = yield* inspectPath(path)
        switch (inspection._tag) {
          case "file": break
          case "missing": return yield* new FileNotFound({ path })
          case "access_denied": return yield* new FileAccessDenied({ path })
          case "directory":
          case "other":
            return yield* new PathNotFile({ path })
          case "unavailable": return yield* new FileSystemUnavailable({ path })
        }
        return yield* fs.readFile(path).pipe(fileErrors(path))
      }),
      watchHostFile,
      listHostSubdirectories,
      revealDirectory: Effect.fn("acn.file-system-manager.reveal-directory")(function* (cwd) {
        const command = process.platform === "darwin"
          ? Command.make("open", "-R", cwd)
          : process.platform === "linux"
            ? Command.make("xdg-open", cwd)
            : null
        if (command === null) return yield* new RevealUnsupported()
        const exitCode = yield* Effect.scoped(
          executor.start(command).pipe(Effect.flatMap((process) => process.exitCode)),
        ).pipe(Effect.mapError(() => new RevealFailed({ path: cwd })))
        if (exitCode !== 0) return yield* new RevealFailed({ path: cwd })
      }),
    })
  }),
)
