import * as FileSystem from "@effect/platform/FileSystem";
import * as Path from "@effect/platform/Path";
import { SystemError, type PlatformError } from "@effect/platform/Error";
import { randomUUID } from "node:crypto";
import { Effect, Schema, SynchronizedRef } from "effect";

// =============================================================================
// Error types
// =============================================================================

export class JsonParseError extends Schema.TaggedError<JsonParseError>()(
  "JsonParseError",
  { path: Schema.String, message: Schema.String }
) {}

export class SchemaDecodeError extends Schema.TaggedError<SchemaDecodeError>()(
  "SchemaDecodeError",
  { path: Schema.String, message: Schema.String }
) {}

export class SchemaEncodeError extends Schema.TaggedError<SchemaEncodeError>()(
  "SchemaEncodeError",
  { path: Schema.String, message: Schema.String }
) {}

export class JsonLinesParseError extends Schema.TaggedError<JsonLinesParseError>()(
  "JsonLinesParseError",
  { path: Schema.String, line: Schema.Number, message: Schema.String }
) {}

export type JsonError = JsonParseError | SchemaDecodeError | SchemaEncodeError;
export type JsonLinesError = JsonLinesParseError | PlatformError;

// =============================================================================
// Types
// =============================================================================

export interface DirectoryEntry {
  readonly name: string;
  readonly path: string;
  readonly isFile: boolean;
  readonly isDirectory: boolean;
}

export interface StorageIo {
  readonly withPathLock: <A, E, R>(
    filePath: string,
    effect: Effect.Effect<A, E, R>
  ) => Effect.Effect<A, E, R>;
  readonly ensureDir: (dir: string) => Effect.Effect<void, PlatformError>;
  readonly ensureParentDir: (
    filePath: string
  ) => Effect.Effect<void, PlatformError>;
  readonly pathExists: (
    filePath: string
  ) => Effect.Effect<boolean, PlatformError>;
  readonly fileSize: (
    filePath: string
  ) => Effect.Effect<number | null, PlatformError>;
  readonly removeFileIfExists: (
    filePath: string
  ) => Effect.Effect<void, PlatformError>;
  readonly removeDirectoryIfExists: (
    dirPath: string
  ) => Effect.Effect<void, PlatformError>;
  readonly readTextFile: (
    filePath: string
  ) => Effect.Effect<string, PlatformError>;
  readonly writeTextFile: (
    filePath: string,
    content: string,
    options?: { readonly mode?: number; readonly appendNewline?: boolean }
  ) => Effect.Effect<void, PlatformError>;
  readonly readJsonFile: <T>(
    filePath: string,
    fallback?: T
  ) => Effect.Effect<T, JsonError | PlatformError>;
  readonly readJsonFileWithSchema: <A, I>(
    filePath: string,
    schema: Schema.Schema<A, I>,
    fallback?: A
  ) => Effect.Effect<A, JsonError | PlatformError>;
  readonly writeJsonFile: (
    filePath: string,
    value: unknown,
    options?: {
      readonly mode?: number;
      readonly spaces?: number;
      readonly appendNewline?: boolean;
    }
  ) => Effect.Effect<void, JsonParseError | PlatformError>;
  readonly writeJsonFileAtomic: (
    filePath: string,
    value: unknown,
    options?: {
      readonly mode?: number;
      readonly spaces?: number;
      readonly appendNewline?: boolean;
    }
  ) => Effect.Effect<void, JsonParseError | PlatformError>;
  readonly writeSecureJsonFile: (
    filePath: string,
    value: unknown
  ) => Effect.Effect<void, JsonParseError | PlatformError>;
  readonly readJsonLines: <T>(
    filePath: string
  ) => Effect.Effect<T[], JsonLinesError>;
  readonly appendJsonLines: <T>(
    filePath: string,
    entries: ReadonlyArray<T>
  ) => Effect.Effect<void, JsonLinesError>;
  readonly listDirectory: (
    dirPath: string
  ) => Effect.Effect<ReadonlyArray<DirectoryEntry>, PlatformError>;
}

// =============================================================================
// Factory
// =============================================================================

export function makeStorageIo(): Effect.Effect<
  StorageIo,
  never,
  FileSystem.FileSystem | Path.Path
> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const pathLocks = yield* SynchronizedRef.make(
      new Map<string, Effect.Semaphore>()
    );

    const withPathLock = <A, E, R>(
      filePath: string,
      effect: Effect.Effect<A, E, R>
    ): Effect.Effect<A, E, R> =>
      Effect.flatMap(
        SynchronizedRef.modifyEffect(pathLocks, (locks) => {
          const existing = locks.get(filePath);
          if (existing) return Effect.succeed([existing, locks] as const);
          return Effect.makeSemaphore(1).pipe(
            Effect.map((semaphore) => {
              const next = new Map(locks);
              next.set(filePath, semaphore);
              return [semaphore, next] as const;
            })
          );
        }),
        (semaphore) => semaphore.withPermits(1)(effect)
      );

    const ensureDir = (dir: string): Effect.Effect<void, PlatformError> =>
      fs.makeDirectory(dir, { recursive: true });

    const ensureParentDir = (
      filePath: string
    ): Effect.Effect<void, PlatformError> => ensureDir(path.dirname(filePath));

    const writeJsonAtomic = (
      filePath: string,
      value: unknown,
      options?: {
        readonly mode?: number;
        readonly spaces?: number;
        readonly appendNewline?: boolean;
      }
    ): Effect.Effect<void, JsonParseError | PlatformError> =>
      Effect.gen(function* () {
        let content = yield* Schema.encodeUnknown(
          Schema.parseJson({ space: options?.spaces ?? 2 })
        )(value).pipe(
          Effect.mapError(
            (e) => new JsonParseError({ path: filePath, message: String(e) })
          )
        );
        if (options?.appendNewline !== false && !content.endsWith("\n"))
          content += "\n";

        yield* ensureParentDir(filePath);
        const tmpPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
        yield* Effect.acquireUseRelease(
          Effect.succeed(tmpPath),
          (temporaryPath) =>
            Effect.gen(function* () {
              yield* fs.writeFileString(
                temporaryPath,
                content,
                options?.mode == null ? undefined : { mode: options.mode }
              );
              if (options?.mode !== undefined)
                yield* fs.chmod(temporaryPath, options.mode);
              yield* fs.rename(temporaryPath, filePath);
            }),
          (temporaryPath) =>
            fs
              .remove(temporaryPath, { force: true })
              .pipe(Effect.catchAll(() => Effect.void))
        );
      });

    return {
      withPathLock,
      ensureDir,
      ensureParentDir,
      pathExists: (filePath) => fs.exists(filePath),
      fileSize: (filePath) =>
        fs.stat(filePath).pipe(
          Effect.map((info) => Number(info.size)),
          Effect.catchTag("SystemError", (e: SystemError) =>
            e.reason === "NotFound" ? Effect.succeed(null) : Effect.fail(e)
          )
        ),
      removeFileIfExists: (filePath) => fs.remove(filePath, { force: true }),
      removeDirectoryIfExists: (dirPath) =>
        fs.remove(dirPath, { recursive: true, force: true }),
      readTextFile: (filePath) => fs.readFileString(filePath),
      writeTextFile: (filePath, content, options) =>
        Effect.gen(function* () {
          yield* ensureParentDir(filePath);
          let c = content;
          if (options?.appendNewline && !c.endsWith("\n")) c += "\n";
          yield* fs.writeFileString(
            filePath,
            c,
            options?.mode != null ? { mode: options.mode } : undefined
          );
        }),

      readJsonFile: <T>(filePath: string, fallback?: T) =>
        Effect.gen(function* () {
          const raw = yield* fs
            .readFileString(filePath)
            .pipe(
              Effect.catchTag("SystemError", (e: SystemError) =>
                e.reason === "NotFound" && fallback !== undefined
                  ? Effect.succeed(null)
                  : Effect.fail(e)
              )
            );
          if (raw === null) return fallback as T;
          return yield* Schema.decodeUnknown(Schema.parseJson())(raw).pipe(
            Effect.map((value) => value as T),
            Effect.mapError(
              (e) => new JsonParseError({ path: filePath, message: String(e) })
            )
          );
        }),

      readJsonFileWithSchema: <A, I>(
        filePath: string,
        schema: Schema.Schema<A, I>,
        fallback?: A
      ) =>
        Effect.gen(function* () {
          const raw = yield* fs
            .readFileString(filePath)
            .pipe(
              Effect.catchTag("SystemError", (e: SystemError) =>
                e.reason === "NotFound" && fallback !== undefined
                  ? Effect.succeed(null)
                  : Effect.fail(e)
              )
            );
          if (raw === null) return fallback as A;
          const json = yield* Schema.decodeUnknown(Schema.parseJson())(
            raw
          ).pipe(
            Effect.mapError(
              (e) => new JsonParseError({ path: filePath, message: String(e) })
            )
          );
          return yield* Schema.decodeUnknown(schema)(json).pipe(
            Effect.mapError(
              (e) =>
                new SchemaDecodeError({ path: filePath, message: String(e) })
            )
          );
        }),

      writeJsonFile: writeJsonAtomic,

      writeJsonFileAtomic: writeJsonAtomic,

      writeSecureJsonFile: (filePath, value) =>
        writeJsonAtomic(filePath, value, { mode: 0o600 }),

      readJsonLines: <T>(filePath: string) =>
        Effect.gen(function* () {
          const raw = yield* fs
            .readFileString(filePath)
            .pipe(
              Effect.catchTag("SystemError", (e: SystemError) =>
                e.reason === "NotFound" ? Effect.succeed("") : Effect.fail(e)
              )
            );
          const result: T[] = [];
          const lines = raw.split("\n");
          const hasTerminatedTail = raw.endsWith("\n");
          for (let i = 0; i < lines.length; i += 1) {
            const line = lines[i];
            if (line.trim() === "") continue;
            const decoded = yield* Effect.either(
              Schema.decodeUnknown(Schema.parseJson())(line).pipe(
                Effect.map((value) => value as T),
                Effect.mapError(
                  (e) =>
                    new JsonLinesParseError({
                      path: filePath,
                      line: i,
                      message: String(e),
                    })
                )
              )
            );
            if (decoded._tag === "Right") {
              result.push(decoded.right);
              continue;
            }
            const isTornFinalAppend =
              i === lines.length - 1 && !hasTerminatedTail;
            if (isTornFinalAppend) {
              yield* Effect.logWarning(
                "[storage] Ignoring torn final JSONL append"
              ).pipe(
                Effect.annotateLogs({
                  filePath,
                  line: i,
                  error: decoded.left.message,
                })
              );
              break;
            }
            return yield* decoded.left;
          }
          return result;
        }),

      appendJsonLines: <T>(filePath: string, entries: ReadonlyArray<T>) =>
        Effect.gen(function* () {
          if (entries.length === 0) return;
          yield* ensureParentDir(filePath);
          const encoded = yield* Effect.forEach(entries, (entry, index) =>
            Schema.encodeUnknown(Schema.parseJson())(entry).pipe(
              Effect.mapError(
                (error) =>
                  new JsonLinesParseError({
                    path: filePath,
                    line: index,
                    message: String(error),
                  })
              )
            )
          );
          yield* fs.writeFileString(filePath, encoded.join("\n") + "\n", {
            flag: "a",
          });
        }),

      listDirectory: (dirPath: string) =>
        Effect.gen(function* () {
          const names = yield* fs
            .readDirectory(dirPath)
            .pipe(
              Effect.catchTag("SystemError", (e: SystemError) =>
                e.reason === "NotFound"
                  ? Effect.succeed([] as ReadonlyArray<string>)
                  : Effect.fail(e)
              )
            );
          return yield* Effect.all(
            names.map((name) =>
              Effect.gen(function* () {
                const entryPath = path.join(dirPath, name);
                const info = yield* fs.stat(entryPath);
                return {
                  name,
                  path: entryPath,
                  isFile: info.type === "File",
                  isDirectory: info.type === "Directory",
                };
              })
            ),
            { concurrency: "unbounded" }
          );
        }),
    };
  });
}
