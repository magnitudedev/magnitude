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

export interface JsonLinesAppendResult {
  readonly previousRecordCount: number;
  readonly appendedRecordCount: number;
}

export interface RecoverableJsonLinesFile<T> {
  readonly readAndRepair: Effect.Effect<T[], JsonLinesError>;
  readonly append: (
    entries: ReadonlyArray<T>
  ) => Effect.Effect<JsonLinesAppendResult, JsonLinesError>;
}

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
  readonly recoverableJsonLines: <T>(
    filePath: string
  ) => RecoverableJsonLinesFile<T>;
  readonly appendDiagnosticJsonLines: <T>(
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
    const jsonLinesMetadata = yield* SynchronizedRef.make(
      new Map<string, { readonly byteLength: number; readonly recordCount: number }>()
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

    const readFileBytes = (filePath: string) =>
      fs.readFile(filePath).pipe(
        Effect.catchTag("SystemError", (e: SystemError) =>
          e.reason === "NotFound"
            ? Effect.succeed(new Uint8Array())
            : Effect.fail(e)
        )
      );

    const decodeUtf8 = (
      filePath: string,
      line: number,
      bytes: Uint8Array
    ): Effect.Effect<string, JsonLinesParseError> =>
      Effect.try({
        try: () => new TextDecoder("utf-8", { fatal: true }).decode(bytes),
        catch: (error) =>
          new JsonLinesParseError({
            path: filePath,
            line,
            message: `Invalid UTF-8: ${String(error)}`,
          }),
      });

    const decodeJsonLine = <T>(
      filePath: string,
      line: number,
      bytes: Uint8Array
    ): Effect.Effect<
      { readonly _tag: "blank" } | { readonly _tag: "value"; readonly value: T },
      JsonLinesParseError
    > =>
      Effect.gen(function* () {
        const text = yield* decodeUtf8(filePath, line, bytes);
        if (text.trim() === "") return { _tag: "blank" as const };
        return yield* Schema.decodeUnknown(Schema.parseJson())(text).pipe(
          Effect.map((value) => ({ _tag: "value" as const, value: value as T })),
          Effect.mapError(
            (error) =>
              new JsonLinesParseError({
                path: filePath,
                line,
                message: String(error),
              })
          )
        );
      });

    const setJsonLinesMetadata = (
      filePath: string,
      byteLength: number,
      recordCount: number
    ) =>
      SynchronizedRef.update(jsonLinesMetadata, (current) => {
        const next = new Map(current);
        next.set(filePath, { byteLength, recordCount });
        return next;
      });

    const removeJsonLinesMetadata = (targetPath: string, directory: boolean) =>
      SynchronizedRef.update(jsonLinesMetadata, (current) => {
        const normalized = path.resolve(targetPath);
        const prefix = normalized.endsWith(path.sep)
          ? normalized
          : `${normalized}${path.sep}`;
        const next = new Map(current);
        for (const filePath of current.keys()) {
          if (filePath === normalized || (directory && filePath.startsWith(prefix))) {
            next.delete(filePath);
          }
        }
        return next;
      });

    const readAndRepairJsonLinesUnlocked = <T>(filePath: string) =>
      Effect.gen(function* () {
        const raw = yield* readFileBytes(filePath);
        const result: T[] = [];
        let recordStart = 0;
        let line = 1;

        for (let offset = 0; offset < raw.byteLength; offset += 1) {
          if (raw[offset] !== 0x0a) continue;
          const decoded = yield* decodeJsonLine<T>(
            filePath,
            line,
            raw.subarray(recordStart, offset)
          );
          if (decoded._tag === "value") result.push(decoded.value);
          recordStart = offset + 1;
          line += 1;
        }

        let repairedByteLength = raw.byteLength;
        if (recordStart < raw.byteLength) {
          const tail = raw.subarray(recordStart);
          const decoded = yield* Effect.either(
            decodeJsonLine<T>(filePath, line, tail)
          );
          if (decoded._tag === "Right" && decoded.right._tag === "value") {
            yield* fs.writeFileString(filePath, "\n", { flag: "a" }).pipe(
              Effect.uninterruptible
            );
            repairedByteLength += 1;
            result.push(decoded.right.value);
            yield* Effect.logWarning(
              "[storage] Repaired JSONL record missing final newline"
            ).pipe(
              Effect.annotateLogs({
                filePath,
                originalByteLength: raw.byteLength,
                repairedByteLength,
                tail: "valid-json",
              })
            );
          } else {
            yield* fs.truncate(filePath, recordStart).pipe(
              Effect.uninterruptible
            );
            repairedByteLength = recordStart;
            yield* Effect.logWarning(
              "[storage] Discarded incomplete final JSONL record"
            ).pipe(
              Effect.annotateLogs({
                filePath,
                originalByteLength: raw.byteLength,
                repairedByteLength,
                discardedByteCount: raw.byteLength - repairedByteLength,
                tail: decoded._tag === "Right" ? "whitespace" : "invalid-json",
              })
            );
          }
        }

        yield* setJsonLinesMetadata(filePath, repairedByteLength, result.length);
        return result;
      });

    const encodeJsonLines = <T>(filePath: string, entries: ReadonlyArray<T>) =>
      Effect.forEach(entries, (entry, index) =>
        Schema.encodeUnknown(Schema.parseJson())(entry).pipe(
          Effect.mapError(
            (error) =>
              new JsonLinesParseError({
                path: filePath,
                line: index + 1,
                message: String(error),
              })
          )
        )
      );

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
      removeFileIfExists: (filePath) =>
        fs.remove(filePath, { force: true }).pipe(
          Effect.zipRight(removeJsonLinesMetadata(filePath, false))
        ),
      removeDirectoryIfExists: (dirPath) =>
        fs.remove(dirPath, { recursive: true, force: true }).pipe(
          Effect.zipRight(removeJsonLinesMetadata(dirPath, true))
        ),
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

      recoverableJsonLines: <T>(filePath: string): RecoverableJsonLinesFile<T> => {
        const normalized = path.resolve(filePath);
        return {
          readAndRepair: withPathLock(
            normalized,
            readAndRepairJsonLinesUnlocked<T>(normalized)
          ),
          append: (entries) =>
            Effect.gen(function* () {
              const encoded = yield* encodeJsonLines(normalized, entries);
              return yield* withPathLock(
                normalized,
                Effect.gen(function* () {
                  let metadata = (
                    yield* SynchronizedRef.get(jsonLinesMetadata)
                  ).get(normalized);
                  const currentByteLength = yield* fs.stat(normalized).pipe(
                    Effect.map((info) => Number(info.size)),
                    Effect.catchTag("SystemError", (e: SystemError) =>
                      e.reason === "NotFound"
                        ? Effect.succeed(0)
                        : Effect.fail(e)
                    )
                  );
                  if (!metadata || metadata.byteLength !== currentByteLength) {
                    const records =
                      yield* readAndRepairJsonLinesUnlocked<T>(normalized);
                    metadata = (
                      yield* SynchronizedRef.get(jsonLinesMetadata)
                    ).get(normalized) ?? {
                      byteLength: 0,
                      recordCount: records.length,
                    };
                  }

                  if (encoded.length === 0) {
                    return {
                      previousRecordCount: metadata.recordCount,
                      appendedRecordCount: 0,
                    } satisfies JsonLinesAppendResult;
                  }

                  yield* ensureParentDir(normalized);
                  const content = encoded.join("\n") + "\n";
                  const appendedByteLength =
                    new TextEncoder().encode(content).byteLength;
                  yield* Effect.uninterruptible(
                    fs
                      .writeFileString(normalized, content, { flag: "a" })
                      .pipe(
                        Effect.tapError(() =>
                          fs.truncate(normalized, metadata.byteLength).pipe(
                            Effect.orDie
                          )
                        ),
                        Effect.zipRight(
                          setJsonLinesMetadata(
                            normalized,
                            metadata.byteLength + appendedByteLength,
                            metadata.recordCount + encoded.length
                          )
                        )
                      )
                  );
                  return {
                    previousRecordCount: metadata.recordCount,
                    appendedRecordCount: encoded.length,
                  } satisfies JsonLinesAppendResult;
                })
              );
            }),
        };
      },

      appendDiagnosticJsonLines: <T>(filePath: string, entries: ReadonlyArray<T>) =>
        Effect.gen(function* () {
          if (entries.length === 0) return;
          yield* ensureParentDir(filePath);
          const encoded = yield* encodeJsonLines(filePath, entries);
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
