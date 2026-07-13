import * as FileSystem from "@effect/platform/FileSystem";
import * as Path from "@effect/platform/Path";
import { type PlatformError } from "@effect/platform/Error";
import { Effect, Schema } from "effect";
import { generateSortableId } from "@magnitudedev/generate-id";
import { SCRATCHPAD_SUBDIRS } from "@magnitudedev/scratchpad";

import {
  makeStorageIo,
  SchemaDecodeError,
  type JsonError,
  type JsonLinesError,
} from "../io/storage";
import { readStructuredFile } from "../io/structured-file";
import { GlobalStorage } from "../services";
import { Version } from "../services/version";
import {
  makeStoredSessionMetaSchema,
  MemoryExtractionJobRecordSchema,
  type CwdIndex,
  type MemoryExtractionJobRecord,
  type SessionDiscoveryOptions,
  type StoredSessionMeta,
} from "../types/session";
import type {
  SessionStorageShape,
  StoredAddressedEntry,
  StoredAddressedEntryStats,
  StoredEventCursor,
} from "./contracts";

const CwdIndexSchema = Schema.Struct({
  cwd: Schema.String,
  sessionIds: Schema.Array(Schema.String),
});

const TIMESTAMP_SESSION_ID_RE = /^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$/;
const SORTABLE_SESSION_ID_RE = /^[0-9a-z]{6,12}$/;

export function makeSessionStorage(): Effect.Effect<
  SessionStorageShape,
  never,
  FileSystem.FileSystem | Path.Path | Version | GlobalStorage
> {
  return Effect.gen(function* () {
    const io = yield* makeStorageIo();
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const version = yield* Version;
    const globalStorage = yield* GlobalStorage;

    const versionStr = version.getVersion();
    const metaSchema = makeStoredSessionMetaSchema(versionStr);
    const g = globalStorage.paths;

    // ------------------------------------------------------------------
    // Private closure helpers
    // ------------------------------------------------------------------

    const readMetaHelper = (
      sessionId: string
    ): Effect.Effect<StoredSessionMeta | null, PlatformError | JsonError> =>
      Effect.gen(function* () {
        const filePath = g.sessionMetaFile(sessionId);
        const result = yield* readStructuredFile(filePath, metaSchema).pipe(
          Effect.provideService(FileSystem.FileSystem, fs)
        );
        if (result._tag === "Missing") return null;
        if (result._tag === "Invalid") {
          return yield* Effect.fail(
            new SchemaDecodeError({ path: filePath, message: result.error.reason })
          );
        }
        return result.value;
      });

    const upsertCwdIndex = (
      meta: StoredSessionMeta
    ): Effect.Effect<void, PlatformError | JsonError> =>
      io.withPathLock(
        g.cwdIndexFile(path.resolve(meta.workingDirectory)),
        Effect.gen(function* () {
          const cwd = path.resolve(meta.workingDirectory);
          const filePath = g.cwdIndexFile(cwd);
          const result = yield* readStructuredFile(filePath, CwdIndexSchema).pipe(
            Effect.provideService(FileSystem.FileSystem, fs)
          );
          const ids = result._tag === "Present" ? result.value.sessionIds : [];
          if (!ids.includes(meta.sessionId)) {
            ids.unshift(meta.sessionId);
          }
          yield* io.ensureDir(g.indexRoot);
          yield* io.writeJsonFile(filePath, { cwd, sessionIds: ids });
        })
      );

    const removeCwdIndexEntry = (
      cwd: string,
      sessionId: string
    ): Effect.Effect<void, PlatformError | JsonError> =>
      io.withPathLock(
        g.cwdIndexFile(path.resolve(cwd)),
        Effect.gen(function* () {
          const resolved = path.resolve(cwd);
          const filePath = g.cwdIndexFile(resolved);
          const result = yield* readStructuredFile(filePath, CwdIndexSchema).pipe(
            Effect.provideService(FileSystem.FileSystem, fs)
          );
          if (result._tag !== "Present") return;
          const ids = result.value.sessionIds.filter((id) => id !== sessionId);
          if (ids.length === 0) {
            yield* io.removeFileIfExists(filePath);
          } else {
            yield* io.writeJsonFile(filePath, { ...result.value, sessionIds: ids });
          }
        })
      );

    const getPendingMemoryJobPath = (
      input: { readonly jobId: string } | { readonly filePath: string }
    ): string =>
      "jobId" in input ? g.pendingMemoryJobFile(input.jobId) : input.filePath;

    // ------------------------------------------------------------------
    // Service
    // ------------------------------------------------------------------

    return {
      paths: {
        root: g.root,
        sessionsRoot: g.sessionsRoot,
        pendingMemoryExtractionRoot: g.pendingMemoryExtractionRoot,
        sessionDir: g.sessionDir,
        sessionMetaFile: g.sessionMetaFile,
        sessionEventsFile: g.sessionEventsFile,
        sessionProjectionSnapshotFile: g.sessionProjectionSnapshotFile,
        sessionAddressedEntriesRoot: g.sessionAddressedEntriesRoot,
        sessionAddressedEntryFile: g.sessionAddressedEntryFile,
        sessionLogFile: g.sessionLogFile,
        sessionScratchpad: g.sessionScratchpad,
        pendingMemoryJobFile: g.pendingMemoryJobFile,
      },

      createTimestampSessionId: () => generateSortableId(),

      listSessionIds: (options) =>
        Effect.gen(function* () {
          const timestampOnly = options?.timestampOnly ?? true;
          const entries = yield* io.listDirectory(g.sessionsRoot);
          return entries
            .filter((entry) => entry.isDirectory)
            .map((entry) => entry.name)
            .filter(
              (name) =>
                !timestampOnly ||
                TIMESTAMP_SESSION_ID_RE.test(name) ||
                SORTABLE_SESSION_ID_RE.test(name)
            )
            .sort()
            .reverse();
        }),

      findLatestSessionId: (options) =>
        Effect.gen(function* () {
          const timestampOnly = options?.timestampOnly ?? true;
          const entries = yield* io.listDirectory(g.sessionsRoot);
          const ids = entries
            .filter((entry) => entry.isDirectory)
            .map((entry) => entry.name)
            .filter(
              (name) =>
                !timestampOnly ||
                TIMESTAMP_SESSION_ID_RE.test(name) ||
                SORTABLE_SESSION_ID_RE.test(name)
            )
            .sort()
            .reverse();
          return ids[0] ?? null;
        }),

      readMeta: readMetaHelper,

      writeMeta: (sessionId, meta) =>
        io.withPathLock(
          g.sessionMetaFile(sessionId),
          Effect.gen(function* () {
            yield* io.ensureDir(g.sessionDir(sessionId));
            yield* io.writeJsonFile(g.sessionMetaFile(sessionId), meta);
            yield* upsertCwdIndex(meta);
          })
        ),

      updateMeta: (sessionId, updater) =>
        io.withPathLock(
          g.sessionMetaFile(sessionId),
          Effect.gen(function* () {
            const current = yield* readMetaHelper(sessionId);
            const next = updater(current);
            if (current && current.workingDirectory !== next.workingDirectory) {
              yield* removeCwdIndexEntry(current.workingDirectory, sessionId);
            }
            yield* io.ensureDir(g.sessionDir(sessionId));
            yield* io.writeJsonFile(g.sessionMetaFile(sessionId), next);
            yield* upsertCwdIndex(next);
            return next;
          })
        ),

      deleteSession: (sessionId) =>
        io.withPathLock(
          g.sessionDir(sessionId),
          Effect.gen(function* () {
            const meta = yield* readMetaHelper(sessionId);
            yield* io.removeDirectoryIfExists(g.sessionDir(sessionId));
            if (meta) {
              yield* removeCwdIndexEntry(meta.workingDirectory, sessionId);
            }
          })
        ),

      readCwdIndex: (cwd) =>
        Effect.gen(function* () {
          const filePath = g.cwdIndexFile(path.resolve(cwd));
          const result = yield* readStructuredFile(filePath, CwdIndexSchema).pipe(
            Effect.provideService(FileSystem.FileSystem, fs)
          );
          if (result._tag === "Missing") return null;
          if (result._tag === "Invalid") {
            return yield* Effect.fail(
              new SchemaDecodeError({ path: filePath, message: result.error.reason })
            );
          }
          return result.value;
        }),

      writeCwdIndex: (cwd, sessionIds) =>
        io.withPathLock(
          g.cwdIndexFile(path.resolve(cwd)),
          Effect.gen(function* () {
            const resolved = path.resolve(cwd);
            yield* io.ensureDir(g.indexRoot);
            yield* io.writeJsonFile(g.cwdIndexFile(resolved), {
              cwd: resolved,
              sessionIds,
            });
          })
        ),

      readEvents: <T>(sessionId: string) =>
        io.readJsonLines<T>(g.sessionEventsFile(sessionId)),

      readEventsAfterCursor: <T extends { readonly timestamp: number }>(
        sessionId: string,
        cursor: StoredEventCursor
      ) =>
        Effect.gen(function* () {
          const events = yield* io.readJsonLines<T>(
            g.sessionEventsFile(sessionId)
          );
          const cursorEvent = events[cursor.index];
          if (!cursorEvent || cursorEvent.timestamp !== cursor.timestamp) {
            return null;
          }
          return events.slice(cursor.index + 1);
        }),

      appendEvents: <T>(sessionId: string, events: readonly T[]) =>
        io.withPathLock(
          g.sessionEventsFile(sessionId),
          Effect.gen(function* () {
            yield* io.ensureDir(g.sessionDir(sessionId));
            yield* io.appendJsonLines(g.sessionEventsFile(sessionId), events);
          })
        ),

      appendEventsWithCursor: <T extends { readonly timestamp: number }>(
        sessionId: string,
        events: readonly T[]
      ) =>
        io.withPathLock(
          g.sessionEventsFile(sessionId),
          Effect.gen(function* () {
            if (events.length === 0) return null;
            yield* io.ensureDir(g.sessionDir(sessionId));
            const existing = yield* io.readJsonLines<T>(
              g.sessionEventsFile(sessionId)
            );
            yield* io.appendJsonLines(g.sessionEventsFile(sessionId), events);
            const lastEvent = events[events.length - 1];
            return {
              index: existing.length + events.length - 1,
              timestamp: lastEvent.timestamp,
            };
          })
        ),

      readProjectionSnapshot: (sessionId: string) =>
        Effect.gen(function* () {
          const filePath = g.sessionProjectionSnapshotFile(sessionId);
          const result = yield* readStructuredFile(filePath, Schema.Unknown).pipe(
            Effect.provideService(FileSystem.FileSystem, fs)
          );
          if (result._tag === "Missing") return null;
          if (result._tag === "Invalid") {
            // Disposable cache — log and rebuild
            yield* Effect.logWarning(
              `[storage] Corrupt projection snapshot for session ${sessionId}, ignoring: ${result.error.reason}`
            );
            return null;
          }
          return result.value;
        }),

      writeProjectionSnapshot: <A>(sessionId: string, envelope: A) =>
        Effect.gen(function* () {
          yield* io.ensureDir(g.sessionDir(sessionId));
          yield* io.writeJsonFileAtomic(
            g.sessionProjectionSnapshotFile(sessionId),
            envelope
          );
        }),

      readAddressedEntry: (sessionId, namespace, address) =>
        Effect.gen(function* () {
          const filePath = g.sessionAddressedEntryFile(
            sessionId,
            namespace,
            address
          );
          const result = yield* readStructuredFile(filePath, Schema.Struct({
            value: Schema.Unknown,
          })).pipe(
            Effect.provideService(FileSystem.FileSystem, fs)
          );
          if (result._tag === "Missing") return null;
          if (result._tag === "Invalid") {
            return yield* Effect.fail(
              new SchemaDecodeError({ path: filePath, message: result.error.reason })
            );
          }
          return { value: result.value.value } satisfies StoredAddressedEntry;
        }),

      statAddressedEntry: (sessionId, namespace, address) =>
        Effect.gen(function* () {
          const filePath = g.sessionAddressedEntryFile(
            sessionId,
            namespace,
            address
          );
          const storedBytes = yield* io.fileSize(filePath);
          return storedBytes === null
            ? null
            : ({ storedBytes } satisfies StoredAddressedEntryStats);
        }),

      writeAddressedEntry: (sessionId, namespace, address, value) =>
        io.writeJsonFileAtomic(
          g.sessionAddressedEntryFile(sessionId, namespace, address),
          { value } satisfies StoredAddressedEntry
        ),

      readEventsFromPath: <T>(eventsPath: string) =>
        io.readJsonLines<T>(eventsPath),

      appendLogs: <T>(sessionId: string, entries: readonly T[]) =>
        Effect.gen(function* () {
          yield* io.ensureDir(g.sessionDir(sessionId));
          yield* io.appendJsonLines(g.sessionLogFile(sessionId), entries);
        }),

      clearLog: (sessionId) =>
        io.removeFileIfExists(g.sessionLogFile(sessionId)),

      createSessionScratchpad: (sessionId) =>
        Effect.gen(function* () {
          const scratchpadPath = g.sessionScratchpad(sessionId);
          yield* io.ensureDir(scratchpadPath);
          for (const subdir of SCRATCHPAD_SUBDIRS) {
            yield* io.ensureDir(path.join(scratchpadPath, subdir));
          }
          return scratchpadPath;
        }),

      createMemoryExtractionJobRecord: (params) => {
        const now = params.now ?? new Date();
        const uniqueId =
          params.createId?.() ?? `${Math.random().toString(36).slice(2)}`;
        return {
          jobId: `${params.sessionId}-${now.getTime()}-${uniqueId}`,
          sessionId: params.sessionId,
          cwd: params.cwd,
          eventsPath: params.eventsPath,
          memoryPath: params.memoryPath,
          createdAt: now.toISOString(),
          attempts: 0,
          status: "pending" as const,
        };
      },

      writePendingMemoryJob: (job) =>
        io.withPathLock(
          g.pendingMemoryJobFile(job.jobId),
          Effect.gen(function* () {
            const filePath = g.pendingMemoryJobFile(job.jobId);
            yield* io.ensureDir(g.pendingMemoryExtractionRoot);
            yield* io.writeJsonFile(filePath, job);
            return filePath;
          })
        ),

      listPendingMemoryJobFiles: () =>
        Effect.gen(function* () {
          const entries = yield* io.listDirectory(
            g.pendingMemoryExtractionRoot
          );
          return entries
            .filter(
              (entry) => entry.isFile && path.extname(entry.name) === ".json"
            )
            .map((entry) => entry.path)
            .sort();
        }),

      listPendingMemoryJobIds: () =>
        Effect.gen(function* () {
          const entries = yield* io.listDirectory(
            g.pendingMemoryExtractionRoot
          );
          return entries
            .filter(
              (entry) => entry.isFile && path.extname(entry.name) === ".json"
            )
            .map((entry) => path.basename(entry.name, ".json"))
            .sort();
        }),

      readPendingMemoryJob: (input) =>
        Effect.gen(function* () {
          const filePath = getPendingMemoryJobPath(input);
          const result = yield* readStructuredFile(
            filePath,
            MemoryExtractionJobRecordSchema
          ).pipe(Effect.provideService(FileSystem.FileSystem, fs));
          if (result._tag === "Missing") {
            return yield* Effect.fail(
              new SchemaDecodeError({ path: filePath, message: "Job file not found" })
            );
          }
          if (result._tag === "Invalid") {
            return yield* Effect.fail(
              new SchemaDecodeError({ path: filePath, message: result.error.reason })
            );
          }
          return result.value;
        }),

      markPendingMemoryJobRunning: (input, job?) =>
        io.withPathLock(
          getPendingMemoryJobPath(input),
          Effect.gen(function* () {
            const filePath = getPendingMemoryJobPath(input);
            const current =
              job ??
              (yield* Effect.gen(function* () {
                const result = yield* readStructuredFile(
                  filePath,
                  MemoryExtractionJobRecordSchema
                ).pipe(Effect.provideService(FileSystem.FileSystem, fs));
                if (result._tag === "Present") return result.value;
                return yield* Effect.fail(
                  new SchemaDecodeError({
                    path: filePath,
                    message:
                      result._tag === "Missing"
                        ? "Job file not found"
                        : result.error.reason,
                  })
                );
              }));
            const next: MemoryExtractionJobRecord = {
              ...current,
              status: "running",
              attempts: (current.attempts ?? 0) + 1,
            };
            yield* io.writeJsonFile(getPendingMemoryJobPath(input), next);
            return next;
          })
        ),

      markPendingMemoryJobPending: (input, job?) =>
        io.withPathLock(
          getPendingMemoryJobPath(input),
          Effect.gen(function* () {
            const filePath = getPendingMemoryJobPath(input);
            const current =
              job ??
              (yield* Effect.gen(function* () {
                const result = yield* readStructuredFile(
                  filePath,
                  MemoryExtractionJobRecordSchema
                ).pipe(Effect.provideService(FileSystem.FileSystem, fs));
                if (result._tag === "Present") return result.value;
                return yield* Effect.fail(
                  new SchemaDecodeError({
                    path: filePath,
                    message:
                      result._tag === "Missing"
                        ? "Job file not found"
                        : result.error.reason,
                  })
                );
              }));
            const next: MemoryExtractionJobRecord = {
              ...current,
              status: "pending",
            };
            yield* io.writeJsonFile(getPendingMemoryJobPath(input), next);
            return next;
          })
        ),

      removePendingMemoryJob: (input) =>
        io.removeFileIfExists(getPendingMemoryJobPath(input)),

      resolvePendingMemoryJobPath: (jobId) => g.pendingMemoryJobFile(jobId),
    };
  });
}
