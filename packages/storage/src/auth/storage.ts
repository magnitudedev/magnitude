import * as FileSystem from "@effect/platform/FileSystem";
import * as Path from "@effect/platform/Path";
import { type PlatformError } from "@effect/platform/Error";
import { Effect, Schema } from "effect";

import { makeStorageIo, type JsonError, SchemaDecodeError } from "../io/storage";
import { readStructuredFile } from "../io/structured-file";
import { GlobalStorage } from "../services";
import { isValidAuthInfo, type AuthInfo } from "../types/auth";
import type { AuthStorageShape } from "./contracts";

function normalizeAuthData(data: unknown): Record<string, AuthInfo> {
  if (typeof data !== "object" || data === null) return {};
  const result: Record<string, AuthInfo> = {};
  for (const [key, value] of Object.entries(data)) {
    if (isValidAuthInfo(value)) {
      result[key] = value;
    }
  }
  return result;
}

export function makeAuthStorage(): Effect.Effect<
  AuthStorageShape,
  never,
  FileSystem.FileSystem | Path.Path | GlobalStorage
> {
  return Effect.gen(function* () {
    const io = yield* makeStorageIo();
    const fs = yield* FileSystem.FileSystem;
    const globalStorage = yield* GlobalStorage;
    const g = globalStorage.paths;

    // Durable read: Missing is a legitimate first-run state (return empty),
    // Invalid is corruption (return typed error — never silently overwrite).
    const readAuthRaw = (): Effect.Effect<Record<string, AuthInfo>, PlatformError | JsonError> =>
      Effect.gen(function* () {
        const result = yield* readStructuredFile(g.authFile, Schema.Unknown).pipe(
          Effect.provideService(FileSystem.FileSystem, fs)
        );
        if (result._tag === "Missing") return {};
        if (result._tag === "Invalid") {
          return yield* Effect.fail(
            new SchemaDecodeError({ path: g.authFile, message: result.error.reason })
          );
        }
        return normalizeAuthData(result.value);
      });

    return {
      loadAll: () => readAuthRaw(),

      get: (providerId) =>
        readAuthRaw().pipe(Effect.map((data) => data[providerId])),

      set: (providerId, info) =>
        io.withPathLock(
          g.authFile,
          Effect.gen(function* () {
            const normalized = yield* readAuthRaw();
            normalized[providerId] = info;
            yield* io.writeSecureJsonFile(g.authFile, normalized);
          })
        ),

      remove: (providerId) =>
        io.withPathLock(
          g.authFile,
          Effect.gen(function* () {
            const normalized = yield* readAuthRaw();
            delete normalized[providerId];
            if (Object.keys(normalized).length === 0) {
              yield* io.removeFileIfExists(g.authFile);
            } else {
              yield* io.writeSecureJsonFile(g.authFile, normalized);
            }
          })
        ),
    };
  });
}
