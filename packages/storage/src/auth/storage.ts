import * as FileSystem from "@effect/platform/FileSystem";
import * as Path from "@effect/platform/Path";
import { type PlatformError } from "@effect/platform/Error";
import { Effect, Option, Schema } from "effect";

import { makeStorageIo, type JsonError, SchemaDecodeError } from "../io/storage";
import { readStructuredFile } from "../io/structured-file";
import { GlobalStorage } from "../services";
import {
  HuggingFaceCredentialSchema,
  isValidAuthInfo,
  type AuthInfo,
} from "../types/auth";
import type { AuthStorageShape } from "./contracts";

const encodeHuggingFace = Schema.encodeSync(HuggingFaceCredentialSchema);

function normalizeAuthData(data: unknown): Record<string, AuthInfo> {
  if (typeof data !== "object" || data === null) return {};
  const providers: Record<string, AuthInfo> = {};
  for (const [key, value] of Object.entries(data)) {
    if (isValidAuthInfo(value)) {
      providers[key] = value;
    }
  }
  return providers;
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
    const readAuthFile = (): Effect.Effect<Record<string, AuthInfo>, PlatformError | JsonError> =>
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

    // The file is written owner-only (0600); an empty file is removed.
    const modify = (update: (providers: Record<string, AuthInfo>) => Record<string, AuthInfo>) =>
      io.withPathLock(
        g.authFile,
        Effect.gen(function* () {
          const next = update(yield* readAuthFile());
          if (Object.keys(next).length === 0) {
            yield* io.removeFileIfExists(g.authFile);
          } else {
            yield* io.writeSecureJsonFile(g.authFile, next);
          }
        })
      );

    return {
      loadAll: readAuthFile,

      get: (providerId) =>
        readAuthFile().pipe(Effect.map((providers) => providers[providerId])),

      set: (providerId, info) =>
        modify((providers) => ({ ...providers, [providerId]: info })),

      remove: (providerId) =>
        modify((providers) => {
          const next = { ...providers };
          delete next[providerId];
          return next;
        }),

      // The Hugging Face sign-in is its own owner-only file; a missing file is
      // signed out, and an invalid one is corruption, reported like auth.json's.
      getHuggingFace: readStructuredFile(g.huggingFaceFile, HuggingFaceCredentialSchema).pipe(
        Effect.provideService(FileSystem.FileSystem, fs),
        Effect.flatMap((result) => {
          switch (result._tag) {
            case "Missing": return Effect.succeed(Option.none());
            case "Invalid": return Effect.fail(
              new SchemaDecodeError({ path: g.huggingFaceFile, message: result.error.reason })
            );
            case "Present": return Effect.succeed(Option.some(result.value));
          }
        })
      ),

      setHuggingFace: (credential) =>
        io.writeSecureJsonFile(g.huggingFaceFile, encodeHuggingFace(credential)),

      removeHuggingFace: io.removeFileIfExists(g.huggingFaceFile),
    };
  });
}
