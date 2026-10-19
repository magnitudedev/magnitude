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
  type HuggingFaceCredential,
} from "../types/auth";
import type { AuthStorageShape } from "./contracts";

/** The `auth.json` key of the Hugging Face account; every other key is a provider. */
export const HUGGING_FACE_AUTH_KEY = "huggingface";

interface AuthFile {
  readonly providers: Record<string, AuthInfo>;
  readonly huggingFace: Option.Option<HuggingFaceCredential>;
}

const decodeHuggingFace = Schema.decodeUnknownOption(HuggingFaceCredentialSchema);
const encodeHuggingFace = Schema.encodeSync(HuggingFaceCredentialSchema);

function normalizeAuthData(data: unknown): AuthFile {
  if (typeof data !== "object" || data === null) return { providers: {}, huggingFace: Option.none() };
  const providers: Record<string, AuthInfo> = {};
  for (const [key, value] of Object.entries(data)) {
    if (key !== HUGGING_FACE_AUTH_KEY && isValidAuthInfo(value)) {
      providers[key] = value;
    }
  }
  return {
    providers,
    huggingFace: decodeHuggingFace((data as Record<string, unknown>)[HUGGING_FACE_AUTH_KEY]),
  };
}

const encodeAuthFile = (file: AuthFile): Record<string, unknown> => Option.match(file.huggingFace, {
  onNone: () => ({ ...file.providers }),
  onSome: (credential) => ({ ...file.providers, [HUGGING_FACE_AUTH_KEY]: encodeHuggingFace(credential) }),
});

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
    const readAuthFile = (): Effect.Effect<AuthFile, PlatformError | JsonError> =>
      Effect.gen(function* () {
        const result = yield* readStructuredFile(g.authFile, Schema.Unknown).pipe(
          Effect.provideService(FileSystem.FileSystem, fs)
        );
        if (result._tag === "Missing") return { providers: {}, huggingFace: Option.none() };
        if (result._tag === "Invalid") {
          return yield* Effect.fail(
            new SchemaDecodeError({ path: g.authFile, message: result.error.reason })
          );
        }
        return normalizeAuthData(result.value);
      });

    // The file is written owner-only (0600); an empty file is removed.
    const modify = (update: (file: AuthFile) => AuthFile) =>
      io.withPathLock(
        g.authFile,
        Effect.gen(function* () {
          const next = encodeAuthFile(update(yield* readAuthFile()));
          if (Object.keys(next).length === 0) {
            yield* io.removeFileIfExists(g.authFile);
          } else {
            yield* io.writeSecureJsonFile(g.authFile, next);
          }
        })
      );

    return {
      loadAll: () => readAuthFile().pipe(Effect.map((file) => file.providers)),

      get: (providerId) =>
        readAuthFile().pipe(Effect.map((file) => file.providers[providerId])),

      set: (providerId, info) =>
        modify((file) => ({ ...file, providers: { ...file.providers, [providerId]: info } })),

      remove: (providerId) =>
        modify((file) => {
          const providers = { ...file.providers };
          delete providers[providerId];
          return { ...file, providers };
        }),

      getHuggingFace: Effect.suspend(readAuthFile).pipe(Effect.map((file) => file.huggingFace)),

      setHuggingFace: (credential) =>
        modify((file) => ({ ...file, huggingFace: Option.some(credential) })),

      removeHuggingFace: Effect.suspend(() => modify((file) => ({ ...file, huggingFace: Option.none() }))),
    };
  });
}
