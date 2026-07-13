import { Context, Effect, Layer, Schema } from "effect";
import * as FileSystem from "@effect/platform/FileSystem";
import * as NodePath from "node:path";
import {
  MagnitudeStorage,
  readStructuredFile,
  writeStructuredFileAtomic,
} from "@magnitudedev/storage";
import {
  SessionOperationFailed,
  type SessionError,
  SessionOptions as SessionOptionsSchema,
} from "@magnitudedev/protocol";

export interface SessionRuntimeOptions {
  readonly disableShellSafeguards: boolean;
  readonly disableCwdSafeguards: boolean;
  readonly atifPath: string | null;
  readonly solo: boolean;
  readonly systemPromptOverride: string | null;
  readonly headless: boolean;
}

const SessionRuntimeOptionsSchema: Schema.Schema<SessionRuntimeOptions> =
  Schema.Struct({
    disableShellSafeguards: Schema.Boolean,
    disableCwdSafeguards: Schema.Boolean,
    atifPath: Schema.NullOr(Schema.String),
    solo: Schema.Boolean,
    systemPromptOverride: Schema.NullOr(Schema.String),
    headless: Schema.Boolean,
  });

type SessionOptionsInput = Schema.Schema.Encoded<typeof SessionOptionsSchema>;

export const normalizeSessionRuntimeOptions = (
  options?: SessionOptionsInput
): SessionRuntimeOptions => ({
  disableShellSafeguards: options?.disableShellSafeguards ?? false,
  disableCwdSafeguards: options?.disableCwdSafeguards ?? false,
  atifPath: options?.atifPath ?? null,
  solo: options?.solo ?? false,
  systemPromptOverride: options?.systemPromptOverride ?? null,
  headless: options?.headless ?? false,
});

export interface SessionRuntimeOptionsStoreApi {
  readonly normalize: (options?: SessionOptionsInput) => SessionRuntimeOptions;
  readonly read: (
    sessionId: string
  ) => Effect.Effect<SessionRuntimeOptions | null, SessionError>;
  readonly write: (
    sessionId: string,
    options: SessionRuntimeOptions
  ) => Effect.Effect<void, SessionError>;
}

export class SessionRuntimeOptionsStore extends Context.Tag(
  "SessionRuntimeOptionsStore"
)<SessionRuntimeOptionsStore, SessionRuntimeOptionsStoreApi>() {}

const toRuntimeOptionsError =
  (sessionId: string, operation: string) =>
  (cause: unknown): SessionError =>
    new SessionOperationFailed({
      operation: `session runtime options ${operation}`,
      reason: `${sessionId}: ${String(cause)}`,
    });

export const SessionRuntimeOptionsStoreLive: Layer.Layer<
  SessionRuntimeOptionsStore,
  never,
  MagnitudeStorage | FileSystem.FileSystem
> = Layer.effect(
  SessionRuntimeOptionsStore,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage;
    const fs = yield* FileSystem.FileSystem;

    const pathFor = (sessionId: string) =>
      NodePath.join(
        storage.sessions.paths.sessionDir(sessionId),
        "runtime-options.json"
      );

    // Optimistic read: missing file → null, corrupt JSON → null. Never writes.
    const read = Effect.fn("acn.session-runtime-options.read")(function* (
      sessionId: string
    ) {
      const result = yield* readStructuredFile(
        pathFor(sessionId),
        SessionRuntimeOptionsSchema
      ).pipe(
        Effect.provideService(FileSystem.FileSystem, fs),
        Effect.mapError(toRuntimeOptionsError(sessionId, "read"))
      );
      if (result._tag === "Missing") return null;
      if (result._tag === "Invalid") {
        return yield* toRuntimeOptionsError(sessionId, "decode")(result.error);
      }
      return result.value;
    });

    const write = Effect.fn("acn.session-runtime-options.write")(function* (
      sessionId: string,
      options: SessionRuntimeOptions
    ) {
      const path = pathFor(sessionId);
      yield* writeStructuredFileAtomic(
        path,
        SessionRuntimeOptionsSchema,
        options
      ).pipe(
        Effect.provideService(FileSystem.FileSystem, fs),
        Effect.mapError(toRuntimeOptionsError(sessionId, "write"))
      );
    });

    return {
      normalize: normalizeSessionRuntimeOptions,
      read,
      write,
    };
  })
);
