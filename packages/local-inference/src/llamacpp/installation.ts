import { createHash } from "node:crypto"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Data, Effect, Option, pipe, Schema } from "effect"
import { sha256File } from "../model-files/platform"
import { runParsedCommand, type CommandOutputParser } from "../command-output"
import { LlamaCppExecutableFingerprint, LlamaCppInstallationId, LlamaBuildCommitId } from "./identity"

export const LlamaBuildNumber = Schema.Number.pipe(Schema.int(), Schema.positive(), Schema.brand("LlamaBuildNumber"))
export type LlamaBuildNumber = Schema.Schema.Type<typeof LlamaBuildNumber>
export const LlamaBuildIdentity = Schema.Struct({
  versionOutput: Schema.String,
  buildNumber: LlamaBuildNumber,
  commit: Schema.OptionFromSelf(LlamaBuildCommitId),
})
export type LlamaBuildIdentity = Schema.Schema.Type<typeof LlamaBuildIdentity>
export const LlamaCppInstallationDiscovery = Schema.Union(
  Schema.TaggedStruct("Configured", { requestedPath: Schema.String }),
  Schema.TaggedStruct("Managed", { markerPath: Schema.String, release: Schema.String }),
  Schema.TaggedStruct("Path", { requestedPath: Schema.String, priority: Schema.NonNegativeInt }),
)
export type LlamaCppInstallationDiscovery = Schema.Schema.Type<typeof LlamaCppInstallationDiscovery>
export const LlamaCppExecutableOperation = Schema.Literal("resolve", "access", "stat", "version", "parse-version", "fingerprint")
export type LlamaCppExecutableOperation = Schema.Schema.Type<typeof LlamaCppExecutableOperation>
export const LlamaCppExecutableFailureReason = Schema.Literal("not-found", "not-executable", "not-a-file", "command-failed", "empty-output", "unrecognized-version", "unreadable", "build-mismatch")
export type LlamaCppExecutableFailureReason = Schema.Schema.Type<typeof LlamaCppExecutableFailureReason>
export const LlamaCppExecutableSchema = Schema.Struct({
  path: Schema.String,
  fingerprint: LlamaCppExecutableFingerprint,
})
export type LlamaCppExecutable = typeof LlamaCppExecutableSchema.Type
export const LlamaCppInstallationSchema = Schema.Struct({
  id: LlamaCppInstallationId,
  build: LlamaBuildNumber,
  commit: Schema.OptionFromSelf(LlamaBuildCommitId),
  executables: Schema.Struct({
    server: LlamaCppExecutableSchema,
    fitParams: LlamaCppExecutableSchema,
  }),
  ownership: Schema.Literal("user", "magnitude"),
  discoveries: Schema.Array(LlamaCppInstallationDiscovery),
})
export type LlamaCppInstallation = typeof LlamaCppInstallationSchema.Type

export class LlamaCppExecutableError extends Data.TaggedError("LlamaCppExecutableError")<{
  readonly executable: string
  readonly operation: LlamaCppExecutableOperation
  readonly reason: LlamaCppExecutableFailureReason
}> {}

export const parseLlamaBuildIdentity = (executable: string, output: string): Effect.Effect<LlamaBuildIdentity, LlamaCppExecutableError> => {
  const text = output.trim()
  if (text.length === 0) return Effect.fail(new LlamaCppExecutableError({ executable, operation: "parse-version", reason: "empty-output" }))
  const build = pipe(
    Option.fromNullable(text.match(/\b(?:build|version)\s*[:#]?\s*(\d+)\b/i)),
    Option.flatMap((match) => Option.fromNullable(match[1])),
  )
  if (Option.isNone(build) || !Number.isSafeInteger(Number(build.value)) || Number(build.value) <= 0) {
    return Effect.fail(new LlamaCppExecutableError({ executable, operation: "parse-version", reason: "unrecognized-version" }))
  }
  const commit = pipe(Option.fromNullable(text.match(/\b(?:commit|revision)\s*[:=]?\s*([a-f0-9]{7,40})\b/i)), Option.flatMap((match) => Option.fromNullable(match[1])))
  return Effect.succeed({
    versionOutput: text,
    buildNumber: LlamaBuildNumber.make(Number(build.value)),
    commit: Option.map(commit, LlamaBuildCommitId.make),
  })
}

export const llamaBuildIdentityParser = (executable: string): CommandOutputParser<LlamaBuildIdentity, LlamaCppExecutableError> => ({
  name: "llama-build-identity",
  parse: (output) => output.exitCode === 0
    ? parseLlamaBuildIdentity(executable, `${output.stdout}\n${output.stderr}`)
    : Effect.fail(new LlamaCppExecutableError({ executable, operation: "version", reason: "command-failed" })),
})

export const validateLlamaCppExecutable = (
  requestedPath: string,
): Effect.Effect<{ readonly executable: LlamaCppExecutable; readonly identity: LlamaBuildIdentity }, LlamaCppExecutableError, FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const requested = path.resolve(requestedPath)
  const executablePath = yield* fs.realPath(requested).pipe(Effect.mapError(() => new LlamaCppExecutableError({ executable: requested, operation: "resolve", reason: "not-found" })))
  yield* fs.access(executablePath, { readable: true, ok: true }).pipe(Effect.mapError(() => new LlamaCppExecutableError({ executable: executablePath, operation: "access", reason: "not-executable" })))
  const info = yield* fs.stat(executablePath).pipe(Effect.mapError(() => new LlamaCppExecutableError({ executable: executablePath, operation: "stat", reason: "unreadable" })))
  if (info.type !== "File") return yield* new LlamaCppExecutableError({ executable: executablePath, operation: "stat", reason: "not-a-file" })
  const identity = yield* runParsedCommand(executablePath, ["--version"], llamaBuildIdentityParser(executablePath), { maxOutputBytes: 1024 * 1024, redactions: [] }).pipe(Effect.timeout("10 seconds"), Effect.mapError((error) => error._tag === "LlamaCppExecutableError" ? error : new LlamaCppExecutableError({ executable: executablePath, operation: "version", reason: "command-failed" })))
  const digest = yield* sha256File(executablePath).pipe(Effect.mapError(() => new LlamaCppExecutableError({ executable: executablePath, operation: "fingerprint", reason: "unreadable" })))
  const fingerprint = LlamaCppExecutableFingerprint.make(createHash("sha256").update(`${digest}\0${info.size}\0${identity.versionOutput}`).digest("hex"))
  return {
    executable: { path: executablePath, fingerprint },
    identity,
  }
})

export const makeLlamaCppInstallation = (input: {
  readonly server: { readonly executable: LlamaCppExecutable; readonly identity: LlamaBuildIdentity }
  readonly fitParams: { readonly executable: LlamaCppExecutable; readonly identity: LlamaBuildIdentity }
  readonly ownership: LlamaCppInstallation["ownership"]
  readonly discoveries: readonly LlamaCppInstallationDiscovery[]
}): Effect.Effect<LlamaCppInstallation, LlamaCppExecutableError> => {
  if (input.server.identity.buildNumber !== input.fitParams.identity.buildNumber) {
    return Effect.fail(new LlamaCppExecutableError({ executable: input.fitParams.executable.path, operation: "version", reason: "build-mismatch" }))
  }
  const id = LlamaCppInstallationId.make(createHash("sha256")
    .update(input.server.executable.path)
    .update("\0")
    .update(input.fitParams.executable.path)
    .digest("hex"))
  return Effect.succeed({
    id,
    build: input.server.identity.buildNumber,
    commit: input.server.identity.commit,
    executables: { server: input.server.executable, fitParams: input.fitParams.executable },
    ownership: input.ownership,
    discoveries: input.discoveries,
  })
}
