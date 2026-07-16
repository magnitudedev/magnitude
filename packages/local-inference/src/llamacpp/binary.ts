import { createHash } from "node:crypto"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Context, Data, Effect, Option, pipe, Schema } from "effect"
import { sha256File } from "../model-files/platform"
import { runParsedCommand, type CommandOutputParser } from "../command-output"
import { LlamaBinaryFingerprint, LlamaBuildCommitId } from "./identity"

export interface LlamaBuildIdentity { readonly versionOutput: string; readonly buildNumber: Option.Option<number>; readonly commit: Option.Option<LlamaBuildCommitId> }
export const LlamaBinarySource = Schema.Literal("managed", "configured", "path")
export type LlamaBinarySource = Schema.Schema.Type<typeof LlamaBinarySource>
export const LlamaBinaryOperation = Schema.Literal("resolve", "access", "stat", "version", "parse-version", "fingerprint")
export type LlamaBinaryOperation = Schema.Schema.Type<typeof LlamaBinaryOperation>
export const LlamaBinaryFailureReason = Schema.Literal("not-found", "not-executable", "not-a-file", "command-failed", "empty-output", "unreadable")
export type LlamaBinaryFailureReason = Schema.Schema.Type<typeof LlamaBinaryFailureReason>
export interface LlamaBinary {
  readonly executable: string
  readonly distributionDirectory: string
  readonly build: LlamaBuildIdentity
  readonly source: LlamaBinarySource
  readonly fingerprint: LlamaBinaryFingerprint
}

export const LlamaBinary = Context.GenericTag<LlamaBinary>(
  "@magnitudedev/local-inference/LlamaBinary",
)

export class LlamaBinaryError extends Data.TaggedError("LlamaBinaryError")<{
  readonly executable: string
  readonly operation: LlamaBinaryOperation
  readonly reason: LlamaBinaryFailureReason
}> {}

export const parseLlamaBuildIdentity = (executable: string, output: string): Effect.Effect<LlamaBuildIdentity, LlamaBinaryError> => {
  const text = output.trim()
  if (text.length === 0) return Effect.fail(new LlamaBinaryError({ executable, operation: "parse-version", reason: "empty-output" }))
  const build = pipe(Option.fromNullable(text.match(/\bbuild\s*[:#]?\s*(\d+)\b/i)), Option.flatMap((match) => Option.fromNullable(match[1])))
  const commit = pipe(Option.fromNullable(text.match(/\b(?:commit|revision)\s*[:=]?\s*([a-f0-9]{7,40})\b/i)), Option.flatMap((match) => Option.fromNullable(match[1])))
  return Effect.succeed({
    versionOutput: text,
    buildNumber: Option.map(build, Number),
    commit: Option.map(commit, LlamaBuildCommitId.make),
  })
}

export const llamaBuildIdentityParser = (executable: string): CommandOutputParser<LlamaBuildIdentity, LlamaBinaryError> => ({
  name: "llama-build-identity",
  parse: (output) => output.exitCode === 0
    ? parseLlamaBuildIdentity(executable, `${output.stdout}\n${output.stderr}`)
    : Effect.fail(new LlamaBinaryError({ executable, operation: "version", reason: "command-failed" })),
})

export const validateLlamaBinary = (input: { readonly executable: string; readonly source: LlamaBinary["source"] }): Effect.Effect<LlamaBinary, LlamaBinaryError, FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const requested = path.resolve(input.executable)
  const executable = yield* fs.realPath(requested).pipe(Effect.mapError(() => new LlamaBinaryError({ executable: requested, operation: "resolve", reason: "not-found" })))
  yield* fs.access(executable, { readable: true, ok: true }).pipe(Effect.mapError(() => new LlamaBinaryError({ executable, operation: "access", reason: "not-executable" })))
  const info = yield* fs.stat(executable).pipe(Effect.mapError(() => new LlamaBinaryError({ executable, operation: "stat", reason: "unreadable" })))
  if (info.type !== "File") return yield* new LlamaBinaryError({ executable, operation: "stat", reason: "not-a-file" })
  const build = yield* runParsedCommand(executable, ["--version"], llamaBuildIdentityParser(executable), { maxOutputBytes: 1024 * 1024, redactions: [] }).pipe(Effect.timeout("10 seconds"), Effect.mapError((error) => error._tag === "LlamaBinaryError" ? error : new LlamaBinaryError({ executable, operation: "version", reason: "command-failed" })))
  const digest = yield* sha256File(executable).pipe(Effect.mapError(() => new LlamaBinaryError({ executable, operation: "fingerprint", reason: "unreadable" })))
  const fingerprint = LlamaBinaryFingerprint.make(createHash("sha256").update(`${digest}\0${info.size}\0${build.versionOutput}`).digest("hex"))
  return { executable, distributionDirectory: path.dirname(executable), build, source: input.source, fingerprint }
})
