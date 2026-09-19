import { FileSystem } from "@effect/platform"
import { Effect, Exit, Queue, Schema, Stream } from "effect"
import { InfrastructureFailure } from "./domain"

export const JsonProcessConfig = Schema.Struct({ executable: Schema.String, args: Schema.Array(Schema.String), cwd: Schema.String,
  environment: Schema.Record({ key: Schema.String, value: Schema.String }), stdoutLog: Schema.String, stderrLog: Schema.String })
export interface JsonProcess {
  readonly send: (value: unknown) => Effect.Effect<void, InfrastructureFailure>
  readonly receive: Effect.Effect<unknown, InfrastructureFailure>
}
const failed = (message: string) => new InfrastructureFailure({ operation: "harness-process", message })
/** LF-only JSONL framing and scoped ownership for long-lived CLI RPC sessions. */
export const jsonProcess = (config: typeof JsonProcessConfig.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  let stdout = "", stderr = "", stdoutBytes = 0
  yield* Effect.addFinalizer(() => Effect.all([
    fs.writeFileString(config.stdoutLog, stdout), fs.writeFileString(config.stderrLog, stderr),
  ]).pipe(Effect.orDie))
  const queue = yield* Effect.acquireRelease(Queue.bounded<Exit.Exit<unknown, InfrastructureFailure>>(64), Queue.shutdown)
  const child = yield* Effect.acquireRelease(Effect.try({ try: () => Bun.spawn([config.executable, ...config.args], {
    cwd: config.cwd, env: config.environment, stdin: "pipe", stdout: "pipe", stderr: "pipe", detached: process.platform !== "win32",
  }), catch: () => failed("Cannot start the pinned harness executable") }), child => Effect.gen(function* () {
    const stop = (signal: "SIGTERM" | "SIGKILL") => Effect.sync(() => {
      try { if (process.platform !== "win32") process.kill(-child.pid, signal); else child.kill(signal) } catch { /* Already exited. */ }
    })
    if (child.exitCode === null) {
      yield* stop("SIGTERM")
      const exited = yield* Effect.promise(() => child.exited).pipe(Effect.timeoutOption("2 seconds"))
      if (exited._tag === "None") yield* stop("SIGKILL")
    }
    yield* Effect.promise(() => child.exited).pipe(Effect.timeoutOption("3 seconds"))
  }))
  const read = (stream: ReadableStream<Uint8Array>) => Stream.fromReadableStream(() => stream, () => failed("Harness output stream failed")).pipe(Stream.decodeText())
  let pending = ""
  yield* read(child.stdout).pipe(Stream.runForEach(part => Effect.gen(function* () {
    stdoutBytes += Buffer.byteLength(part, "utf8")
    if (stdoutBytes > 16 * 1024 * 1024) return yield* failed("Harness JSON output exceeded 16 MiB")
    stdout += part; pending += part
    for (;;) {
      const newline = pending.indexOf("\n")
      if (newline === -1) break
      const line = pending.slice(0, newline).replace(/\r$/, "")
      if (Buffer.byteLength(line, "utf8") > 2 * 1024 * 1024) return yield* failed("Harness JSON line exceeded 2 MiB")
      pending = pending.slice(newline + 1)
      if (!line) continue
      const value = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Unknown))(line).pipe(Effect.mapError(() => failed("Harness emitted invalid JSONL")))
      yield* Queue.offer(queue, Exit.succeed(value))
    }
    if (Buffer.byteLength(pending, "utf8") > 2 * 1024 * 1024) return yield* failed("Harness JSON line exceeded 2 MiB")
  })), Effect.zipRight(Effect.suspend(() => Effect.fail(failed(pending ? "Harness ended with an incomplete JSONL frame" : "Harness process closed its output")))),
    Effect.catchAll(error => Queue.offer(queue, Exit.fail(error))), Effect.forkScoped)
  yield* read(child.stderr).pipe(Stream.runForEach(part => Effect.sync(() => { stderr = (stderr + part).slice(-2 * 1024 * 1024) })),
    Effect.catchAll(error => Queue.offer(queue, Exit.fail(error))), Effect.forkScoped)
  return {
    send: value => Effect.gen(function* () {
      const line = yield* Schema.encode(Schema.parseJson(Schema.Unknown))(value).pipe(Effect.mapError(() => failed("Invalid harness RPC command")))
      yield* Effect.tryPromise({ try: async () => { child.stdin.write(`${line}\n`); await child.stdin.flush() }, catch: () => failed("Harness input pipe closed") })
    }),
    receive: Queue.take(queue).pipe(Effect.flatMap(exit => Exit.match(exit, { onSuccess: Effect.succeed, onFailure: Effect.failCause }))),
  } satisfies JsonProcess
})
