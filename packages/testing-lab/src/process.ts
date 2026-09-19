import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { InfrastructureFailure } from "./domain"

export const CommandSpec = Schema.Struct({
  executable: Schema.NonEmptyString, args: Schema.Array(Schema.String),
  cwd: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  env: Schema.Record({ key: Schema.String, value: Schema.String }),
  inheritEnv: Schema.Boolean, timeoutMs: Schema.Int.pipe(Schema.positive()),
  maxOutputBytes: Schema.Int.pipe(Schema.positive()),
  stdin: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})
export type CommandSpec = typeof CommandSpec.Type
export const CommandOutput = Schema.Struct({ exitCode: Schema.Int, stdout: Schema.String, stderr: Schema.String })
export type CommandOutput = typeof CommandOutput.Type
export interface ProcessExecutor {
  readonly run: (spec: CommandSpec) => Effect.Effect<CommandOutput, InfrastructureFailure>
}
export const ProcessExecutor = Context.GenericTag<ProcessExecutor>("@magnitudedev/testing-lab/ProcessExecutor")

const failure = (operation: string, message: string) => new InfrastructureFailure({ operation, message })
/** Only this OS boundary owns subprocesses. Cancellation closes the scope and reaps the child. */
export const ProcessExecutorLive = Layer.succeed(ProcessExecutor, {
  run: spec => Effect.scoped(Effect.gen(function* () {
    const child = yield* Effect.acquireRelease(Effect.try({
      try: () => Bun.spawn([spec.executable, ...spec.args], {
        cwd: Option.getOrUndefined(spec.cwd), env: spec.inheritEnv ? { ...process.env, ...spec.env } : { ...spec.env },
        stdin: Option.isSome(spec.stdin) ? new TextEncoder().encode(spec.stdin.value) : "ignore",
        stdout: "pipe", stderr: "pipe", detached: process.platform !== "win32",
      }),
      catch: error => failure("spawn", error instanceof Error ? error.message : String(error)),
    }), child => Effect.gen(function* () {
      if (child.exitCode === null) {
        yield* Effect.sync(() => {
          try { if (process.platform !== "win32") process.kill(-child.pid, "SIGTERM"); else child.kill("SIGTERM") } catch { /* Already exited. */ }
        })
        const exited = yield* Effect.promise(() => child.exited).pipe(Effect.timeoutOption("2 seconds"))
        if (Option.isNone(exited)) yield* Effect.sync(() => {
          try { if (process.platform !== "win32") process.kill(-child.pid, "SIGKILL"); else child.kill("SIGKILL") } catch { /* Already exited. */ }
        })
      }
      yield* Effect.promise(() => child.exited).pipe(Effect.timeoutOption("3 seconds"))
    }))
    const collect = (stream: ReadableStream<Uint8Array>) => Stream.fromReadableStream(() => stream, () => failure("read-output", "Subprocess output stream failed")).pipe(
      Stream.runFoldEffect({ bytes: 0, chunks: [] as Uint8Array[] }, (state, chunk) => state.bytes + chunk.byteLength > spec.maxOutputBytes
        ? Effect.fail(failure("read-output", `Subprocess exceeded ${spec.maxOutputBytes} bytes of output`))
        : Effect.succeed({ bytes: state.bytes + chunk.byteLength, chunks: [...state.chunks, chunk] })),
      Effect.map(state => Buffer.concat(state.chunks).toString("utf8")),
    )
    const [stdout, stderr, exitCode] = yield* Effect.all([
      collect(child.stdout), collect(child.stderr), Effect.promise(() => child.exited),
    ], { concurrency: "unbounded" })
    return { stdout, stderr, exitCode }
  })).pipe(Effect.timeoutFail({ duration: spec.timeoutMs, onTimeout: () => failure("timeout", `${spec.executable} exceeded ${spec.timeoutMs}ms`) })),
})

export const command = (executable: string, args: readonly string[], options: Partial<Omit<CommandSpec, "executable" | "args">> = {}) =>
  Effect.flatMap(ProcessExecutor, executor => executor.run(CommandSpec.make({ executable, args, cwd: Option.none(), env: {},
    inheritEnv: true, timeoutMs: 60_000, maxOutputBytes: 8 * 1024 * 1024, stdin: Option.none(), ...options })))
export const checkedCommand = (executable: string, args: readonly string[], options: Partial<Omit<CommandSpec, "executable" | "args">> = {}) =>
  command(executable, args, options).pipe(Effect.flatMap(result => result.exitCode === 0 ? Effect.succeed(result)
    : Effect.fail(failure("command", `${executable} exited ${result.exitCode}: ${result.stderr.slice(-4000)}`))))
