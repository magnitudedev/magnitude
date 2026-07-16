import * as Command from "@effect/platform/Command"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import type { PlatformError } from "@effect/platform/Error"
import { Data, Effect, Fiber, Redacted, Ref, Schema, Scope, Stream } from "effect"

export interface CommandOutput {
  readonly stdout: string
  readonly stderr: string
  readonly exitCode: number
}

export interface CommandOutputParser<A, E> {
  readonly name: string
  readonly parse: (output: CommandOutput) => Effect.Effect<A, E>
}

export const CommandExecutionOperation = Schema.Literal("start", "read-output", "wait")
export type CommandExecutionOperation = Schema.Schema.Type<typeof CommandExecutionOperation>

export class CommandExecutionError extends Data.TaggedError("CommandExecutionError")<{
  readonly executable: string
  readonly operation: CommandExecutionOperation
}> {}

export interface CommandCaptureOptions {
  readonly maxOutputBytes: number
  readonly redactions: readonly Redacted.Redacted<string>[]
}
export const CommandCaptureOptions = {
  Default: { maxOutputBytes: 16 * 1024 * 1024, redactions: [] } satisfies CommandCaptureOptions,
  ProcessDefault: { maxOutputBytes: 64 * 1024, redactions: [] } satisfies CommandCaptureOptions,
} as const

export interface CommandOutputCapture {
  readonly text: Effect.Effect<string>
  readonly completed: Effect.Effect<void, CommandExecutionError>
}

const sanitize = (text: string, redactions: readonly Redacted.Redacted<string>[]): string =>
  redactions.reduce((value, redaction) => value.replaceAll(Redacted.value(redaction), "<redacted>"), text)

const collect = (
  executable: string,
  stream: Stream.Stream<Uint8Array, PlatformError>,
  redactions: readonly Redacted.Redacted<string>[],
  limit: number,
): Effect.Effect<{ readonly output: Ref.Ref<string>; readonly fiber: Fiber.RuntimeFiber<void, CommandExecutionError> }, never, Scope.Scope> => Effect.gen(function* () {
  const output = yield* Ref.make("")
  const fiber = yield* stream.pipe(
    Stream.decodeText(),
    Stream.runForEach((chunk) => Ref.update(output, (current) => `${current}${sanitize(chunk, redactions)}`.slice(-limit))),
    Effect.mapError(() => new CommandExecutionError({ executable, operation: "read-output" })),
    Effect.forkScoped,
  )
  return { output, fiber }
})

export const runCommand = (
  executable: string,
  args: readonly string[],
  options: CommandCaptureOptions,
): Effect.Effect<CommandOutput, CommandExecutionError, CommandExecutor.CommandExecutor> => Effect.scoped(Effect.gen(function* () {
  const process = yield* Command.start(Command.make(executable, ...args)).pipe(
    Effect.mapError(() => new CommandExecutionError({ executable, operation: "start" })),
  )
  const limit = options.maxOutputBytes
  const redactions = options.redactions
  const stdout = yield* collect(executable, process.stdout, redactions, limit)
  const stderr = yield* collect(executable, process.stderr, redactions, limit)
  const exitCode = yield* process.exitCode.pipe(
    Effect.map(Number),
    Effect.mapError(() => new CommandExecutionError({ executable, operation: "wait" })),
  )
  yield* Fiber.join(stdout.fiber)
  yield* Fiber.join(stderr.fiber)
  return { stdout: yield* Ref.get(stdout.output), stderr: yield* Ref.get(stderr.output), exitCode }
}))

export const runParsedCommand = <A, E>(
  executable: string,
  args: readonly string[],
  parser: CommandOutputParser<A, E>,
  options: CommandCaptureOptions,
): Effect.Effect<A, E | CommandExecutionError, CommandExecutor.CommandExecutor> =>
  runCommand(executable, args, options).pipe(Effect.flatMap(parser.parse))

export const captureProcessOutput = (
  executable: string,
  process: CommandExecutor.Process,
  options: CommandCaptureOptions,
): Effect.Effect<CommandOutputCapture, never, Scope.Scope> => Effect.gen(function* () {
  const limit = options.maxOutputBytes
  const redactions = options.redactions
  const stdout = yield* collect(executable, process.stdout, redactions, limit)
  const stderr = yield* collect(executable, process.stderr, redactions, limit)
  return {
    text: Effect.all([Ref.get(stdout.output), Ref.get(stderr.output)]).pipe(Effect.map(([out, err]) => `${out}${err}`.slice(-limit))),
    completed: Effect.all([Fiber.join(stdout.fiber), Fiber.join(stderr.fiber)], { discard: true }),
  }
})
