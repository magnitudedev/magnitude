import { randomUUID } from "node:crypto"
import {
  Cause,
  Data,
  Deferred,
  Either,
  Effect,
  Exit,
  Fiber,
  Layer,
  Logger,
  Option,
  Runtime,
  Schema,
  Stream,
} from "effect"
import {
  HeadlessSessionClient,
  HeadlessSessionIdSchema,
  makeHeadlessSessionClientLayer,
  runHeadlessSession,
  type HeadlessSessionId,
  type HeadlessSessionResult,
  type Platform,
} from "@magnitudedev/client-common"
import type { SessionOptions } from "@magnitudedev/sdk"
import type { SessionStart } from "../app"
import {
  createHeadlessOutputRenderer,
  renderInterruptedSummary,
  renderUsageSummary,
  sanitizeHeadlessText,
} from "../headless/output"

const HeadlessLoggingLayer = Logger.replace(Logger.defaultLogger, Logger.none)
const DEFAULT_FIBER_INTERRUPT_TIMEOUT_MS = 1_000
const DEFAULT_LATE_PLATFORM_ACQUISITION_TIMEOUT_MS = 50
const DEFAULT_PLATFORM_SHUTDOWN_TIMEOUT_MS = 1_000
const DEFAULT_SIGNAL_OUTPUT_TIMEOUT_MS = 250

const SessionStartSchema = Schema.Union(
  Schema.TaggedStruct("new", {}),
  Schema.TaggedStruct("latest", {}),
  Schema.TaggedStruct("resume", {
    sessionId: HeadlessSessionIdSchema,
  }),
)

const RunHeadlessOptionsSchema = Schema.Struct({
  debug: Schema.Boolean,
  autopilot: Schema.Boolean,
  initialPrompt: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  sessionStart: SessionStartSchema,
  disableShellSafeguards: Schema.Boolean,
  disableCwdSafeguards: Schema.Boolean,
  atifPath: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  goal: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  solo: Schema.Boolean,
  systemOverride: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  setup: Schema.optionalWith(Schema.Boolean, { as: "Option", exact: true }),
})
type DecodedRunHeadlessOptions = Schema.Schema.Type<typeof RunHeadlessOptionsSchema>

export interface RunHeadlessOptions {
  debug: boolean
  autopilot: boolean
  initialPrompt?: string
  sessionStart: SessionStart
  disableShellSafeguards: boolean
  disableCwdSafeguards: boolean
  atifPath?: string
  goal?: string
  solo: boolean
  systemOverride?: string
  setup?: boolean
}

interface TextWriter {
  readonly write: (
    chunk: string,
    callback: (error?: Error | null) => void,
  ) => unknown
  readonly once?: (event: "error" | "close", listener: (error?: unknown) => void) => unknown
  readonly off?: (event: "error" | "close", listener: (error?: unknown) => void) => unknown
}

interface HeadlessSignalTarget {
  readonly once: (signal: "SIGINT" | "SIGTERM", listener: () => void) => unknown
  readonly removeListener: (signal: "SIGINT" | "SIGTERM", listener: () => void) => unknown
}

export interface RunHeadlessDependencies {
  readonly createPlatform: () => Promise<Platform>
  readonly sessionClientLayer?: Layer.Layer<HeadlessSessionClient>
  readonly stdout?: TextWriter
  readonly stderr?: TextWriter
  readonly cwd?: () => string
  readonly registerSignalHandlers?: boolean
  readonly signalTarget?: HeadlessSignalTarget
  readonly fiberInterruptTimeoutMs?: number
  readonly latePlatformAcquisitionTimeoutMs?: number
  readonly platformShutdownTimeoutMs?: number
  readonly signalOutputTimeoutMs?: number
  readonly onTerminationSignal?: (exitCode: number) => void
  readonly makeSessionId?: () => HeadlessSessionId
}

export class HeadlessDaemonStartFailed extends Data.TaggedError("HeadlessDaemonStartFailed")<{
  readonly message: string
}> {}

export class HeadlessPlatformOperationFailed extends Data.TaggedError("HeadlessPlatformOperationFailed")<{
  readonly operation: "acquire" | "shutdown"
  readonly message: string
}> {}

export class HeadlessOutputWriteFailed extends Data.TaggedError("HeadlessOutputWriteFailed")<{
  readonly message: string
}> {}

type BoundedEffectResult<T> =
  | { readonly _tag: "succeeded"; readonly value: T }
  | { readonly _tag: "failed"; readonly message: string }
  | { readonly _tag: "timed-out" }

interface ValidHeadlessRequest {
  readonly initial: { readonly type: "message" | "goal"; readonly content: string }
  readonly options: SessionOptions
}

export function runHeadless(
  options: RunHeadlessOptions,
  dependencies: RunHeadlessDependencies,
): Effect.Effect<number> {
  const stdout = dependencies.stdout ?? process.stdout
  const stderr = dependencies.stderr ?? process.stderr
  return Effect.suspend(() => {
    const validation = validateHeadlessOptions(options)
    if (validation._tag === "invalid") {
      return writeHeadlessLine(stderr, `Error: ${validation.message}`).pipe(Effect.as(2))
    }

    return runValidatedHeadless(validation.request, dependencies, stdout, stderr)
  }).pipe(
    Effect.catchAllCause((cause) => writeHeadlessLine(stderr, `Error: ${formatFailureCause(cause)}`).pipe(
      Effect.as(1),
      Effect.catchAll(() => Effect.succeed(1)),
    )),
  )
}

function runValidatedHeadless(
  request: ValidHeadlessRequest,
  dependencies: RunHeadlessDependencies,
  stdout: TextWriter,
  stderr: TextWriter,
): Effect.Effect<number, HeadlessOutputWriteFailed> {
  return Effect.gen(function* () {
    let platform: Platform | null = null
    let exitCode = 1
    let signalExitCode: number | null = null
    const terminationSignal = yield* Deferred.make<number>()
    const runtime = yield* Effect.runtime<never>()
    const runFork = Runtime.runFork(runtime)
    const signalTarget = dependencies.signalTarget ?? process
    const interrupt = (code: number) => {
      if (signalExitCode !== null) return
      signalExitCode = code
      try {
        dependencies.onTerminationSignal?.(code)
      } catch {
        // Signal ownership must not depend on a scheduling hook.
      }
      runFork(Deferred.succeed(terminationSignal, code))
    }
    const onSigint = () => interrupt(130)
    const onSigterm = () => interrupt(143)
    const registerSignalHandlers = dependencies.registerSignalHandlers ?? true
    if (registerSignalHandlers) {
      signalTarget.once("SIGINT", onSigint)
      signalTarget.once("SIGTERM", onSigterm)
    }

    const platformFiber = yield* Effect.tryPromise({
      try: dependencies.createPlatform,
      catch: (error) => new HeadlessPlatformOperationFailed({
        operation: "acquire",
        message: errorMessage(error),
      }),
    }).pipe(Effect.forkDaemon)

    try {
      const acquisition = yield* Effect.raceFirst(
        Fiber.await(platformFiber).pipe(
          Effect.map((value) => ({ _tag: "platform" as const, value })),
        ),
        Deferred.await(terminationSignal).pipe(
          Effect.map((code) => ({ _tag: "signal" as const, code })),
        ),
      )
      if (acquisition._tag === "signal") {
        exitCode = acquisition.code
        const late = yield* Fiber.await(platformFiber).pipe(Effect.timeoutOption(
          dependencies.latePlatformAcquisitionTimeoutMs
            ?? DEFAULT_LATE_PLATFORM_ACQUISITION_TIMEOUT_MS,
        ))
        if (Option.isSome(late) && Exit.isSuccess(late.value)) {
          platform = late.value.value
        } else if (Option.isNone(late)) {
          yield* Fiber.await(platformFiber).pipe(
            Effect.flatMap((lateExit) => Exit.isSuccess(lateExit)
              ? shutdownPlatformWithin(
                  lateExit.value,
                  dependencies.platformShutdownTimeoutMs ?? DEFAULT_PLATFORM_SHUTDOWN_TIMEOUT_MS,
                ).pipe(Effect.asVoid)
              : Effect.void),
            Effect.forkDaemon,
          )
        }
      } else if (Exit.isFailure(acquisition.value)) {
        yield* writeHeadlessLine(stderr, `Error: ${formatFailureCause(acquisition.value.cause)}`)
        exitCode = 1
      } else {
        platform = acquisition.value.value
        if (signalExitCode !== null) {
          exitCode = signalExitCode
        } else {
          const renderer = createHeadlessOutputRenderer()
          const sessionId = (dependencies.makeSessionId ?? (() =>
            HeadlessSessionIdSchema.make(randomUUID())))()
          const sessionClientLayer = dependencies.sessionClientLayer
            ?? makeHeadlessSessionClientLayer(platform.protocolLayer)
          const program = awaitDaemonReady(platform).pipe(
            Effect.zipRight(writeHeadlessLine(stderr, `Session: ${sessionId}`)),
            Effect.zipRight(runHeadlessSession({
              sessionId,
              cwd: (dependencies.cwd ?? process.cwd)(),
              initial: request.initial,
              options: request.options,
            }, {
              onSnapshot: (snapshot) => {
                const output = renderer.handleSnapshot(snapshot)
                return Effect.forEach(
                  output.lines,
                  (line) => writeHeadlessLine(stdout, line),
                  { discard: true },
                )
              },
            })),
            Effect.provide(Layer.mergeAll(sessionClientLayer, HeadlessLoggingLayer)),
          )

          const commandStartedAt = yield* Effect.clockWith((clock) => clock.currentTimeMillis)
          const fiber = yield* program.pipe(Effect.forkDaemon)
          const completion = yield* Effect.raceFirst(
            Fiber.await(fiber).pipe(Effect.map((value) => ({ _tag: "exit" as const, value }))),
            Deferred.await(terminationSignal).pipe(
              Effect.map((code) => ({ _tag: "signal" as const, code })),
            ),
          )
          let effectExit: Exit.Exit<HeadlessSessionResult, unknown> | null = null
          if (completion._tag === "exit") {
            effectExit = completion.value
          } else {
            yield* Fiber.interrupt(fiber).pipe(Effect.forkDaemon)
            const interrupted = yield* Fiber.await(fiber).pipe(Effect.timeoutOption(
              dependencies.fiberInterruptTimeoutMs ?? DEFAULT_FIBER_INTERRUPT_TIMEOUT_MS,
            ))
            if (Option.isSome(interrupted)) effectExit = interrupted.value
          }

          if (signalExitCode !== null) {
            const commandFinishedAt = yield* Effect.clockWith((clock) => clock.currentTimeMillis)
            yield* writeHeadlessLine(stdout, renderInterruptedSummary(
              Math.max(0, commandFinishedAt - commandStartedAt),
              renderer.getToolCount(),
            )).pipe(
              Effect.timeoutOption(
                dependencies.signalOutputTimeoutMs ?? DEFAULT_SIGNAL_OUTPUT_TIMEOUT_MS,
              ),
              Effect.asVoid,
            )
            exitCode = signalExitCode
          } else if (effectExit === null) {
            yield* writeHeadlessLine(stderr, "Error: headless execution stopped without a terminal result")
            exitCode = 1
          } else if (Exit.isSuccess(effectExit)) {
            const result = effectExit.value
            if (result.status === "interrupted") {
              yield* writeHeadlessLine(stdout, renderInterruptedSummary(result.elapsedMs, renderer.getToolCount()))
              exitCode = 130
            } else {
              const success = result.status === "completed"
              yield* writeHeadlessLine(stdout, renderUsageSummary(result.elapsedMs, renderer.getToolCount(), success))
              exitCode = success ? 0 : 1
            }
          } else {
            yield* writeHeadlessLine(stderr, `Error: ${formatFailureCause(effectExit.cause)}`)
            exitCode = Cause.isInterruptedOnly(effectExit.cause) ? 130 : 1
          }
        }
      }
    } finally {
      let shutdownDiagnostic: string | null = null
      if (platform) {
        const shutdown = yield* shutdownPlatformWithin(
          platform,
          dependencies.platformShutdownTimeoutMs ?? DEFAULT_PLATFORM_SHUTDOWN_TIMEOUT_MS,
        )
        if (shutdown._tag === "timed-out") {
          shutdownDiagnostic = "Error: timed out while releasing the daemon client"
          if (exitCode === 0) exitCode = 1
        } else if (shutdown._tag === "failed") {
          shutdownDiagnostic = `Error: failed to release the daemon client: ${errorMessage(shutdown.message)}`
          if (exitCode === 0) exitCode = 1
        }
      }
      if (registerSignalHandlers) {
        signalTarget.removeListener("SIGINT", onSigint)
        signalTarget.removeListener("SIGTERM", onSigterm)
      }
      if (signalExitCode !== null) exitCode = signalExitCode
      if (shutdownDiagnostic !== null) {
        const write = writeHeadlessLine(stderr, shutdownDiagnostic)
        yield* signalExitCode === null
          ? write
          : write.pipe(
              Effect.timeoutOption(
                dependencies.signalOutputTimeoutMs ?? DEFAULT_SIGNAL_OUTPUT_TIMEOUT_MS,
              ),
              Effect.asVoid,
            )
      }
    }

    return exitCode
  })
}

function shutdownPlatformWithin(
  platform: Platform,
  timeoutMs: number,
): Effect.Effect<BoundedEffectResult<void>> {
  return settleEffectWithin(Effect.tryPromise({
    try: platform.shutdown,
    catch: (error) => new HeadlessPlatformOperationFailed({
      operation: "shutdown",
      message: errorMessage(error),
    }),
  }).pipe(Effect.asVoid), timeoutMs)
}

function settleEffectWithin<T>(
  effect: Effect.Effect<T, unknown>,
  timeoutMs: number,
): Effect.Effect<BoundedEffectResult<T>> {
  return effect.pipe(
    Effect.match({
      onFailure: (error): BoundedEffectResult<T> => ({ _tag: "failed", message: errorMessage(error) }),
      onSuccess: (value): BoundedEffectResult<T> => ({ _tag: "succeeded", value }),
    }),
    Effect.timeoutOption(Math.max(0, timeoutMs)),
    Effect.map(Option.match({
      onNone: (): BoundedEffectResult<T> => ({ _tag: "timed-out" }),
      onSome: (result) => result,
    })),
  )
}

function writeHeadlessLine(
  writer: TextWriter,
  line: string,
): Effect.Effect<void, HeadlessOutputWriteFailed> {
  const output = `${sanitizeHeadlessText(line)}\n`
  return Effect.async<void, HeadlessOutputWriteFailed>((resume) => {
    let settled = false
    let pending: ReturnType<typeof setImmediate> | null = null
    const removeListeners = () => {
      writer.off?.("error", onError)
      writer.off?.("close", onClose)
    }
    const settle = (effect: Effect.Effect<void, HeadlessOutputWriteFailed>) => {
      if (settled) return
      settled = true
      if (pending !== null) clearImmediate(pending)
      removeListeners()
      resume(effect)
    }
    const settleAfterIoEvents = (effect: Effect.Effect<void, HeadlessOutputWriteFailed>) => {
      if (settled || pending !== null) return
      pending = setImmediate(() => settle(effect))
    }
    function onError(error?: unknown) {
      settle(Effect.fail(new HeadlessOutputWriteFailed({ message: errorMessage(error) })))
    }
    function onClose() {
      settle(Effect.fail(new HeadlessOutputWriteFailed({
        message: "output stream closed before write completed",
      })))
    }

    writer.once?.("error", onError)
    writer.once?.("close", onClose)
    try {
      writer.write(output, (error) => {
        settleAfterIoEvents(error
          ? Effect.fail(new HeadlessOutputWriteFailed({ message: errorMessage(error) }))
          : Effect.void)
      })
    } catch (error) {
      settle(Effect.fail(new HeadlessOutputWriteFailed({ message: errorMessage(error) })))
    }

    return Effect.sync(() => {
      settled = true
      if (pending !== null) clearImmediate(pending)
      removeListeners()
    })
  })
}

function validateHeadlessOptions(input: unknown):
  | { readonly _tag: "valid"; readonly request: ValidHeadlessRequest }
  | { readonly _tag: "invalid"; readonly message: string } {
  const decoded = Schema.decodeUnknownEither(RunHeadlessOptionsSchema, {
    onExcessProperty: "error",
  })(input)
  if (Either.isLeft(decoded)) {
    return { _tag: "invalid", message: "invalid --headless options" }
  }
  return validateDecodedHeadlessOptions(decoded.right)
}

function validateDecodedHeadlessOptions(options: DecodedRunHeadlessOptions):
  | { readonly _tag: "valid"; readonly request: ValidHeadlessRequest }
  | { readonly _tag: "invalid"; readonly message: string } {
  if (Option.getOrElse(options.setup, () => false)) {
    return { _tag: "invalid", message: "--setup requires the interactive TUI and cannot be used with --headless" }
  }
  if (options.autopilot) {
    return { _tag: "invalid", message: "--autopilot is not currently supported with --headless" }
  }
  if (options.sessionStart._tag !== "new") {
    return {
      _tag: "invalid",
      message: "--resume is not supported with --headless because headless runtime options are fixed when a session is created",
    }
  }
  if (Option.isSome(options.initialPrompt) && Option.isSome(options.goal)) {
    return { _tag: "invalid", message: "--prompt and --goal are mutually exclusive in --headless mode" }
  }

  const initial = Option.isSome(options.goal)
    ? { type: "goal" as const, content: options.goal.value }
    : Option.isSome(options.initialPrompt)
      ? { type: "message" as const, content: options.initialPrompt.value }
      : null
  if (!initial || initial.content.trim().length === 0) {
    return { _tag: "invalid", message: "--headless requires a non-empty --prompt or --goal" }
  }

  const sessionOptions: SessionOptions = {
    headless: true,
    solo: options.solo,
    disableShellSafeguards: options.disableShellSafeguards,
    disableCwdSafeguards: options.disableCwdSafeguards,
    ...(Option.isNone(options.atifPath) ? {} : { atifPath: options.atifPath.value }),
    ...(Option.isNone(options.systemOverride)
      ? {}
      : { systemPromptOverride: options.systemOverride.value }),
  }
  return { _tag: "valid", request: { initial, options: sessionOptions } }
}

function awaitDaemonReady(platform: Platform): Effect.Effect<void, HeadlessDaemonStartFailed> {
  return platform.acnStartup.prepare.pipe(
    Effect.flatMap((initial) => {
      if (initial._tag === "Ready") return Effect.void
      if (initial._tag === "Failed") {
        return new HeadlessDaemonStartFailed({
          message: `ACN ${initial.stage} failed: ${initial.message}`,
        })
      }
      return platform.acnStartup.state.changes.pipe(
        Stream.filter((state) => state._tag === "Ready" || state._tag === "Failed"),
        Stream.runHead,
        Effect.flatMap(Option.match({
          onNone: () => new HeadlessDaemonStartFailed({
            message: "ACN startup ended before the daemon became ready",
          }),
          onSome: (state) => state._tag === "Ready"
            ? Effect.void
            : new HeadlessDaemonStartFailed({
                message: `ACN ${state.stage} failed: ${state.message}`,
              }),
        })),
      )
    }),
  )
}

function formatFailureCause(cause: Cause.Cause<unknown>): string {
  const failure = Cause.failureOption(cause)
  return Option.match(failure, {
    onNone: () => Cause.pretty(cause),
    onSome: errorMessage,
  })
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  if (typeof error === "object" && error !== null && "message" in error) {
    const message = Reflect.get(error, "message")
    if (typeof message === "string" && message.length > 0) return message
  }
  return String(error)
}
