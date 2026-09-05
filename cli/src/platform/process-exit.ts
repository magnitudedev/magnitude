import { Deferred, Effect, Scope } from "effect"

export type ProcessExitRequest =
  | {
      readonly _tag: "Signal"
      readonly signal: "SIGINT" | "SIGTERM" | "SIGHUP" | "beforeExit"
    }
  | {
      readonly _tag: "Fatal"
      readonly label: "Uncaught exception" | "Unhandled rejection"
      readonly error: unknown
    }

export interface ProcessExitSource {
  readonly await: Effect.Effect<ProcessExitRequest>
}

export type UntilProcessExit<A> =
  | { readonly _tag: "Completed"; readonly value: A }
  | { readonly _tag: "Exit"; readonly request: ProcessExitRequest }

/**
 * The one place work yields to a user exit request. Whichever side completes first
 * wins, failure included: a failing effect propagates immediately instead of waiting
 * for a signal that may never come, and an exit request interrupts the effect so its
 * finalizers run.
 */
export const untilProcessExit = <A, E, R>(
  effect: Effect.Effect<A, E, R>,
  source: ProcessExitSource,
): Effect.Effect<UntilProcessExit<A>, E, R> => Effect.raceFirst(
  effect.pipe(Effect.map((value): UntilProcessExit<A> => ({ _tag: "Completed", value }))),
  source.await.pipe(Effect.map((request): UntilProcessExit<A> => ({ _tag: "Exit", request }))),
)

export const restoreTerminalState = (): void => {
  process.stdout.write([
    "\x1b[?1000l",
    "\x1b[?1002l",
    "\x1b[?1003l",
    "\x1b[?1006l",
    "\x1b[?1004l",
    "\x1b[?2004l",
    "\x1b[?25h",
  ].join(""))
}

export const makeProcessExitSource: Effect.Effect<
  ProcessExitSource,
  never,
  Scope.Scope
> = Effect.gen(function* () {
  const request = yield* Deferred.make<ProcessExitRequest>()
  const complete = (value: ProcessExitRequest): void => {
    Deferred.unsafeDone(request, Effect.succeed(value))
  }

  const onSigint = () => complete({ _tag: "Signal", signal: "SIGINT" })
  const onSigterm = () => complete({ _tag: "Signal", signal: "SIGTERM" })
  const onSighup = () => complete({ _tag: "Signal", signal: "SIGHUP" })
  const onBeforeExit = () => complete({ _tag: "Signal", signal: "beforeExit" })
  const onExit = () => restoreTerminalState()
  const onUncaughtException = (error: unknown) => complete({
    _tag: "Fatal",
    label: "Uncaught exception",
    error,
  })
  const onUnhandledRejection = (error: unknown) => complete({
    _tag: "Fatal",
    label: "Unhandled rejection",
    error,
  })

  yield* Effect.acquireRelease(
    Effect.sync(() => {
      process.on("SIGINT", onSigint)
      process.on("SIGTERM", onSigterm)
      process.on("SIGHUP", onSighup)
      process.on("beforeExit", onBeforeExit)
      process.on("exit", onExit)
      process.on("uncaughtException", onUncaughtException)
      process.on("unhandledRejection", onUnhandledRejection)
    }),
    () => Effect.sync(() => {
      process.off("SIGINT", onSigint)
      process.off("SIGTERM", onSigterm)
      process.off("SIGHUP", onSighup)
      process.off("beforeExit", onBeforeExit)
      process.off("exit", onExit)
      process.off("uncaughtException", onUncaughtException)
      process.off("unhandledRejection", onUnhandledRejection)
    }),
  )

  return { await: Deferred.await(request) }
})
