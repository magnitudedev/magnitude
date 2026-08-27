import { spawn, type ChildProcess } from "node:child_process"
import { constants } from "node:os"
import { Deferred, Duration, Effect, Option, Schema } from "effect"

export interface InteractiveProcess {
  readonly executable: string
  readonly args: ReadonlyArray<string>
  readonly environment: Readonly<Record<string, string | undefined>>
  readonly workingDirectory?: string
}

export type InteractiveProcessTermination =
  | { readonly _tag: "Exited"; readonly code: number }
  | { readonly _tag: "Signaled"; readonly signal: NodeJS.Signals }

export class InteractiveProcessFailed extends Schema.TaggedError<InteractiveProcessFailed>()(
  "InteractiveProcessFailed",
  {
    operation: Schema.Literal("spawn", "run"),
    message: Schema.String,
  },
) {}

export const interactiveProcessExitCode = (
  termination: InteractiveProcessTermination,
): number => termination._tag === "Exited"
  ? termination.code
  : 128 + (constants.signals[termination.signal] ?? 0)

interface RunningInteractiveProcess {
  readonly child: ChildProcess
  readonly termination: Deferred.Deferred<
    InteractiveProcessTermination,
    InteractiveProcessFailed
  >
}

const messageFrom = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause)

const definedEnvironment = (
  environment: Readonly<Record<string, string | undefined>>,
): Record<string, string> => Object.fromEntries(
  Object.entries(environment).filter((entry): entry is [string, string] =>
    entry[1] !== undefined
  ),
)

const acquireInteractiveProcess = (
  input: InteractiveProcess,
): Effect.Effect<RunningInteractiveProcess, InteractiveProcessFailed> => Effect.gen(function* () {
  const termination = yield* Deferred.make<
    InteractiveProcessTermination,
    InteractiveProcessFailed
  >()

  const child = yield* Effect.async<ChildProcess, InteractiveProcessFailed>((resume) => {
    let acquired = false
    let handle: ChildProcess

    try {
      handle = spawn(input.executable, [...input.args], {
        cwd: input.workingDirectory,
        detached: false,
        env: definedEnvironment(input.environment),
        shell: false,
        stdio: "inherit",
      })
    } catch (cause) {
      resume(Effect.fail(new InteractiveProcessFailed({
        operation: "spawn",
        message: messageFrom(cause),
      })))
      return
    }

    const onError = (cause: Error) => {
      const failure = new InteractiveProcessFailed({
        operation: acquired ? "run" : "spawn",
        message: messageFrom(cause),
      })
      if (acquired) {
        Deferred.unsafeDone(termination, Effect.fail(failure))
      } else {
        resume(Effect.fail(failure))
      }
    }
    const onExit = (code: number | null, signal: NodeJS.Signals | null) => {
      Deferred.unsafeDone(
        termination,
        Effect.succeed(code === null
          ? { _tag: "Signaled", signal: signal! }
          : { _tag: "Exited", code }),
      )
    }
    const onSpawn = () => {
      acquired = true
      resume(Effect.succeed(handle))
    }

    handle.once("error", onError)
    handle.once("exit", onExit)
    handle.once("spawn", onSpawn)

    return Effect.sync(() => {
      if (!acquired) {
        handle.removeListener("error", onError)
        handle.removeListener("exit", onExit)
        handle.removeListener("spawn", onSpawn)
        if (handle.exitCode === null && handle.signalCode === null) {
          try {
            handle.kill("SIGTERM")
          } catch {
            // Acquisition interruption is already terminal for the caller.
          }
        }
      }
    })
  })

  return { child, termination }
})

const FORWARDED_SIGNALS: ReadonlyArray<NodeJS.Signals> = process.platform === "win32"
  ? ["SIGINT", "SIGTERM", "SIGBREAK"]
  : ["SIGINT", "SIGTERM", "SIGHUP"]

const awaitWithSignalForwarding = ({ child, termination }: RunningInteractiveProcess) =>
  Effect.acquireUseRelease(
    Effect.sync(() => {
      const listeners = new Map<NodeJS.Signals, () => void>()
      for (const signal of FORWARDED_SIGNALS) {
        const listener = () => {
          try {
            child.kill(signal)
          } catch {
            // The child may have exited between signal delivery and forwarding.
          }
        }
        listeners.set(signal, listener)
        process.on(signal, listener)
      }
      return listeners
    }),
    () => Deferred.await(termination),
    (listeners) => Effect.sync(() => {
      for (const [signal, listener] of listeners) {
        process.removeListener(signal, listener)
      }
    }),
  )

const terminateAndReap = ({ child, termination }: RunningInteractiveProcess): Effect.Effect<void> =>
  Effect.gen(function* () {
    if (yield* Deferred.isDone(termination)) return

    yield* Effect.sync(() => {
      try {
        child.kill("SIGTERM")
      } catch {
        // Waiting below remains the authority for whether the child exited.
      }
    })
    const graceful = yield* Deferred.await(termination).pipe(
      Effect.ignore,
      Effect.timeoutOption(Duration.seconds(2)),
    )
    if (Option.isSome(graceful)) return

    yield* Effect.sync(() => {
      try {
        child.kill("SIGKILL")
      } catch {
        // The child may have exited at the timeout boundary.
      }
    })
    yield* Deferred.await(termination).pipe(
      Effect.ignore,
      Effect.timeout(Duration.seconds(2)),
      Effect.ignore,
    )
  })

/**
 * Runs one terminal-owning child without creating a new process group.
 *
 * The caller must release any active terminal renderer before invoking this
 * Effect. The child inherits the terminal and receives terminal-generated
 * resize and job-control signals through the existing foreground process group.
 */
export const runInteractiveProcess = (
  input: InteractiveProcess,
): Effect.Effect<InteractiveProcessTermination, InteractiveProcessFailed> =>
  Effect.acquireUseRelease(
    acquireInteractiveProcess(input),
    awaitWithSignalForwarding,
    terminateAndReap,
  )
