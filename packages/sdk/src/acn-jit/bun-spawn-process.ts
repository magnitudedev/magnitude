import { Deferred, Duration, Effect, Fiber, Option, Ref, Stream } from "effect"
import {
  AcnCandidateBootstrapProcessExitUnproven,
  AcnCandidateBootstrapProcessStopFailed,
  AcnCandidateParentChannelReleaseFailed,
  AcnCandidateSpawnFailed,
} from "./errors"
import {
  scopeAcnCandidate,
  type AcnCandidateExit,
  type ChildProcessSpawner,
} from "./child-process"

const MAXIMUM_STDERR_BYTES = 64 * 1024
const STDERR_DRAIN_GRACE = Duration.millis(100)

const appendStderrTail = (
  tail: Uint8Array<ArrayBufferLike>,
  chunk: Uint8Array<ArrayBufferLike>,
): Uint8Array<ArrayBufferLike> => {
  if (chunk.length === 0) return tail
  if (chunk.length >= MAXIMUM_STDERR_BYTES) {
    return chunk.slice(chunk.length - MAXIMUM_STDERR_BYTES)
  }
  const retainedTail = tail.slice(Math.max(0, tail.length - (MAXIMUM_STDERR_BYTES - chunk.length)))
  const combined = new Uint8Array(retainedTail.length + chunk.length)
  combined.set(retainedTail)
  combined.set(chunk, retainedTail.length)
  return combined
}

const messageFrom = (cause: unknown): string => cause instanceof Error ? cause.message : String(cause)

/** Bun implementation of a detached, scope-owned ACN candidate process group. */
export const BunDetachedChildProcessSpawner: ChildProcessSpawner = {
  spawn: (command) => Effect.uninterruptible(Effect.gen(function* () {
    const rootExit = yield* Deferred.make<number>()
    const child = yield* Effect.try({
      try: () => Bun.spawn({
        cmd: Array.from(command),
        detached: true,
        stdio: ["pipe", "ignore", "pipe"],
        env: globalThis.process.env,
        onExit: (_process, exitCode, signalCode) => {
          Deferred.unsafeDone(
            rootExit,
            Effect.succeed(exitCode ?? (signalCode === null ? 1 : 128 + signalCode)),
          )
        },
      }),
      catch: (cause) => new AcnCandidateSpawnFailed({ message: messageFrom(cause) }),
    })
    const stderrTail = yield* Ref.make<Uint8Array<ArrayBufferLike>>(new Uint8Array(0))
    const stderrComplete = yield* Deferred.make<void>()
    const stderrDrain = yield* Stream.fromReadableStream({
      evaluate: () => child.stderr,
      onError: () => undefined,
    }).pipe(
      Stream.runForEach((chunk) => Ref.update(stderrTail, (tail) => appendStderrTail(tail, chunk))),
      Effect.ignore,
      Effect.interruptible,
      Effect.ensuring(Deferred.succeed(stderrComplete, undefined)),
      Effect.forkDaemon,
    )
    const exitResult = yield* Deferred.make<AcnCandidateExit>()
    const exitObserver = yield* Effect.gen(function* () {
      const code = yield* Deferred.await(rootExit)
      const drained = yield* Deferred.await(stderrComplete).pipe(
        Effect.timeoutOption(STDERR_DRAIN_GRACE),
      )
      if (Option.isNone(drained)) yield* Fiber.interruptFork(stderrDrain)
      const stderr = new TextDecoder().decode(yield* Ref.get(stderrTail)).trim()
      yield* Deferred.succeed(exitResult, { code, stderr })
    }).pipe(Effect.interruptible, Effect.forkDaemon)
    // A descendant may retain stderr after the candidate root is admitted.
    // Neither diagnostic capture nor exit observation may keep scope closure alive.
    yield* Effect.addFinalizer(() => Effect.all([
      Fiber.interruptFork(exitObserver),
      Fiber.interruptFork(stderrDrain),
    ], { discard: true }))
    const exited = Deferred.await(exitResult)
    child.unref()

    const stopBootstrapProcess = Effect.gen(function* () {
      if (child.exitCode === null) {
        yield* Effect.try({
          try: () => { child.kill("SIGKILL") },
          catch: (cause) => new AcnCandidateBootstrapProcessStopFailed({
            pid: child.pid,
            message: messageFrom(cause),
          }),
        })
      }
      const stopped = yield* exited.pipe(Effect.timeoutOption(Duration.seconds(2)))
      if (Option.isNone(stopped)) {
        return yield* new AcnCandidateBootstrapProcessExitUnproven({ pid: child.pid })
      }
    })

    return yield* scopeAcnCandidate({
      pid: child.pid,
      exited,
      stopBootstrapProcess,
      releaseParentChannel: Effect.tryPromise({
        try: () => Promise.resolve(child.stdin.end()),
        catch: (cause) => new AcnCandidateParentChannelReleaseFailed({
          pid: child.pid,
          message: messageFrom(cause),
        }),
      }),
    })
  })),
}
