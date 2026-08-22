import { Duration, Effect, Option } from "effect"
import {
  AcnCandidateBootstrapProcessExitUnproven,
  AcnCandidateBootstrapProcessStopFailed,
  AcnCandidateParentChannelReleaseFailed,
  AcnCandidateSpawnFailed,
} from "./errors"
import { scopeAcnCandidate, type ChildProcessSpawner } from "./child-process"

const MAXIMUM_STDERR_BYTES = 64 * 1024

const readStderrTail = async (
  stream: ReadableStream<Uint8Array>,
): Promise<string> => {
  const reader = stream.getReader()
  let tail = new Uint8Array(0)
  while (true) {
    const next = await reader.read()
    if (next.done) break
    const combined = new Uint8Array(tail.length + next.value.length)
    combined.set(tail)
    combined.set(next.value, tail.length)
    tail = combined.length <= MAXIMUM_STDERR_BYTES
      ? combined
      : combined.slice(combined.length - MAXIMUM_STDERR_BYTES)
  }
  return new TextDecoder().decode(tail).trim()
}

const messageFrom = (cause: unknown): string => cause instanceof Error ? cause.message : String(cause)

/** Bun implementation of a detached, scope-owned ACN candidate process group. */
export const BunDetachedChildProcessSpawner: ChildProcessSpawner = {
  spawn: (command) => Effect.uninterruptible(Effect.gen(function* () {
    const child = yield* Effect.try({
      try: () => Bun.spawn({
        cmd: Array.from(command),
        detached: true,
        stdio: ["pipe", "ignore", "pipe"],
        env: globalThis.process.env,
      }),
      catch: (cause) => new AcnCandidateSpawnFailed({ message: messageFrom(cause) }),
    })
    const stderr = readStderrTail(child.stderr)
    const exited = Effect.promise(async () => ({
      code: await child.exited,
      stderr: await stderr,
    }))
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
        try: async () => {
          await child.stdin.end()
        },
        catch: (cause) => new AcnCandidateParentChannelReleaseFailed({
          pid: child.pid,
          message: messageFrom(cause),
        }),
      }),
    })
  })),
}
