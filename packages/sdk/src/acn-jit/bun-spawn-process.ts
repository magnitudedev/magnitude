import { Duration, Effect, Option, Schedule } from "effect"
import { AcnEnsuranceFailed } from "./errors"
import { scopeAcnCandidate, type ChildProcessSpawner } from "./child-process"

const spawnFailure = (operation: string, cause: unknown) =>
  new AcnEnsuranceFailed({
    reason: `${operation}: ${cause instanceof Error ? cause.message : String(cause)}`,
  })

/** Bun implementation of a detached, scope-owned ACN candidate tree. */
export const BunDetachedChildProcessSpawner: ChildProcessSpawner = {
  spawn: (command) => Effect.uninterruptible(Effect.gen(function* () {
    const child = yield* Effect.try({
      try: () => Bun.spawn({
        cmd: Array.from(command),
        detached: true,
        stdio: ["pipe", "ignore", "ignore"],
        env: globalThis.process.env,
      }),
      catch: (cause) => spawnFailure("Failed to spawn Magnitude", cause),
    })
    const exited = Effect.promise(() => child.exited)
    child.unref()

    const treeAbsent = Effect.sync(() => {
      if (process.platform === "win32") return child.exitCode !== null
      try {
        process.kill(-child.pid, 0)
        return false
      } catch (cause) {
        if (cause instanceof Error && "code" in cause) {
          if (cause.code === "ESRCH") return true
          if (cause.code === "EPERM") return false
        }
        throw cause
      }
    })
    const waitForTreeAbsence = (duration: Duration.DurationInput) => treeAbsent.pipe(
      Effect.flatMap((absent) => absent ? Effect.void : Effect.fail("TreePresent" as const)),
      Effect.retry(Schedule.spaced(Duration.millis(20))),
      Effect.timeoutOption(duration),
      Effect.catchAll(() => Effect.succeed(Option.none())),
      Effect.map(Option.isSome),
    )
    const signalTree = (signal: NodeJS.Signals) => Effect.try({
      try: () => {
        try {
          if (process.platform === "win32") child.kill(signal)
          else process.kill(-child.pid, signal)
        } catch (cause) {
          if (cause instanceof Error && "code" in cause && cause.code === "ESRCH") return
          throw cause
        }
      },
      catch: (cause) => spawnFailure(
        `Failed to send ${signal} to ACN candidate tree ${child.pid}`,
        cause,
      ),
    }).pipe(
      Effect.asVoid,
    )
    const stopAndReap = Effect.gen(function* () {
      if (yield* treeAbsent) return
      yield* signalTree("SIGTERM")
      if (yield* waitForTreeAbsence(Duration.seconds(2))) return
      yield* signalTree("SIGKILL")
      if (!(yield* waitForTreeAbsence(Duration.seconds(2)))) {
        return yield* new AcnEnsuranceFailed({
          reason: `ACN candidate tree ${child.pid} did not exit after SIGKILL`,
        })
      }
    })

    return yield* scopeAcnCandidate({
      pid: child.pid,
      exited,
      stopAndReap,
      releaseParentChannel: Effect.tryPromise({
        try: async () => {
          await child.stdin.end()
        },
        catch: (cause) => spawnFailure("Failed to release ACN parent channel", cause),
      }),
    })
  })),
}
