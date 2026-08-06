import { Duration, Effect, Option } from "effect"
import { DaemonSpawnFailed } from "./errors"
import { scopePreHandoffCandidate, type ChildProcessSpawner } from "./child-process"

const spawnFailure = (operation: string, cause: unknown) =>
  new DaemonSpawnFailed({ reason: `${operation}: ${String(cause)}` })

/** Bun implementation of scoped pre-handoff ACN spawning. */
export const BunDetachedChildProcessSpawner: ChildProcessSpawner = {
  spawn: (command) =>
    Effect.uninterruptible(
      Effect.gen(function* () {
        const child = yield* Effect.try({
          try: () =>
            Bun.spawn({
              cmd: Array.from(command),
              detached: true,
              stdio: ["pipe", "ignore", "ignore"],
              env: globalThis.process.env,
            }),
          catch: (cause) => spawnFailure("Failed to spawn Magnitude", cause),
        })
        const exited = Effect.promise(() => child.exited)
        child.unref()

        const signal = (name: NodeJS.Signals) =>
          Effect.try({
            try: () => child.kill(name),
            catch: (cause) => spawnFailure(`Failed to send ${name} to ACN ${child.pid}`, cause),
          }).pipe(Effect.asVoid)

        const waitForExit = (duration: Duration.DurationInput) =>
          exited.pipe(Effect.timeoutOption(duration))
        const stopAndReap = Effect.gen(function* () {
          if (Option.isSome(yield* waitForExit("1 millis"))) return
          yield* signal("SIGTERM")
          if (Option.isSome(yield* waitForExit("2 seconds"))) return
          yield* signal("SIGKILL")
          if (Option.isNone(yield* waitForExit("2 seconds"))) {
            return yield* new DaemonSpawnFailed({
              reason: `Pre-handoff ACN ${child.pid} did not exit after SIGKILL`,
            })
          }
        })

        return yield* scopePreHandoffCandidate({
          pid: child.pid,
          releaseForHandoff: Effect.tryPromise({
            try: async () => {
              child.stdin.write("1")
              await child.stdin.flush()
              await child.stdin.end()
            },
            catch: (cause) => spawnFailure("Failed to hand off ACN bootstrap", cause),
          }),
          stopAndReap,
        })
      }),
    ),
}
