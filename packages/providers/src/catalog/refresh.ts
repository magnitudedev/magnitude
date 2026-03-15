import { Effect, Scope } from 'effect'

const DEFAULT_REFRESH_INTERVAL_MS = 15 * 60 * 1000

export function makeRefreshSchedule(
  refreshFn: Effect.Effect<void>,
  intervalMs = DEFAULT_REFRESH_INTERVAL_MS,
): Effect.Effect<void, never, Scope.Scope> {
  return Effect.acquireRelease(
    Effect.sync(() => {
      const id = setInterval(() => {
        Effect.runFork(refreshFn)
      }, intervalMs)
      id.unref?.()
      return id
    }),
    (id) => Effect.sync(() => clearInterval(id)),
  ).pipe(Effect.asVoid)
}