/**
 * The ACN change subscription, drained into the connection's operation cache.
 *
 * Every poke names a operation; that operation is invalidated (and refetched if
 * observed). Every (re)connection of the subscription rereads everything the
 * cache holds, since pokes may have been missed while disconnected. Nothing
 * else in the client interprets pokes.
 */
import { Atom, Registry } from "@effect-atom/atom-react"
import { Duration, Effect, Layer, Option, Runtime, Stream } from "effect"
import { Key, QueryClient, Subscription } from "@magnitudedev/effect-query"
import { ClientEffectQuery } from "./client-effect-query"
import { MagnitudeClient } from "@magnitudedev/sdk"

/** The wait before reopening a change stream that ended: 250 ms, doubling with each ending in a row, at most 5 seconds. */
export const reopenDelay = (endingsInARow: number): Duration.Duration =>
  Duration.millis(Math.min(5_000, 250 * 2 ** endingsInARow))

/**
 * Reopens a change subscription that failed (a defect, which the subscription never retries, or its
 * retries exhausted) or that the service ended, after `reopenDelay`, then runs `reopened` so the cache
 * rereads what it may have missed. An event resets the backoff.
 */
export const reopenEndedSubscription = <S extends { readonly status: string }>(
  registry: Registry.Registry,
  atom: Atom.Writable<S, Atom.Reset>,
  reopened: Effect.Effect<void>,
): Effect.Effect<void> => Registry.toStream(registry, atom).pipe(
  Stream.mapAccum(0, (endings, state): readonly [number, Option.Option<number>] =>
    state.status === "active" ? [0, Option.none()]
      : state.status === "failed" || state.status === "completed" ? [endings + 1, Option.some(endings)]
      : [endings, Option.none()]),
  Stream.filterMap((ending) => ending),
  Stream.runForEach((endings) => Effect.sleep(reopenDelay(endings)).pipe(
    Effect.zipRight(Effect.sync(() => registry.set(atom, Atom.Reset))),
    Effect.zipRight(reopened),
  )),
)

export const ChangesLive = Layer.scopedDiscard(Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const runtime = yield* Effect.runtime()
  const sdk = yield* MagnitudeClient
  const changes = effectQuery.Changes.StreamChanges({})
  const runSync = Runtime.runSync(runtime)

  // SDK transport recovery can reopen a subscription without a new Query attempt.
  // Each later Ready admission therefore invalidates the same existing cache.
  yield* sdk.connection.changes.pipe(
    Stream.filter(state => state._tag === "Ready"),
    Stream.drop(1),
    Stream.runForEach(() => queryClient.invalidate()),
    Effect.forkScoped,
  )

  // A reconnection (any attempt after the first) may have missed pokes: reread everything.
  // The first attempt opens before any operation is read, so nothing can be stale yet.
  yield* Effect.acquireRelease(
    Effect.sync(() => {
      let attempt = 0
      return registry.subscribe(changes, (state) => {
        if (state.attempt === attempt) return
        attempt = state.attempt
        if (attempt > 1) queueMicrotask(() => runSync(queryClient.invalidate()))
      })
    }),
    (unsubscribe) => Effect.sync(unsubscribe),
  )

  // The subscription retries a failure itself, but not a defect or the stream ending; never stay deaf to pokes.
  yield* reopenEndedSubscription(registry, changes, queryClient.invalidate()).pipe(Effect.forkScoped)

  yield* Subscription.events(changes).pipe(
    Stream.runForEach((change) => queryClient.invalidate({
      name: change.operation,
      key: change.key === undefined ? undefined : Key.canonical(change.key),
      exact: true,
    })),
    Effect.forkScoped,
  )

}))
