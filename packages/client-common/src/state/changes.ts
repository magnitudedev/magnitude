/**
 * The ACN change subscription, drained into the connection's operation cache.
 *
 * Every poke names a operation; that operation is invalidated (and refetched if
 * observed). Every (re)connection of the subscription rereads everything the
 * cache holds, since pokes may have been missed while disconnected. Nothing
 * else in the client interprets pokes.
 */
import { Registry } from "@effect-atom/atom-react"
import { Effect, Layer, Runtime, Stream } from "effect"
import { Key, QueryClient, Subscription } from "@magnitudedev/effect-query"
import { ClientEffectQuery } from "./client-effect-query"
import { MagnitudeClient } from "@magnitudedev/sdk"

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

  yield* Subscription.events(changes).pipe(
    Stream.runForEach((change) => queryClient.invalidate({
      name: change.operation,
      key: change.key === undefined ? undefined : Key.canonical(change.key),
      exact: true,
    })),
    Effect.forkScoped,
  )

}))
