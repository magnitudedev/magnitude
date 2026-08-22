/**
 * The ACN change subscription, drained into the connection's query cache.
 *
 * Every poke names a query; that query is invalidated (and refetched if
 * observed). Every (re)connection of the subscription rereads everything the
 * cache holds, since pokes may have been missed while disconnected. Nothing
 * else in the client interprets pokes.
 */
import { Registry } from "@effect-atom/atom-react"
import { Effect, Layer, Runtime, Stream } from "effect"
import { Key, QueryClient, Subscription } from "@magnitudedev/effect-query"
import { StreamChanges } from "@magnitudedev/sdk"
import { ClientEffectQuery } from "./client-effect-query"

export const ChangesLive = Layer.scopedDiscard(Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const runtime = yield* Effect.runtime()
  const changes = effectQuery.subscription(StreamChanges, {})
  const runSync = Runtime.runSync(runtime)

  // A reconnection (any attempt after the first) may have missed pokes: reread everything.
  // The first attempt opens before any query is read, so nothing can be stale yet.
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
      name: change.query,
      key: change.key === undefined ? undefined : Key.canonical(change.key),
      exact: true,
    })),
    Effect.forkScoped,
  )
}))
