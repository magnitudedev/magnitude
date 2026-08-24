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
import {
  invalidateAllInferenceQueries,
  invalidateInferenceTopic,
} from "@magnitudedev/sdk"
import { ClientEffectQuery } from "./client-effect-query"

export const ChangesLive = Layer.scopedDiscard(Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const runtime = yield* Effect.runtime()
  const changes = effectQuery.Changes.StreamChanges({})
  const inferenceEvents = effectQuery.Inference.WatchInferenceEvents({})
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

  yield* Effect.acquireRelease(
    Effect.sync(() => {
      let attempt = 0
      return registry.subscribe(inferenceEvents, (state) => {
        if (state.attempt === attempt) return
        attempt = state.attempt
        if (attempt > 1) {
          queueMicrotask(() => runSync(invalidateAllInferenceQueries().pipe(
            Effect.provideService(QueryClient.QueryClient, queryClient),
          )))
        }
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

  yield* Subscription.events(inferenceEvents).pipe(
    Stream.runForEach((event) => invalidateInferenceTopic(event.topic)),
    Effect.forkScoped,
  )

  // Memory consumed by unrelated OS processes has no ICN mutation event.
  // Invalidate only mounted hardware-dependent Queries on the same cadence as
  // the backend projection's hardware observer.
  yield* invalidateInferenceTopic("hardware").pipe(
    Effect.provideService(QueryClient.QueryClient, queryClient),
    Effect.delay("2 seconds"),
    Effect.forever,
    Effect.forkScoped,
  )
}))
