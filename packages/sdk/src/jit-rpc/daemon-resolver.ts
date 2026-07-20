import { Effect, Option, Ref } from "effect"
import type { JitDaemonEndpoint, JitDaemonProvider } from "./daemon-provider"

export interface JitDaemonCoordinator<E> {
  readonly ensure: Effect.Effect<JitDaemonLease, E, never>
  readonly invalidate: (lease: JitDaemonLease) => Effect.Effect<void>
  readonly current: Effect.Effect<Option.Option<JitDaemonEndpoint>>
}

export interface JitDaemonLease {
  readonly endpoint: JitDaemonEndpoint
  readonly generation: number
}

/**
 * The coordinator is the process-local authority for one JIT daemon.
 * It is intentionally stateful: every protocol consumer in an application
 * process must share the same instance.
 */
export const makeJitDaemonCoordinator = <E>(
  provider: JitDaemonProvider<E>,
): Effect.Effect<JitDaemonCoordinator<E>> => Effect.gen(function* () {
  const current = yield* Ref.make<JitDaemonLease | null>(null)
  const generation = yield* Ref.make(0)
  const semaphore = yield* Effect.makeSemaphore(1)

  const ensure: Effect.Effect<JitDaemonLease, E, never> = semaphore.withPermits(1)(
    Effect.gen(function* () {
      const cached = yield* Ref.get(current)
      if (cached !== null) return cached

      const endpoint = yield* provider.discover().pipe(
        Effect.tap((found) =>
          Effect.logDebug("jit-rpc ensure discover").pipe(
            Effect.annotateLogs({ discovered: Option.isSome(found) ? found.value.url : "none" }),
          ),
        ),
        Effect.flatMap(Option.match({
          onNone: () =>
            provider.spawn().pipe(
              Effect.tap((endpoint) =>
                Effect.logDebug("jit-rpc ensure spawned").pipe(
                  Effect.annotateLogs({ url: endpoint.url }),
                ),
              ),
            ),
          onSome: Effect.succeed,
        })),
      )
      const nextGeneration = yield* Ref.updateAndGet(generation, (value) => value + 1)
      const lease = { endpoint, generation: nextGeneration }
      yield* Ref.set(current, lease)
      return lease
    }),
  )

  const invalidate = (lease: JitDaemonLease): Effect.Effect<void> =>
    semaphore.withPermits(1)(Effect.gen(function* () {
      const cached = yield* Ref.get(current)
      yield* Effect.logDebug("jit-rpc invalidate endpoint").pipe(
        Effect.annotateLogs({
          url: lease.endpoint.url,
          generation: lease.generation,
          currentUrl: cached?.endpoint.url ?? null,
          currentGeneration: cached?.generation ?? null,
        }),
      )
      if (cached?.generation === lease.generation) yield* Ref.set(current, null)
    }))

  return {
    ensure,
    invalidate,
    current: Ref.get(current).pipe(
      Effect.map(Option.fromNullable),
      Effect.map(Option.map((lease) => lease.endpoint)),
    ),
  }
})
