import { Effect, Option, Ref } from "effect"
import type { JitDaemonEndpoint, JitDaemonProvider } from "./daemon-provider"

export interface JitDaemonCoordinator<E> {
  readonly ensure: Effect.Effect<JitDaemonLease, E, never>
  /** Waits for a discovered endpoint after an authoritative termination. Never spawns. */
  readonly awaitSuccessor: (lease: JitDaemonLease) => Effect.Effect<JitDaemonLease, E, never>
  readonly invalidate: (
    lease: JitDaemonLease,
    options?: { readonly awaitDifferentEndpoint?: boolean },
  ) => Effect.Effect<void>
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
): Effect.Effect<JitDaemonCoordinator<E>> =>
  Effect.gen(function* () {
    const current = yield* Ref.make<JitDaemonLease | null>(null)
    const excludedUrl = yield* Ref.make<string | null>(null)
    const generation = yield* Ref.make(0)
    const semaphore = yield* Effect.makeSemaphore(1)

    const cacheEndpoint = (endpoint: JitDaemonEndpoint) =>
      Effect.gen(function* () {
        const excluded = yield* Ref.get(excludedUrl)
        if (excluded !== endpoint.url) yield* Ref.set(excludedUrl, null)
        const nextGeneration = yield* Ref.updateAndGet(generation, (value) => value + 1)
        const lease = { endpoint, generation: nextGeneration }
        yield* Ref.set(current, lease)
        return lease
      })

    const observeNow = semaphore.withPermits(1)(
      Effect.gen(function* () {
        const cached = yield* Ref.get(current)
        if (cached !== null) return Option.some(cached)
        const discovered = yield* provider.discover()
        if (Option.isNone(discovered)) return Option.none<JitDaemonLease>()
        if (discovered.value.url === (yield* Ref.get(excludedUrl))) {
          return Option.none<JitDaemonLease>()
        }
        return Option.some(yield* cacheEndpoint(discovered.value))
      }),
    )

    const ensure: Effect.Effect<JitDaemonLease, E, never> = Effect.suspend(() =>
      semaphore.withPermits(1)(Effect.gen(function* () {
        const cached = yield* Ref.get(current)
        if (cached !== null) return Option.some(cached)

        const discovered = yield* provider.discover().pipe(
          Effect.tap((found) =>
            Effect.logDebug("jit-rpc ensure discover").pipe(
              Effect.annotateLogs({
                discovered: Option.isSome(found) ? found.value.url : "none",
              }),
            ),
          ),
        )
        if (Option.isSome(discovered)) {
          if (discovered.value.url === (yield* Ref.get(excludedUrl))) {
            return Option.none<JitDaemonLease>()
          }
          return Option.some(yield* cacheEndpoint(discovered.value))
        }
        const endpoint = yield* provider.spawn().pipe(
          Effect.tap((spawned) =>
            Effect.logDebug("jit-rpc ensure spawned").pipe(
              Effect.annotateLogs({ url: spawned.url }),
            ),
          ),
        )
        if (endpoint.url === (yield* Ref.get(excludedUrl))) {
          return Option.none<JitDaemonLease>()
        }
        return Option.some(yield* cacheEndpoint(endpoint))
      })).pipe(
        Effect.flatMap(
          Option.match({
            onSome: Effect.succeed,
            onNone: () => Effect.sleep("50 millis").pipe(Effect.zipRight(ensure)),
          }),
        ),
      ),
    )

    const invalidate = (
      lease: JitDaemonLease,
      options?: { readonly awaitDifferentEndpoint?: boolean },
    ): Effect.Effect<void> =>
      semaphore.withPermits(1)(
        Effect.gen(function* () {
          const cached = yield* Ref.get(current)
          yield* Effect.logDebug("jit-rpc invalidate endpoint").pipe(
            Effect.annotateLogs({
              url: lease.endpoint.url,
              generation: lease.generation,
              currentUrl: cached?.endpoint.url ?? null,
              currentGeneration: cached?.generation ?? null,
            }),
          )
          if (cached?.generation === lease.generation) {
            yield* Ref.set(current, null)
            if (options?.awaitDifferentEndpoint === true) {
              yield* Ref.set(excludedUrl, lease.endpoint.url)
            }
          }
        }),
      )

    const awaitDiscoveredSuccessor: Effect.Effect<JitDaemonLease, E, never> = Effect.suspend(() =>
      observeNow.pipe(
        Effect.flatMap(
          Option.match({
            onSome: Effect.succeed,
            onNone: () =>
              Effect.sleep("250 millis").pipe(Effect.zipRight(awaitDiscoveredSuccessor)),
          }),
        ),
      ),
    )

    const awaitSuccessor = (lease: JitDaemonLease) =>
      invalidate(lease, { awaitDifferentEndpoint: true }).pipe(
        Effect.zipRight(awaitDiscoveredSuccessor),
      )

    return {
      ensure,
      awaitSuccessor,
      invalidate,
      current: Ref.get(current).pipe(
        Effect.map(Option.fromNullable),
        Effect.map(Option.map((lease) => lease.endpoint)),
      ),
    }
  })
