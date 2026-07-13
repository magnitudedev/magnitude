import { Effect, Option } from "effect"
import type { JitDaemonEndpoint, JitDaemonProvider } from "./daemon-provider"

export interface JitDaemonResolver<E> {
  readonly resolve: Effect.Effect<JitDaemonEndpoint, E, never>
  readonly invalidate: (endpoint: JitDaemonEndpoint) => Effect.Effect<void>
}

export const makeJitDaemonResolver = <E>(
  provider: JitDaemonProvider<E>,
): JitDaemonResolver<E> => {
  let current: JitDaemonEndpoint | null = null
  const semaphore = Effect.unsafeMakeSemaphore(1)

  const resolve: Effect.Effect<JitDaemonEndpoint, E, never> = semaphore.withPermits(1)(
    Effect.suspend(() => {
      if (current !== null) return Effect.succeed(current)

      return provider.discover().pipe(
        Effect.tap((found) =>
          Effect.logDebug("jit-rpc resolve discover").pipe(
            Effect.annotateLogs({ discovered: Option.isSome(found) ? found.value.url : "none" }),
          ),
        ),
        Effect.flatMap(Option.match({
          onNone: () =>
            provider.spawn().pipe(
              Effect.tap((endpoint) =>
                Effect.logDebug("jit-rpc resolve spawned").pipe(
                  Effect.annotateLogs({ url: endpoint.url }),
                ),
              ),
            ),
          onSome: Effect.succeed,
        })),
        Effect.tap((endpoint) => Effect.sync(() => { current = endpoint })),
      )
    }),
  )

  const invalidate = (endpoint: JitDaemonEndpoint): Effect.Effect<void> =>
    Effect.gen(function* () {
      yield* Effect.logDebug("jit-rpc invalidate endpoint").pipe(
        Effect.annotateLogs({ url: endpoint.url, current: current?.url ?? null }),
      )
      if (current?.url === endpoint.url) current = null
    })

  return { resolve, invalidate }
}
