import { Context, Deferred, Effect, Layer, SynchronizedRef } from 'effect'

export type ForkKey = string | null

export interface InterruptBaseline {
  readonly executionEpoch: number
  readonly interruptEpoch: number
}

interface InterruptEntry extends InterruptBaseline {
  readonly waiters: Set<Deferred.Deferred<void, never>>
}

export interface InterruptCoordinator {
  readonly beginExecution: (forkId: ForkKey) => Effect.Effect<InterruptBaseline>
  readonly current: (forkId: ForkKey) => Effect.Effect<InterruptBaseline>
  readonly interrupt: (forkId: ForkKey) => Effect.Effect<void>
  readonly waitForInterrupt: (
    forkId: ForkKey,
    baseline: InterruptBaseline
  ) => Effect.Effect<never>
}

const emptyEntry = (): InterruptEntry => ({
  executionEpoch: 0,
  interruptEpoch: 0,
  waiters: new Set(),
})

const toBaseline = (entry: InterruptEntry): InterruptBaseline => ({
  executionEpoch: entry.executionEpoch,
  interruptEpoch: entry.interruptEpoch,
})

const wakeWaiters = (waiters: ReadonlySet<Deferred.Deferred<void, never>>) =>
  Effect.forEach(waiters, (waiter) => Deferred.succeed(waiter, undefined), {
    discard: true,
  })

export const InterruptCoordinator = Context.GenericTag<InterruptCoordinator>('InterruptCoordinator')

export const InterruptCoordinatorLive = Layer.effect(
  InterruptCoordinator,
  Effect.gen(function* () {
    const state = yield* SynchronizedRef.make<Map<ForkKey, InterruptEntry>>(new Map())

    return {
      beginExecution: (forkId) =>
        SynchronizedRef.modifyEffect(state, (map) => {
          const current = map.get(forkId) ?? emptyEntry()
          const next: InterruptEntry = {
            executionEpoch: current.executionEpoch + 1,
            interruptEpoch: 0,
            waiters: new Set(),
          }
          const nextMap = new Map(map)
          nextMap.set(forkId, next)
          return Effect.as(
            wakeWaiters(current.waiters),
            [toBaseline(next), nextMap] as const,
          )
        }),

      current: (forkId) =>
        SynchronizedRef.modifyEffect(state, (map) =>
          Effect.succeed([toBaseline(map.get(forkId) ?? emptyEntry()), map] as const)
        ),

      interrupt: (forkId) =>
        SynchronizedRef.modifyEffect(state, (map) => {
          const current = map.get(forkId) ?? emptyEntry()
          const next: InterruptEntry = {
            executionEpoch: current.executionEpoch,
            interruptEpoch: current.interruptEpoch + 1,
            waiters: new Set(),
          }
          const nextMap = new Map(map)
          nextMap.set(forkId, next)
          return Effect.as(wakeWaiters(current.waiters), [undefined, nextMap] as const)
        }),

      waitForInterrupt: (forkId, baseline) =>
        Effect.forever(
          Effect.flatMap(
            Deferred.make<void, never>(),
            (waiter) =>
              SynchronizedRef.modifyEffect(state, (map) => {
                const current = map.get(forkId) ?? emptyEntry()
                if (
                  current.executionEpoch === baseline.executionEpoch &&
                  current.interruptEpoch > baseline.interruptEpoch
                ) {
                  return Effect.interrupt.pipe(
                    Effect.map(() => [undefined as never, map] as const)
                  )
                }

                const nextEntry: InterruptEntry = {
                  ...current,
                  waiters: new Set(current.waiters).add(waiter),
                }
                const nextMap = new Map(map)
                nextMap.set(forkId, nextEntry)

                return Effect.succeed([waiter, nextMap] as const)
              }).pipe(
                Effect.flatMap((registeredWaiter) => Deferred.await(registeredWaiter))
              )
          )
        ) as Effect.Effect<never>,
    } satisfies InterruptCoordinator
  })
)
