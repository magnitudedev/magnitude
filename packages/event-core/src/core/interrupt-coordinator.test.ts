import { describe, test, expect } from 'bun:test'
import { Effect, Fiber, Option } from 'effect'
import { InterruptCoordinator, InterruptCoordinatorLive } from './interrupt-coordinator'

describe('InterruptCoordinator', () => {
  test('same execution interrupt wakes waiter', async () => {
    const completedAfterInterrupt = await Effect.runPromise(
      Effect.gen(function* () {
        const coordinator = yield* InterruptCoordinator
        const baseline = yield* coordinator.beginExecution('fork-a')
        const waiterFiber = yield* Effect.fork(
          coordinator.waitForInterrupt('fork-a', baseline).pipe(Effect.exit),
        )

        const beforeInterrupt = yield* Fiber.poll(waiterFiber)
        yield* coordinator.interrupt('fork-a')
        yield* Fiber.await(waiterFiber)

        return Option.isNone(beforeInterrupt)
      }).pipe(Effect.provide(InterruptCoordinatorLive)),
    )

    expect(completedAfterInterrupt).toBe(true)
  })

  test('new execution wakes prior waiters without treating it as interrupt', async () => {
    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const coordinator = yield* InterruptCoordinator
        const baseline = yield* coordinator.beginExecution('fork-a')
        const waiterFiber = yield* Effect.fork(
          coordinator.waitForInterrupt('fork-a', baseline).pipe(Effect.exit),
        )

        yield* coordinator.beginExecution('fork-a')
        yield* Effect.yieldNow()

        const pollAfterRollover = yield* waiterFiber.poll
        yield* Fiber.interrupt(waiterFiber)
        return pollAfterRollover
      }).pipe(Effect.provide(InterruptCoordinatorLive)),
    )

    expect(Option.isNone(result)).toBe(true)
  })
})
