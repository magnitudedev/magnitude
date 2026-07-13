import { Chunk, Duration, Effect, Layer, TestClock, TestContext } from 'effect'

export interface FakeClock {
  readonly layer: Layer.Layer<TestClock.TestClock, never, never>
  now(): number
  advanceBy(ms: number): Promise<void>
  runAll(): Promise<void>
}

export function createFakeClock(startMs = Date.now()): FakeClock {
  let runVoid: ((effect: Effect.Effect<void>) => Promise<void>) | null = null
  let runSleeps: ((effect: Effect.Effect<Chunk.Chunk<number>>) => Promise<Chunk.Chunk<number>>) | null = null
  let nowMs = startMs

  const provided = Layer.provideMerge(
    TestClock.live(TestClock.makeData(startMs, Chunk.empty())),
    TestContext.TestContext,
  )

  const layer: Layer.Layer<TestClock.TestClock, never, never> = Layer.effect(
    TestClock.TestClock,
    Effect.gen(function* () {
      runVoid = (effect) => Effect.runPromise(Effect.provide(effect, provided))
      runSleeps = (effect) => Effect.runPromise(Effect.provide(effect, provided))
      return yield* Effect.provide(TestClock.testClock(), provided)
    }),
  )

  const requireRunVoid = (): ((effect: Effect.Effect<void>) => Promise<void>) => {
    if (!runVoid) {
      throw new Error('Fake clock is not initialized yet. Ensure harness client is created before using clock controls.')
    }
    return runVoid
  }

  const requireRunSleeps = (): ((effect: Effect.Effect<Chunk.Chunk<number>>) => Promise<Chunk.Chunk<number>>) => {
    if (!runSleeps) {
      throw new Error('Fake clock is not initialized yet. Ensure harness client is created before using clock controls.')
    }
    return runSleeps
  }

  return {
    layer,
    now: () => nowMs,
    advanceBy: async (ms: number) => {
      if (ms <= 0) return
      const run = requireRunVoid()
      await run(TestClock.adjust(Duration.millis(ms)))
      nowMs += ms
    },
    runAll: async () => {
      const run = requireRunVoid()
      const getSleeps = requireRunSleeps()

      while (true) {
        const sleeps = await getSleeps(TestClock.sleeps())
        const pending = Chunk.toReadonlyArray(sleeps).slice().sort((a, b) => a - b)
        if (pending.length === 0) {
          return
        }

        const target = pending[0]
        const delta = Math.max(0, target - nowMs)
        await run(TestClock.adjust(Duration.millis(delta)))
        nowMs += delta
      }
    },
  }
}