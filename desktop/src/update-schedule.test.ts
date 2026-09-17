import { Effect, Ref, TestClock, TestContext } from "effect"
import { expect, it } from "vitest"
import { makeUpdateSchedule } from "./update-schedule"

it("checks after startup, keeps an hourly cadence after failure, and does not check for every resume", async () => {
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const calls = yield* Ref.make(0)
    const scheduler = yield* makeUpdateSchedule(Ref.update(calls, n => n + 1).pipe(Effect.zipRight(Effect.fail("offline"))))
    yield* TestClock.adjust("2 seconds")
    expect(yield* Ref.get(calls)).toBe(0)
    yield* TestClock.adjust("1 second")
    expect(yield* Ref.get(calls)).toBe(1)
    yield* Effect.all(Array.from({ length: 20 }, () => scheduler.resume), { concurrency: "unbounded" })
    yield* TestClock.adjust("59 minutes")
    expect(yield* Ref.get(calls)).toBe(1)
    yield* TestClock.adjust("2 minutes")
    expect(yield* Ref.get(calls)).toBe(2)
    yield* TestClock.adjust("10 seconds")
    expect(yield* Ref.get(calls)).toBe(2)
  })).pipe(Effect.provide(TestContext.TestContext)))
})

it("manual checks reset the automatic deadline", async () => {
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const calls = yield* Ref.make(0)
    const scheduler = yield* makeUpdateSchedule(Ref.update(calls, n => n + 1))
    yield* TestClock.adjust("2 seconds")
    yield* scheduler.check
    yield* TestClock.adjust("5 seconds")
    expect(yield* Ref.get(calls)).toBe(1)
    yield* TestClock.adjust("61 minutes")
    expect(yield* Ref.get(calls)).toBe(2)
  })).pipe(Effect.provide(TestContext.TestContext)))
})
