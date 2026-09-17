import { Clock, Duration, Effect, Queue, Random, Ref } from "effect"

/** One application-owned timer. Resume only wakes it to re-evaluate the deadline. */
export const makeUpdateSchedule = <E>(check: Effect.Effect<void, E>) => Effect.gen(function* () {
  const wakeups = yield* Queue.sliding<void>(1)
  yield* Effect.addFinalizer(() => Queue.shutdown(wakeups))
  const deadline = yield* Ref.make((yield* Clock.currentTimeMillis) + 3_000)
  const gate = yield* Effect.makeSemaphore(1)
  const runCheck = Effect.gen(function* () {
    const now = yield* Clock.currentTimeMillis
    const jitter = yield* Random.nextIntBetween(0, 60_001)
    yield* Ref.set(deadline, now + 3_600_000 + jitter)
    yield* Queue.offer(wakeups, undefined)
    yield* check
  })
  const attempt = gate.withPermits(1)(runCheck)
  const tick = Effect.gen(function* () {
    const remaining = (yield* Ref.get(deadline)) - (yield* Clock.currentTimeMillis)
    if (remaining > 0) {
      yield* Effect.race(Effect.sleep(Duration.millis(remaining)), Queue.take(wakeups))
      return
    }
    // Re-check under the gate: a manual check may have moved the deadline.
    yield* gate.withPermits(1)(Effect.gen(function* () {
      const now = yield* Clock.currentTimeMillis
      if ((yield* Ref.get(deadline)) > now) return
      yield* runCheck.pipe(Effect.catchAll(() => Effect.logDebug("Scheduled update check failed")))
    }))
  })
  yield* tick.pipe(Effect.forever, Effect.forkScoped)
  return { check: attempt, resume: Queue.offer(wakeups, undefined).pipe(Effect.asVoid) }
})
