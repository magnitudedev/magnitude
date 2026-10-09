import { Clock, Duration, Effect, Queue, Random, Ref } from "effect"
import type { UpdateCheckReason } from "@magnitudedev/release/hosted-update"

/**
 * One application-owned timer. Resume only wakes it to re-evaluate the deadline. The first check
 * runs 3 seconds after `start` completes, so an owner can wait until its service is ready before
 * reporting the previous update's outcome.
 */
export const makeUpdateSchedule = <E>(check: (reason: UpdateCheckReason) => Effect.Effect<void, E>, start: Effect.Effect<void> = Effect.void) => Effect.gen(function* () {
  const wakeups = yield* Queue.sliding<void>(1)
  yield* Effect.addFinalizer(() => Queue.shutdown(wakeups))
  const deadline = yield* Ref.make((yield* Clock.currentTimeMillis) + 3_000)
  const launched = yield* Ref.make(false)
  const gate = yield* Effect.makeSemaphore(1)
  const runCheck = (reason: UpdateCheckReason) => Effect.gen(function* () {
    const now = yield* Clock.currentTimeMillis
    const jitter = yield* Random.nextIntBetween(0, 60_001)
    yield* Ref.set(deadline, now + 3_600_000 + jitter)
    yield* Queue.offer(wakeups, undefined)
    yield* check(reason)
  })
  const attempt = gate.withPermits(1)(runCheck("manual"))
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
      const reason = (yield* Ref.getAndSet(launched, true)) ? "scheduled" : "launch"
      yield* runCheck(reason).pipe(Effect.catchAll(() => Effect.logDebug("Scheduled update check failed")))
    }))
  })
  yield* start.pipe(
    Effect.zipRight(Clock.currentTimeMillis.pipe(Effect.flatMap(now => Ref.update(deadline, current => Math.max(current, now + 3_000))))),
    Effect.zipRight(tick.pipe(Effect.forever)), Effect.forkScoped)
  return { check: attempt, resume: Queue.offer(wakeups, undefined).pipe(Effect.asVoid) }
})
