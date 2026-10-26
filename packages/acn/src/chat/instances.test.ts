import { Deferred, Effect, Either, Fiber, Option, Ref, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { type Instance, type Instances, makeInstances } from "./instances"

interface Fake extends Instance {
  readonly id: number
  readonly fail: Effect.Effect<void>
  /** What a removal that closed it did. */
  readonly removed: Effect.Effect<void, string>
}

/** Instances over fake runtimes that record what happened to them. */
const fixture = Effect.gen(function* () {
  const log = yield* Ref.make<ReadonlyArray<string>>([])
  const record = (entry: string) => Ref.update(log, (entries) => [...entries, entry])
  const opens = yield* Ref.make(0)
  const failOpen = yield* Ref.make(false)
  const failRemove = yield* Ref.make(false)
  /** What an open that fails, having opened something, and a delete wait for first. */
  const opening = yield* Ref.make<Effect.Effect<void>>(Effect.void)
  const deleting = yield* Ref.make<Effect.Effect<void>>(Effect.void)
  /** What an instance waits for to be idle, and what its close waits for first. */
  const idling = yield* Ref.make<Effect.Effect<void>>(Effect.void)
  const closing = yield* Ref.make<Effect.Effect<void>>(Effect.void)
  const instances: Instances<string, Fake, string> = yield* makeInstances<string, Fake, string, never>({
    open: (key, removed) => Effect.gen(function* () {
      if (yield* Ref.get(failOpen)) {
        yield* Effect.addFinalizer(() => record(`close failed ${key}`))
        yield* Effect.flatten(Ref.get(opening))
        return yield* Effect.fail("unavailable")
      }
      const id = yield* Ref.updateAndGet(opens, (n) => n + 1)
      const failed = yield* Deferred.make<boolean>()
      yield* record(`open ${key} ${id}`)
      yield* Effect.addFinalizer(() => Effect.flatten(Ref.get(closing)).pipe(
        Effect.zipRight(record(`close ${key} ${id}`)),
        Effect.zipRight(Deferred.succeed(failed, false))
      ))
      return {
        id,
        awaitIdle: Effect.flatten(Ref.get(idling)),
        close: (reason) => record(`${reason} ${key} ${id}`),
        failed: Deferred.await(failed),
        fail: Deferred.succeed(failed, true).pipe(Effect.asVoid),
        removed,
      }
    }),
    remove: (key) => Effect.zipRight(Effect.flatten(Ref.get(deleting)), Effect.flatMap(Ref.get(failRemove), (fails) =>
      fails ? Effect.zipRight(record(`delete ${key} failed`), Effect.fail("unavailable")) : record(`delete ${key}`))),
    failed: (key) => Effect.flatMap(Effect.map(instances.current(key), Option.isSome), (open) => record(`failed ${key}${open ? " while open" : ""}`)),
    idleTimeToLive: "30 seconds",
  })
  return { instances, log: Ref.get(log), failOpen, failRemove, opening, deleting, idling, closing }
})

const run = <A, E>(effect: Effect.Effect<A, E, import("effect").Scope.Scope>) =>
  Effect.runPromise(Effect.scoped(effect).pipe(Effect.provide(TestContext.TestContext)))

/** Gates an instance's close; `started` completes once a close waits on the gate. */
const gatedClose = (closing: Ref.Ref<Effect.Effect<void>>) => Effect.gen(function* () {
  const started = yield* Deferred.make<void>()
  const gate = yield* Deferred.make<void>()
  yield* Ref.set(closing, Effect.zipRight(Deferred.succeed(started, undefined), Deferred.await(gate)))
  return { started: Deferred.await(started), open: Deferred.succeed(gate, undefined) }
})

/** Lets forked work run until the log has this entry: a holder's hold ends once its instance is idle, after it leaves. */
const logged = (log: Effect.Effect<ReadonlyArray<string>>, entry: string) => Effect.gen(function* () {
  for (let turn = 0; turn < 100 && !(yield* log).includes(entry); turn++) yield* Effect.yieldNow()
})

describe("Instances", () => {
  it("shares one instance between concurrent holders", () =>
    run(Effect.gen(function* () {
      const { instances, log } = yield* fixture
      const [left, right] = yield* Effect.all([
        Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id)),
        Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id)),
      ], { concurrency: "unbounded" })
      expect(left).toBe(right)
      expect(yield* log).toEqual(["open chat 1"])
    })))

  it("closes an idle instance after its time to live, unless it is used again first", () =>
    run(Effect.gen(function* () {
      const { instances, log } = yield* fixture
      yield* Effect.scoped(instances.get("chat"))
      yield* TestClock.adjust("20 seconds")
      expect(yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id))).toBe(1)
      yield* TestClock.adjust("29 seconds")
      expect(yield* log).toEqual(["open chat 1"])
      yield* TestClock.adjust("1 second")
      expect(yield* log).toEqual(["open chat 1", "close chat 1"])
      expect(yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id))).toBe(2)
    })))

  it("opens a key used during its idle close only once the previous instance has closed", () =>
    run(Effect.gen(function* () {
      const { instances, log, closing } = yield* fixture
      const close = yield* gatedClose(closing)
      yield* Effect.scoped(instances.get("chat"))
      yield* TestClock.adjust("30 seconds")
      yield* close.started
      const next = yield* Effect.fork(Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id)))
      yield* Effect.repeatN(Effect.yieldNow(), 100)
      expect(yield* Fiber.poll(next)).toEqual(Option.none())
      expect(yield* instances.current("chat")).toEqual(Option.none())
      expect(yield* log).toEqual(["open chat 1"])
      yield* close.open
      expect(yield* Fiber.join(next)).toBe(2)
      expect(yield* log).toEqual(["open chat 1", "close chat 1", "open chat 2"])
    })))

  it("deletes a key removed during its idle close only once the previous instance has closed", () =>
    run(Effect.gen(function* () {
      const { instances, log, closing } = yield* fixture
      const close = yield* gatedClose(closing)
      yield* Effect.scoped(instances.get("chat"))
      yield* TestClock.adjust("30 seconds")
      yield* close.started
      const removal = yield* Effect.fork(instances.remove("chat"))
      yield* Effect.repeatN(Effect.yieldNow(), 100)
      expect(yield* Fiber.poll(removal)).toEqual(Option.none())
      expect(yield* log).toEqual(["open chat 1"])
      yield* close.open
      yield* Fiber.join(removal)
      expect(yield* log).toEqual(["open chat 1", "close chat 1", "delete chat"])
    })))

  it("removes an open instance as removed, then deletes it, without reopening it", () =>
    run(Effect.gen(function* () {
      const { instances, log } = yield* fixture
      yield* Effect.scoped(instances.get("chat"))
      yield* instances.remove("chat")
      yield* logged(log, "close chat 1")
      expect(yield* log).toEqual(["open chat 1", "removed chat 1", "delete chat", "close chat 1"])
    })))

  it("forgets a removed instance whose delete failed, so the next use reopens it", () =>
    run(Effect.gen(function* () {
      const { instances, log, failRemove } = yield* fixture
      yield* Effect.scoped(instances.get("chat"))
      yield* Ref.set(failRemove, true)
      expect(yield* Effect.flip(instances.remove("chat"))).toBe("unavailable")
      yield* logged(log, "close chat 1")
      expect(yield* log).toEqual(["open chat 1", "removed chat 1", "delete chat failed", "close chat 1"])
      expect(yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id))).toBe(2)
      // Only the reopened instance idles: its time to live passing closes it, and nothing else.
      yield* TestClock.adjust("30 seconds")
      expect(yield* log).toEqual(["open chat 1", "removed chat 1", "delete chat failed", "close chat 1", "open chat 2", "close chat 2"])
    })))

  it("keeps an instance its holder left while it was busy open until it is idle, then for its time to live", () =>
    run(Effect.gen(function* () {
      const { instances, log, idling } = yield* fixture
      const idle = yield* Deferred.make<void>()
      yield* Ref.set(idling, Deferred.await(idle))
      yield* Effect.scoped(instances.get("chat"))
      yield* TestClock.adjust("1 minute")
      expect(yield* log).toEqual(["open chat 1"])
      yield* Deferred.succeed(idle, undefined)
      yield* TestClock.adjust("29 seconds")
      expect(yield* log).toEqual(["open chat 1"])
      yield* TestClock.adjust("1 second")
      expect(yield* log).toEqual(["open chat 1", "close chat 1"])
    })))

  it("tells an instance's removal only once its delete has ended, as it ended", () =>
    run(Effect.gen(function* () {
      const { instances, log, deleting, failRemove } = yield* fixture
      for (const fails of [false, true]) {
        yield* Ref.set(failRemove, fails)
        const gate = yield* Deferred.make<void>()
        yield* Ref.set(deleting, Deferred.await(gate))
        const removed = yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.removed))
        const told = yield* Effect.fork(Effect.either(removed))
        const removal = yield* Effect.fork(Effect.either(instances.remove("chat")))
        yield* logged(log, `removed chat ${fails ? 2 : 1}`)
        // Closed as removed, with its delete still to come: nothing is told yet.
        expect(yield* Fiber.poll(told)).toEqual(Option.none())
        yield* Deferred.succeed(gate, undefined)
        const outcome = fails ? Either.left("unavailable") : Either.right(undefined)
        expect(yield* Fiber.join(told)).toEqual(outcome)
        expect(yield* Fiber.join(removal)).toEqual(outcome)
      }
      // The instance whose delete failed is forgotten: the next use opens the key afresh.
      expect(yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id))).toBe(3)
    })))

  it("completes a removal whose caller is interrupted, closing and deleting as one step", () =>
    run(Effect.gen(function* () {
      const { instances, log, deleting } = yield* fixture
      const gate = yield* Deferred.make<void>()
      yield* Ref.set(deleting, Deferred.await(gate))
      const removed = yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.removed))
      const removal = yield* Effect.fork(instances.remove("chat"))
      yield* logged(log, "removed chat 1")
      const interrupting = yield* Effect.fork(Fiber.interrupt(removal))
      yield* Effect.yieldNow()
      // The caller's interruption waits for the delete, which the removal still makes.
      expect(yield* Fiber.poll(interrupting)).toEqual(Option.none())
      yield* Deferred.succeed(gate, undefined)
      yield* Fiber.join(interrupting)
      yield* removed
      yield* logged(log, "close chat 1")
      expect(yield* log).toEqual(["open chat 1", "removed chat 1", "delete chat", "close chat 1"])
      expect(yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id))).toBe(2)
    })))

  it("deletes a key that is not open without opening it", () =>
    run(Effect.gen(function* () {
      const { instances, log } = yield* fixture
      yield* instances.remove("chat")
      expect(yield* log).toEqual(["delete chat"])
    })))

  it("makes a use during removal wait for it, then open afresh", () =>
    run(Effect.gen(function* () {
      const { instances, log } = yield* fixture
      const held = yield* Deferred.make<void>()
      const holder = yield* Effect.fork(Effect.scoped(Effect.zipRight(instances.get("chat"), Deferred.await(held))))
      yield* Effect.yieldNow()
      const removal = yield* Effect.fork(instances.remove("chat"))
      yield* Effect.yieldNow()
      const next = yield* Effect.fork(Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id)))
      yield* Fiber.join(removal)
      yield* Deferred.succeed(held, undefined)
      yield* Fiber.join(holder)
      expect(yield* Fiber.join(next)).toBe(2)
      expect((yield* log).slice(0, 3)).toEqual(["open chat 1", "removed chat 1", "delete chat"])
    })))

  it("forgets an instance that failed, and retries an open that failed", () =>
    run(Effect.gen(function* () {
      const { instances, failOpen } = yield* fixture
      yield* Effect.scoped(Effect.flatMap(instances.get("chat"), (r) => r.fail))
      yield* Effect.yieldNow()
      expect(yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id))).toBe(2)
      yield* Ref.set(failOpen, true)
      yield* instances.remove("chat")
      expect(yield* Effect.scoped(instances.get("chat")).pipe(Effect.flip)).toBe("unavailable")
      yield* Ref.set(failOpen, false)
      expect(yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id))).toBe(3)
    })))

  it("closes an instance that fails while no one holds it, without waiting for its time to live", () =>
    run(Effect.gen(function* () {
      const { instances, log } = yield* fixture
      const fail = yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.fail))
      yield* fail
      for (let turn = 0; turn < 100 && !(yield* log).includes("close chat 1"); turn++) yield* Effect.yieldNow()
      // It is forgotten before it is told failed, then closes.
      expect(yield* log).toEqual(["open chat 1", "failed chat", "close chat 1"])
      yield* TestClock.adjust("30 seconds")
      expect(yield* log).toEqual(["open chat 1", "failed chat", "close chat 1"])
      expect(yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id))).toBe(2)
    })))

  it("closes what an open that failed opened, whether its holder waited for it or left first", () =>
    run(Effect.gen(function* () {
      const { instances, log, failOpen, opening } = yield* fixture
      yield* Ref.set(failOpen, true)
      expect(yield* Effect.flip(Effect.scoped(instances.get("waited")))).toBe("unavailable")
      expect(yield* log).toEqual(["close failed waited"])

      const gate = yield* Deferred.make<void>()
      yield* Ref.set(opening, Deferred.await(gate))
      const holder = yield* Effect.fork(Effect.scoped(instances.get("left")))
      yield* Effect.yieldNow()
      // The holder leaves while it opens, so it idles; then the open fails.
      yield* Fiber.interrupt(holder)
      yield* Deferred.succeed(gate, undefined)
      for (let turn = 0; turn < 100 && !(yield* log).includes("close failed left"); turn++) yield* Effect.yieldNow()
      expect(yield* log).toEqual(["close failed waited", "close failed left"])
      expect(yield* instances.current("left")).toEqual(Option.none())
    })))

  it("gives the current instance once it is open, without opening or holding one", () =>
    run(Effect.gen(function* () {
      const { instances, log, opening, failOpen } = yield* fixture
      expect(yield* instances.current("chat")).toEqual(Option.none())
      const id = yield* Effect.scoped(Effect.map(instances.get("chat"), (r) => r.id))
      expect(Option.map(yield* instances.current("chat"), (r) => r.id)).toEqual(Option.some(id))
      // One still opening is current once it opens; this one fails to open, so it never is.
      const gate = yield* Deferred.make<void>()
      yield* Ref.set(failOpen, true)
      yield* Ref.set(opening, Deferred.await(gate))
      const holder = yield* Effect.fork(Effect.scoped(instances.get("other")))
      yield* Effect.yieldNow()
      const current = yield* Effect.fork(instances.current("other"))
      yield* Effect.yieldNow()
      expect(yield* Fiber.poll(current)).toEqual(Option.none())
      yield* Deferred.succeed(gate, undefined)
      yield* Effect.flip(Fiber.join(holder))
      expect(yield* Fiber.join(current)).toEqual(Option.none())
      expect(yield* log).toEqual(["open chat 1", "close failed other"])
      // One that has opened is current.
      yield* Ref.set(failOpen, false)
      const opened = yield* Effect.fork(Effect.scoped(Effect.zipRight(instances.get("third"), Effect.never)))
      yield* logged(log, "open third 2")
      const third = yield* instances.current("third")
      expect(Option.map(third, (r) => r.id)).toEqual(Option.some(2))
      yield* Fiber.interrupt(opened)
    })))

  it("lets a use waiting for a removal be interrupted", () =>
    run(Effect.gen(function* () {
      const { instances, log, deleting } = yield* fixture
      const gate = yield* Deferred.make<void>()
      yield* Ref.set(deleting, Deferred.await(gate))
      const removal = yield* Effect.fork(instances.remove("chat"))
      yield* Effect.yieldNow()
      const waiting = yield* Effect.fork(Effect.scoped(instances.get("chat")))
      yield* Effect.yieldNow()
      const interrupting = yield* Effect.fork(Fiber.interrupt(waiting))
      for (let turn = 0; turn < 100 && Option.isNone(yield* Fiber.poll(interrupting)); turn++) yield* Effect.yieldNow()
      // Interrupted while the removal still runs, having opened nothing.
      expect(Option.isSome(yield* Fiber.poll(interrupting))).toBe(true)
      expect(Option.isNone(yield* Fiber.poll(removal))).toBe(true)
      yield* Deferred.succeed(gate, undefined)
      yield* Fiber.join(removal)
      expect(yield* log).toEqual(["delete chat"])
      expect(yield* instances.current("chat")).toEqual(Option.none())
    })))
})
