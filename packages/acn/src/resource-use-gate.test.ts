import {
  Deferred,
  Either,
  Effect,
  Fiber,
  Option,
  TestClock,
  TestContext,
} from "effect";
import { describe, expect, it } from "vitest";
import { makeResourceUseGate, ResourceRetired } from "./resource-use-gate";

const run = <A, E>(effect: Effect.Effect<A, E, never>) =>
  Effect.runPromise(Effect.provide(effect, TestContext.TestContext));

describe("ResourceUseGate", () => {
  it("retires exactly at the monotonic idle deadline", async () => {
    const program = Effect.scoped(
      Effect.gen(function* () {
        const retired = yield* Deferred.make<void>();
        const gate = yield* makeResourceUseGate({
          resource: "test",
          generation: 1,
          idleTimeout: "2 minutes",
          retire: () =>
            Deferred.succeed(retired, undefined).pipe(Effect.as(true)),
        });
        yield* Effect.yieldNow();
        yield* TestClock.adjust("119999 millis");
        expect((yield* gate.snapshot).phase).toBe("open");
        yield* TestClock.adjust("1 millis");
        yield* Deferred.await(retired);
        expect((yield* gate.snapshot).phase).toBe("retired");
      })
    );
    await run(program);
  });

  it("holds the resource until the exact lease is released", async () => {
    const program = Effect.scoped(
      Effect.gen(function* () {
        let retireCount = 0;
        const gate = yield* makeResourceUseGate({
          resource: "test",
          generation: 2,
          idleTimeout: "100 millis",
          retire: () =>
            Effect.sync(() => {
              retireCount += 1;
            }).pipe(Effect.as(true)),
        });
        const release = yield* gate.acquire("operation");
        yield* Effect.sleep("150 millis");
        expect((yield* gate.snapshot).leaseCount).toBe(1);
        expect(retireCount).toBe(0);
        yield* release;
        yield* release;
        yield* Effect.sleep("50 millis");
        expect(retireCount).toBe(0);
        yield* Effect.sleep("100 millis");
        yield* gate.awaitRetired;
        expect(retireCount).toBe(1);
      })
    );
    await Effect.runPromise(program);
  });

  it("joinIfBusy cannot turn an idle resource into demand", async () => {
    const program = Effect.scoped(
      Effect.gen(function* () {
        const gate = yield* makeResourceUseGate({
          resource: "test",
          generation: 3,
          idleTimeout: "2 minutes",
          retire: () => Effect.succeed(true),
        });
        expect(Option.isNone(yield* gate.joinIfBusy("passive"))).toBe(true);
        const releaseDemand = yield* gate.acquire("demand");
        const passive = yield* gate.joinIfBusy("passive");
        expect(Option.isSome(passive)).toBe(true);
        yield* releaseDemand;
        expect((yield* gate.snapshot).leaseCount).toBe(1);
        if (Option.isSome(passive)) yield* passive.value;
        expect((yield* gate.snapshot).leaseCount).toBe(0);
      })
    );
    await run(program);
  });

  it("linearizes admission against retirement and waits through rollback", async () => {
    const program = Effect.scoped(
      Effect.gen(function* () {
        const allowRetirement = yield* Deferred.make<void>();
        let attempts = 0;
        const gate = yield* makeResourceUseGate({
          resource: "test",
          generation: 4,
          idleTimeout: "1 second",
          retire: () =>
            Effect.gen(function* () {
              attempts += 1;
              yield* Deferred.await(allowRetirement);
              return false;
            }),
        });
        const retirementFiber = yield* gate
          .retireNow("test-claim")
          .pipe(Effect.fork);
        yield* Effect.yieldNow();
        const acquireFiber = yield* gate
          .acquire("racing-demand")
          .pipe(Effect.fork);
        yield* Effect.yieldNow();
        expect((yield* gate.snapshot).phase).toBe("retirement-claimed");
        expect(Option.isNone(yield* Fiber.poll(acquireFiber))).toBe(true);
        yield* Deferred.succeed(allowRetirement, undefined);
        expect(yield* Fiber.join(retirementFiber)).toBe(false);
        const release = yield* Fiber.join(acquireFiber);
        expect((yield* gate.snapshot).leaseCount).toBe(1);
        expect(attempts).toBe(1);
        yield* release;
      })
    );
    await run(program);
  });

  it("rejects demand after a committed retirement", async () => {
    const program = Effect.scoped(
      Effect.gen(function* () {
        const gate = yield* makeResourceUseGate({
          resource: "test",
          generation: 5,
          idleTimeout: "1 second",
          retire: () => Effect.succeed(true),
        });
        yield* Effect.yieldNow();
        yield* TestClock.adjust("1 second");
        yield* gate.awaitRetired;
        return yield* Effect.either(gate.acquire("late"));
      })
    );
    const result = await run(program);
    expect(Either.isLeft(result)).toBe(true);
    if (Either.isLeft(result))
      expect(result.left).toBeInstanceOf(ResourceRetired);
  });

  it("releases a scoped lease when its user is interrupted", async () => {
    const program = Effect.scoped(
      Effect.gen(function* () {
        const gate = yield* makeResourceUseGate({
          resource: "test",
          generation: 6,
          idleTimeout: "2 minutes",
          retire: () => Effect.succeed(true),
        });
        const never = yield* gate
          .withUse("interruptible", Effect.never)
          .pipe(Effect.fork);
        yield* Effect.yieldNow();
        expect((yield* gate.snapshot).leaseCount).toBe(1);
        yield* Fiber.interrupt(never);
        expect((yield* gate.snapshot).leaseCount).toBe(0);
      })
    );
    await run(program);
  });

  it("starts idleness only when the final concurrent lease releases", async () => {
    const program = Effect.scoped(
      Effect.gen(function* () {
        const gate = yield* makeResourceUseGate({
          resource: "test",
          generation: 7,
          idleTimeout: "1 second",
          retire: () => Effect.succeed(true),
        });
        const first = yield* gate.acquire("first");
        const second = yield* gate.acquire("second");
        yield* first;
        yield* TestClock.adjust("10 seconds");
        expect((yield* gate.snapshot).phase).toBe("open");
        yield* second;
        yield* TestClock.adjust("999 millis");
        expect((yield* gate.snapshot).phase).toBe("open");
        yield* TestClock.adjust("1 millis");
        yield* gate.awaitRetired;
      }),
    );
    await run(program);
  });

  it("gives demand arriving before the deadline a full post-release interval", async () => {
    const program = Effect.scoped(
      Effect.gen(function* () {
        const gate = yield* makeResourceUseGate({
          resource: "test",
          generation: 8,
          idleTimeout: "1 second",
          retire: () => Effect.succeed(true),
        });
        yield* TestClock.adjust("999 millis");
        const release = yield* gate.acquire("last-moment-demand");
        yield* TestClock.adjust("1 hour");
        expect((yield* gate.snapshot).phase).toBe("open");
        yield* release;
        yield* TestClock.adjust("999 millis");
        expect((yield* gate.snapshot).phase).toBe("open");
        yield* TestClock.adjust("1 millis");
        yield* gate.awaitRetired;
      }),
    );
    await run(program);
  });

  it("forced retirement drains leases and commits once", async () => {
    const program = Effect.scoped(
      Effect.gen(function* () {
        let retired = 0;
        const gate = yield* makeResourceUseGate({
          resource: "test",
          generation: 9,
          idleTimeout: "1 hour",
          retire: () => Effect.sync(() => ++retired).pipe(Effect.as(true)),
        });
        const release = yield* gate.acquire("in-flight");
        const retirement = yield* gate.retireNow("forced").pipe(Effect.fork);
        yield* Effect.yieldNow();
        expect(Option.isNone(yield* Fiber.poll(retirement))).toBe(true);
        yield* release;
        expect(yield* Fiber.join(retirement)).toBe(true);
        expect(yield* gate.retireNow("duplicate")).toBe(false);
        expect(retired).toBe(1);
      }),
    );
    await run(program);
  });

  it("closes admission immediately while existing leases remain releasable", async () => {
    const program = Effect.scoped(
      Effect.gen(function* () {
        const gate = yield* makeResourceUseGate({
          resource: "test",
          generation: 10,
          idleTimeout: "1 hour",
          retire: () => Effect.succeed(true),
        });
        const release = yield* gate.acquire("accepted-before-shutdown");

        yield* gate.closeAdmission;
        expect((yield* gate.snapshot).leaseCount).toBe(1);
        expect(yield* Effect.either(gate.acquire("late-demand"))).toMatchObject({
          _tag: "Left",
          left: { _tag: "ResourceRetired" },
        });
        const continuing = yield* gate.joinIfBusy("accepted-continuation");
        expect(Option.isSome(continuing)).toBe(true);
        expect(yield* Effect.either(gate.acquire("late-after-transfer"))).toMatchObject({
          _tag: "Left",
          left: { _tag: "ResourceRetired" },
        });

        const retirement = yield* gate.retireNow("shutdown").pipe(Effect.fork);
        yield* Effect.yieldNow();
        expect(Option.isNone(yield* Fiber.poll(retirement))).toBe(true);
        yield* release;
        if (Option.isSome(continuing)) yield* continuing.value;
        expect(yield* Fiber.join(retirement)).toBe(true);
      }),
    );
    await run(program);
  });
});
