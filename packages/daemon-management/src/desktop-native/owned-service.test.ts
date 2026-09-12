import { MagnitudeHealthResponseSchema, AcnIdentitySchema, AcnInstanceIdSchema, AcnRevisionSchema, type MagnitudeHealthResponse } from "@magnitudedev/acn-protocol"
import type { DesktopChildEvent, DesktopOwnerCommand } from "@magnitudedev/acn-protocol/desktop-control"
import { ProcessStartIdentitySchema } from "@magnitudedev/utils/process-groups"
import { Deferred, Effect, Fiber, Logger, Queue, Ref, Schema, Stream, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { OwnedChildSpawner } from "./owned-child"
import { makeOwnedService } from "./owned-service"

const health: MagnitudeHealthResponse = Schema.decodeUnknownSync(MagnitudeHealthResponseSchema)({
  service: "magnitude-acn", version: AcnIdentitySchema.make("0.0.14"), revision: AcnRevisionSchema.make(1),
  id: AcnInstanceIdSchema.make("test-child"), pid: 42, rpcVersion: 1, state: { _tag: "Ready" },
})
const setup = Effect.gen(function* () {
  const events = yield* Queue.unbounded<DesktopChildEvent>()
  const exit = yield* Deferred.make<number>()
  const commands = yield* Ref.make<DesktopOwnerCommand[]>([])
  const stops = yield* Ref.make(0)
  const launches = yield* Ref.make(0)
  const spawner = OwnedChildSpawner.of({ spawn: () => Effect.gen(function* () {
    yield* Ref.update(launches, value => value + 1)
    return {
      identity: { pid: 42, processStartIdentity: ProcessStartIdentitySchema.make("test") },
      events: Stream.fromQueue(events), exit: Deferred.await(exit), diagnosticTail: Effect.succeed(""),
      send: command => Ref.update(commands, values => [...values, command]),
      stop: Ref.update(stops, value => value + 1),
    }
  }) })
  const service = yield* makeOwnedService({ executable: "test", arguments: [], environment: {} }, 1).pipe(Effect.provideService(OwnedChildSpawner, spawner))
  return { service, events, exit, commands, launches, stops }
})
const run = <A, E>(effect: Effect.Effect<A, E, import("effect").Scope.Scope>) => Effect.runPromise(Effect.scoped(effect).pipe(Effect.provide(TestContext.TestContext)))

describe("desktop-owned service supervision", () => {
  it("retains diagnostics when control EOF precedes process exit without exposing them in Status", async () => {
    const messages: unknown[] = []
    const logger = Logger.make(({ message }) => { messages.push(message) })
    await run(Effect.gen(function* () {
      const spawner = OwnedChildSpawner.of({ spawn: () => Effect.succeed({
        identity: { pid: 42, processStartIdentity: ProcessStartIdentitySchema.make("test") },
        events: Stream.empty, exit: Effect.never,
        diagnosticTail: Effect.succeed("private child stack trace"),
        send: () => Effect.void, stop: Effect.void,
      }) })
      const service = yield* makeOwnedService({ executable: "test", arguments: [], environment: {} }, 1).pipe(
        Effect.provideService(OwnedChildSpawner, spawner),
      )
      yield* TestClock.adjust("8 seconds")
      const state = yield* service.state
      expect(state._tag).toBe("Failed")
      if (state._tag === "Failed") expect(state.message).toBe("Service control stream ended")
      expect(messages.filter(message => Array.isArray(message) && message.includes("private child stack trace"))).toHaveLength(4)
      yield* service.shutdown
    }).pipe(Effect.provide(Logger.replace(Logger.defaultLogger, logger))))
  })

  it("keeps the stopping detail when acknowledgement immediately releases process exit", async () => run(Effect.gen(function* () {
    let acknowledgements = 0
    const terminalHealth = Schema.decodeUnknownSync(MagnitudeHealthResponseSchema)({
      ...health, state: { _tag: "Stopping", reason: "startup-failed", safeDetail: "Inference installation is missing." },
    })
    const spawner = OwnedChildSpawner.of({ spawn: () => Effect.gen(function* () {
      const exit = yield* Deferred.make<number>()
      return {
        identity: { pid: 42, processStartIdentity: ProcessStartIdentitySchema.make("test") },
        events: Stream.fromIterable<DesktopChildEvent>([{ _tag: "Booted", pid: 42 }, { _tag: "Health", health: terminalHealth }]),
        exit: Deferred.await(exit), diagnosticTail: Effect.succeed(""), stop: Effect.void,
        send: command => command._tag === "StoppingObserved" ? Effect.sync(() => { acknowledgements++ }).pipe(
          Effect.zipRight(Deferred.succeed(exit, 1)), Effect.zipRight(Effect.never),
        ) : Effect.void,
      }
    }) })
    const service = yield* makeOwnedService({ executable: "test", arguments: [], environment: {} }, 1).pipe(
      Effect.provideService(OwnedChildSpawner, spawner),
    )
    yield* TestClock.adjust("8 seconds")
    const state = yield* service.state
    expect(state._tag).toBe("Failed")
    if (state._tag === "Failed") expect(state.message).toBe("Inference installation is missing.")
    expect(acknowledgements).toBe(4)
    yield* service.shutdown
  })))

  it("waits for matching Booted, observes readiness, and shuts down once", async () => run(Effect.gen(function* () {
    const f = yield* setup
    yield* Queue.offer(f.events, { _tag: "Booted", pid: 42 })
    yield* Queue.offer(f.events, { _tag: "Health", health })
    expect((yield* f.service.awaitReady).id).toBe(health.id)
    expect((yield* Ref.get(f.commands)).map(c => c._tag)).toEqual(["Start"])
    yield* Effect.all([f.service.shutdown, f.service.shutdown], { concurrency: "unbounded" })
    expect((yield* f.service.state)._tag).toBe("Stopped")
    expect(yield* Ref.get(f.stops)).toBe(1)
    expect((yield* Ref.get(f.commands)).map(c => c._tag)).toEqual(["Start", "Shutdown"])
  })))

  it("removes Ready immediately on child loss and Quit prevents recovery", async () => run(Effect.gen(function* () {
    const f = yield* setup
    yield* Queue.offer(f.events, { _tag: "Booted", pid: 42 })
    yield* Queue.offer(f.events, { _tag: "Health", health })
    yield* f.service.awaitReady
    yield* Deferred.succeed(f.exit, 9)
    yield* TestClock.adjust("1 millis")
    expect((yield* f.service.state)._tag).toBe("Starting")
    yield* f.service.shutdown
    yield* TestClock.adjust("20 seconds")
    expect(yield* Ref.get(f.launches)).toBe(1)
    expect((yield* f.service.state)._tag).toBe("Stopped")
  })))

  it("bounds repeated crashes and permits only explicit retry", async () => run(Effect.gen(function* () {
    const f = yield* setup
    yield* Deferred.succeed(f.exit, 9)
    yield* TestClock.adjust("8 seconds")
    expect((yield* f.service.state)._tag).toBe("Failed")
    expect(yield* Ref.get(f.launches)).toBe(4)
    yield* TestClock.adjust("1 minute")
    expect(yield* Ref.get(f.launches)).toBe(4)
    yield* f.service.retry
    yield* TestClock.adjust("1 millis")
    expect(yield* Ref.get(f.launches)).toBe(5)
    yield* f.service.shutdown
  })))

  it("never admits mismatched health", async () => run(Effect.gen(function* () {
    const f = yield* setup
    const waiter = yield* f.service.awaitReady.pipe(Effect.either, Effect.forkScoped)
    for (let index = 0; index < 4; index++) {
      yield* Queue.offer(f.events, { _tag: "Booted", pid: 42 })
      yield* Queue.offer(f.events, { _tag: "Health", health: { ...health, rpcVersion: 2 } })
      yield* TestClock.adjust(`${2 ** index} seconds`)
    }
    expect((yield* Fiber.join(waiter))._tag).toBe("Left")
    expect((yield* f.service.state)._tag).toBe("Failed")
  })))
  it("presents the service's safe startup detail without a stack trace", async () => run(Effect.gen(function* () {
    const f = yield* setup
    for (let index = 0; index < 4; index++) {
      yield* Queue.offer(f.events, { _tag: "Booted", pid: 42 })
      yield* Queue.offer(f.events, { _tag: "Health", health: Schema.decodeUnknownSync(MagnitudeHealthResponseSchema)({
        ...health, state: { _tag: "Stopping", reason: "startup-failed", safeDetail: "Inference engine installation is missing." },
      }) })
      yield* TestClock.adjust(`${2 ** index} seconds`)
    }
    const state = yield* f.service.state
    expect(state._tag).toBe("Failed")
    if (state._tag === "Failed") expect(state.message).toBe("Inference engine installation is missing.")
    yield* f.service.shutdown
  })))

})
