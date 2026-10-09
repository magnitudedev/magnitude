import { MagnitudeHealthResponseSchema, AcnIdentitySchema, AcnInstanceIdSchema, AcnRevisionSchema, type ApplicationOwnerState } from "@magnitudedev/acn-protocol"
import type { DesktopChildEvent, DesktopOwnerCommand } from "@magnitudedev/acn-protocol/desktop-control"
import { ProcessStartIdentitySchema } from "@magnitudedev/utils/process-groups"
import { Deferred, Effect, Option, Queue, Ref, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { OwnedChildSpawner } from "./owned-child"
import { makeOwnedService, type OwnerAgent } from "./owned-service"
import { ownerDone, ownerResult, ownerUnsupported } from "./owner-agent"

const ready = Schema.decodeUnknownSync(MagnitudeHealthResponseSchema)({
  service: "magnitude-acn", version: AcnIdentitySchema.make("0.0.14"), revision: AcnRevisionSchema.make(1),
  id: AcnInstanceIdSchema.make("relay-child"), pid: 42, rpcVersion: 1, state: { _tag: "Ready" },
})
const ownerState: ApplicationOwnerState = { owner: "Headless", capabilities: ["Updates", "Quit"], updates: Option.none(), loginStartup: Option.none() }

const start = (owner: OwnerAgent) => Effect.gen(function* () {
  const events = yield* Queue.unbounded<DesktopChildEvent>()
  const commands = yield* Ref.make<DesktopOwnerCommand[]>([])
  const spawner = OwnedChildSpawner.of({ spawn: () => Effect.succeed({
    identity: { pid: 42, processStartIdentity: ProcessStartIdentitySchema.make("relay") },
    events: Stream.fromQueue(events), exit: Effect.never, diagnosticTail: Effect.succeed(""),
    send: command => Ref.update(commands, values => [...values, command]),
    stop: Effect.void,
  }) })
  yield* makeOwnedService({ output: "DiagnosticTail" as const, logFile: Option.none(), executable: "test", arguments: [], environment: {} }, 1, owner).pipe(Effect.provideService(OwnedChildSpawner, spawner))
  yield* Queue.offer(events, { _tag: "Booted", pid: 42 })
  yield* Queue.offer(events, { _tag: "Health", health: ready })
  const settle = Effect.sleep("20 millis")
  return { events, commands, settle }
})
const run = <A, E>(effect: Effect.Effect<A, E, import("effect").Scope.Scope>) => Effect.runPromise(Effect.scoped(effect))

describe("owner relay", () => {
  it("forwards the owner's state after admitting the service", () => run(Effect.gen(function* () {
    const { commands, settle } = yield* start({ state: Stream.make(ownerState), handle: () => ownerUnsupported })
    yield* settle
    expect((yield* Ref.get(commands)).filter(command => command._tag === "OwnerState")).toEqual([{ _tag: "OwnerState", state: ownerState }])
  })))

  it("answers each request by id and runs its follow-up only after replying", () => run(Effect.gen(function* () {
    const order = yield* Ref.make<string[]>([])
    const { events, commands, settle } = yield* start({
      state: Stream.never,
      handle: request => request._tag === "Quit"
        ? Effect.succeed(ownerDone(Ref.update(order, values => [...values, "quit"])))
        : request._tag === "CheckUpdate" ? ownerResult(Effect.fail({ message: "offline" })) : ownerUnsupported,
    })
    yield* Queue.offer(events, { _tag: "OwnerRequest", id: 7, request: { _tag: "CheckUpdate" } })
    yield* Queue.offer(events, { _tag: "OwnerRequest", id: 8, request: { _tag: "SetLoginStartup", enabled: true } })
    yield* Queue.offer(events, { _tag: "OwnerRequest", id: 9, request: { _tag: "Quit" } })
    yield* settle
    const replies = (yield* Ref.get(commands)).filter(command => command._tag === "OwnerResponse")
    expect(replies).toEqual(expect.arrayContaining([
      { _tag: "OwnerResponse", id: 7, reply: { _tag: "Failed", message: "offline" } },
      { _tag: "OwnerResponse", id: 8, reply: { _tag: "Unsupported" } },
      { _tag: "OwnerResponse", id: 9, reply: { _tag: "Done" } },
    ]))
    expect(yield* Ref.get(order)).toEqual(["quit"])
  })))

  it("refuses owner requests before the service is admitted", () => run(Effect.gen(function* () {
    const handled = yield* Deferred.make<void>()
    const events = yield* Queue.unbounded<DesktopChildEvent>()
    const spawner = OwnedChildSpawner.of({ spawn: () => Effect.succeed({
      identity: { pid: 42, processStartIdentity: ProcessStartIdentitySchema.make("early") },
      events: Stream.fromQueue(events), exit: Effect.never, diagnosticTail: Effect.succeed(""),
      send: () => Effect.void, stop: Effect.void,
    }) })
    const service = yield* makeOwnedService({ output: "DiagnosticTail" as const, logFile: Option.none(), executable: "test", arguments: [], environment: {} }, 1, {
      state: Stream.never, handle: () => Deferred.succeed(handled, undefined).pipe(Effect.as({ reply: { _tag: "Done" as const }, afterReply: Effect.void })),
    }).pipe(Effect.provideService(OwnedChildSpawner, spawner))
    yield* Queue.offer(events, { _tag: "OwnerRequest", id: 1, request: { _tag: "Quit" } })
    yield* Effect.sleep("20 millis")
    expect(yield* Deferred.isDone(handled)).toBe(false)
    expect((yield* service.state)._tag).not.toBe("Ready")
  })))
})
