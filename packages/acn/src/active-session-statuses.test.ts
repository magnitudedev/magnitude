import { describe, expect, it } from "vitest"
import { Effect, Layer, PubSub, Ref, Stream } from "effect"
import type { SessionMetadata } from "@magnitudedev/protocol"
import { AgentRuntime, type AgentRuntimeApi, type ResidentSessionSnapshot } from "./agent-runtime"
import { ActiveSessionStatusesLive, ActiveSessionStatusesService } from "./active-session-statuses"
import { SessionStore, type SessionStoreApi } from "./session-store"

const resident = (
  sessionId: string,
  workStatus: ResidentSessionSnapshot["workStatus"],
): ResidentSessionSnapshot => ({
  sessionId,
  generation: 1,
  title: "Session",
  cwd: "/tmp",
  scratchpadPath: `/tmp/${sessionId}/scratchpad`,
  createdAt: 1,
  updatedAt: 2,
  residentSince: 1,
  workStatus,
})

const protocolMeta = (sessionId: string, updatedAt: number): SessionMetadata => ({
  sessionId,
  title: "Session",
  cwd: "/tmp",
  createdAt: 1,
  updatedAt,
  messageCount: 0,
  lastMessage: null,
})

const makeSetup = Effect.gen(function* () {
  const residents = yield* Ref.make<ReadonlyArray<ResidentSessionSnapshot>>([])
  const metas = yield* Ref.make(new Map<string, SessionMetadata>())
  const changed = yield* PubSub.unbounded<void>()
  const runtime: AgentRuntimeApi = {
    withSession: () => Effect.die("unused"),
    withSessionRequest: () => Effect.die("unused"),
    tryWithResident: () => Effect.die("unused"),
    tryWithBusyResident: () => Effect.die("unused"),
    residentSessions: Ref.get(residents),
    dispose: () => Effect.void,
    deleteSession: (_sessionId, remove) => remove,
    registerRetirementObserver: () => Effect.succeed(Effect.void),
    changes: Stream.fromPubSub(changed),
  }
  const store: SessionStoreApi = {
    createId: Effect.die("unused"),
    readMeta: () => Effect.die("unused"),
    readProtocolMeta: (sessionId) =>
      Ref.get(metas).pipe(Effect.map((all) => all.get(sessionId) ?? null)),
    promoteDraft: () => Effect.die("unused"),
    listDraftSessionIds: () => Effect.die("unused"),
    listProtocolMetas: () => Effect.die("unused"),
    listSessionCwds: () => Effect.die("unused"),
    deleteSessionFiles: () => Effect.die("unused"),
    validateCwd: () => Effect.die("unused"),
    getScratchpadPath: () => Effect.die("unused"),
    getExecutionContext: () => Effect.die("unused"),
  }
  return {
    residents,
    metas,
    changed,
    layer: ActiveSessionStatusesLive.pipe(
      Layer.provide(
        Layer.mergeAll(Layer.succeed(AgentRuntime, runtime), Layer.succeed(SessionStore, store)),
      ),
    ),
  }
})

describe("ActiveSessionStatuses", () => {
  it("projects the authoritative runtime work status", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Ref.set(setup.residents, [resident("session-a", { _tag: "Working", workerCount: 2 })])
      yield* Ref.update(setup.metas, (all) =>
        new Map(all).set("session-a", protocolMeta("session-a", 42)),
      )
      const result = yield* Effect.gen(function* () {
        const statuses = yield* ActiveSessionStatusesService
        return yield* statuses.snapshot
      }).pipe(Effect.provide(setup.layer))
      expect(result.sessions).toEqual([
        {
          sessionId: "session-a",
          workStatus: "working",
          activeWorkerCount: 2,
          lastMessageAt: 42,
        },
      ])
    })
    await Effect.runPromise(program)
  })

  it("emits when a resident generation's authoritative work status changes", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Ref.set(setup.residents, [resident("session-a", { _tag: "Quiescent", workerCount: 0 })])
      yield* Ref.update(setup.metas, (all) =>
        new Map(all).set("session-a", protocolMeta("session-a", 10)),
      )
      const values = yield* Effect.gen(function* () {
        const statuses = yield* ActiveSessionStatusesService
        const fiber = yield* statuses.stream.pipe(Stream.take(2), Stream.runCollect, Effect.fork)
        yield* Effect.sleep("10 millis")
        yield* Ref.set(setup.residents, [resident("session-a", { _tag: "Working", workerCount: 1 })])
        yield* PubSub.publish(setup.changed, undefined)
        return yield* fiber
      }).pipe(Effect.provide(setup.layer))
      expect([...values][1]?.sessions[0]?.workStatus).toBe("working")
    })
    await Effect.runPromise(program)
  })
})
