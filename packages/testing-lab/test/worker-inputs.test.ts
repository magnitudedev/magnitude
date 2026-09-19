import { DateTime, Effect, Layer, Redacted, Schema, Stream } from "effect"
import { expect, test } from "vitest"
import { planRun } from "../src/catalog"
import { InfrastructureFailure, RunId, RunRequest } from "../src/domain"
import { InputRegistry } from "../src/inputs"
import { Fence } from "../src/lease"
import { sha256 } from "../src/snapshot"
import { WorkerInputs, WorkerInputsLive } from "../src/worker-inputs"
import { WorkerInvocation } from "../src/worker-protocol"
import { WorkerAccessDenied, WorkerTickets } from "../src/worker-tickets"

for (const transient of [false, true]) test(`manifest cache coalesces downloads without caching authority or transient failures (${transient})`, () => Effect.runPromise(Effect.gen(function* () {
  const content = new TextEncoder().encode("Assigned bytes")
  const digest = sha256(content)
  const manifest = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "source", commit: "a".repeat(40),
    entries: [{ kind: "file", path: "file.ts", sha256: digest, bytes: content.length, executable: false }] })
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "worker-cache-test", owner: "owner", trust: "developer",
    input: { kind: "source", digest: sha256(manifest) }, selection: { kind: "profile", profile: "quick", target: "ubuntu-24.04-x64-cpu-intel" }, mode: "verify", allowSpark: false,
    limits: { concurrency: 1, deadlineMinutes: 60, budgetUsd: 10, idleMinutes: 15 } })
  const plan = yield* planRun(request)
  const invocation = WorkerInvocation.make({ schemaVersion: 1, disposable: true, port: 11279, model: "fixture", assignment: { plan, target: plan.targets[0]!,
    claim: { runId: RunId.make("run-00000000-0000-0000-0000-000000000001"), targetId: plan.targets[0]!.target.id, fence: Fence.make(1), worker: "fixture" }, deadline: DateTime.unsafeMake(Date.now() + 60000) } })
  let authorized = true, manifestReads = 0, checks = 0
  const tickets = Layer.succeed(WorkerTickets, { issue: () => Effect.dieMessage("Not used"), withAuthority: () => Effect.dieMessage("Not used"), revoke: () => Effect.void, authorize: () => Effect.suspend(() => {
    checks++
    return authorized ? Effect.succeed(invocation) : Effect.fail(new WorkerAccessDenied({}))
  }) })
  const inputs = Layer.succeed(InputRegistry, { require: () => Effect.void, register: () => Effect.void, missing: () => Effect.succeed([]), upload: () => Effect.void,
    read: (_owner, key) => Effect.gen(function* () {
      if (key === request.input.digest) {
        manifestReads++
        if (transient && manifestReads === 1) return yield* new InfrastructureFailure({ operation: "fixture-read", message: "Transient manifest read failure" })
        yield* Effect.yieldNow()
        return Stream.make(new TextEncoder().encode(manifest))
      }
      return Stream.make(content)
    }) })
  yield* Effect.gen(function* () {
    const service = yield* WorkerInputs
    const token = Redacted.make("fixture-token")
    if (transient) expect((yield* service.read(token, digest).pipe(Effect.either))._tag).toBe("Left")
    yield* Effect.forEach(Array.from({ length: 100 }), () => service.read(token, digest).pipe(Effect.flatMap(Stream.runCollect)), { concurrency: 16 })
    expect(manifestReads).toBe(transient ? 2 : 1)
    expect(checks).toBeGreaterThanOrEqual(200)
    authorized = false
    expect((yield* service.read(token, digest).pipe(Effect.either))._tag).toBe("Left")
    expect(manifestReads).toBe(transient ? 2 : 1)
  }).pipe(Effect.provide(WorkerInputsLive.pipe(Layer.provide(Layer.merge(tickets, inputs)))))
})))
