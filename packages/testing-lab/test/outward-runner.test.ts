import { DateTime, Deferred, Effect, Fiber, Layer, Option, Redacted, Schema } from "effect"
import { expect, test } from "vitest"
import { planRun } from "../src/catalog"
import { InfrastructureFailure, LeaseId, RunId, RunRequest } from "../src/domain"
import { InputRegistry } from "../src/inputs"
import { Fence } from "../src/lease"
import { LocalMachine } from "../src/machines"
import { outwardWorkerRunner, type WorkerBootstrap, WorkerBootstraps } from "../src/outward-runner"
import { WorkerRunner } from "../src/scheduler"
import { WorkerInvocation, WorkerReply } from "../src/worker-protocol"
import { WorkerResults } from "../src/worker-results"
import { WorkerAccessDenied, WorkerTicketId, WorkerTickets } from "../src/worker-tickets"

for (const mode of ["success", "revoke-error", "bootstrap-error", "cancel", "deadline", "foreign-lease", "untrusted-local", "shared-disposable"] as const) test(`outward runner preserves result and credential cleanup for ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "outward-runner-test", owner: "fixture", trust: mode === "untrusted-local" ? "untrusted-ci" : "developer", mode: "verify", allowSpark: false,
    input: { kind: "source", digest: "a".repeat(64) }, selection: { kind: "custom", targets: ["ubuntu-24.04-x64-cpu-intel"], suites: ["package"], harnesses: ["pi"] },
    limits: { concurrency: 1, deadlineMinutes: 5, budgetUsd: 10, idleMinutes: 15 } })
  const plan = yield* planRun(request)
  const runId = RunId.make("run-00000000-0000-0000-0000-000000000001")
  const deadline = DateTime.unsafeMake(Date.now() + (mode === "deadline" ? 150 : 60_000))
  const assignment = { plan, target: plan.targets[0]!, claim: { runId, targetId: plan.targets[0]!.target.id, fence: Fence.make(1), worker: "fixture" }, deadline }
  const invocation = WorkerInvocation.make({ schemaVersion: 1, assignment, disposable: false, port: 11279, model: "fixture" })
  const machine = LocalMachine.make({ provider: "local", root: "/tmp/worker-fixture", tags: { schemaVersion: 1, runId: mode === "foreign-lease" ? RunId.make("run-00000000-0000-0000-0000-000000000002") : runId,
    leaseId: LeaseId.make("lease-00000000-0000-0000-0000-000000000001"), expiresAt: deadline } })
  const now = new Date().toISOString()
  const reply = WorkerReply.make({ schemaVersion: 1, claim: assignment.claim, result: { cleanupErrors: [], cases: assignment.target.cases.map(test => ({
    targetId: assignment.claim.targetId, caseId: test.id, harness: test.harness, startedAt: now, endedAt: now, evidence: [], outcome: { status: "passed", detail: "Runner fixture" },
  })) } })
  let issued = 0, revoked = 0, reads = 0
  const started = yield* Deferred.make<void>()
  const token = Redacted.make("a".repeat(43))
  const tickets = Layer.succeed(WorkerTickets, { issue: received => Effect.sync(() => { expect(received).toEqual(invocation); issued++; return { id: WorkerTicketId.make(crypto.randomUUID()), token } }),
    revoke: () => Effect.suspend(() => { revoked++; return mode === "revoke-error" ? Effect.fail(new InfrastructureFailure({ operation: "fixture-revoke", message: "Revocation failed" })) : Effect.void }),
    authorize: () => revoked ? Effect.fail(new WorkerAccessDenied({})) : Effect.succeed(invocation), withAuthority: () => Effect.dieMessage("Unused fixture method") })
  const results = Layer.succeed(WorkerResults, { submit: () => Effect.dieMessage("Unused fixture method"), read: () => Effect.sync(() => {
    reads++; return mode === "deadline" || reads === 1 ? Option.none() : Option.some(reply)
  }) })
  const inputs = Layer.succeed(InputRegistry, { require: () => Effect.void, register: () => Effect.void, missing: () => Effect.succeed([]), upload: () => Effect.void, read: () => Effect.dieMessage("Unused fixture method") })
  const bootstrap = Layer.succeed(WorkerBootstraps, { providers: new Map<"local", WorkerBootstrap>([["local", { start: (_machine, launch) => Effect.gen(function* () {
    expect(launch.root).toBe(`/tmp/worker-fixture/${machine.tags.leaseId}/attempt-1`)
    expect(launch.token).toEqual(token)
    yield* Deferred.succeed(started, undefined)
    if (mode === "bootstrap-error") return yield* new InfrastructureFailure({ operation: "fixture-start", message: "Bootstrap failed" })
    if (mode === "cancel") return yield* Effect.never
  }) }]]) })
  const runner = outwardWorkerRunner({ origin: "https://lab.example.com", pollMs: 10, runtimes: [{ provider: "local", artifactHost: "linux-x64-gnu", root: "/tmp/unused", executable: "/runner/bun", args: ["outward-worker.ts"], disposable: mode === "shared-disposable", port: 11279, model: "fixture" }] }).pipe(Layer.provide(Layer.mergeAll(tickets, results, inputs, bootstrap)))
  yield* Effect.gen(function* () {
    const executing = (yield* WorkerRunner).run(machine, assignment)
    if (mode === "cancel") {
      const fiber = yield* Effect.fork(executing)
      yield* Deferred.await(started)
      yield* Fiber.interrupt(fiber)
    } else {
      const result = yield* executing.pipe(Effect.either)
      expect(result._tag).toBe(mode === "success" || mode === "revoke-error" ? "Right" : "Left")
      if (result._tag === "Right") { expect(result.right.cases).toEqual(reply.result.cases); expect(result.right.cleanupErrors.length).toBe(mode === "revoke-error" ? 1 : 0) }
    }
    expect(issued).toBe(mode === "foreign-lease" || mode === "untrusted-local" || mode === "shared-disposable" ? 0 : 1)
    expect(revoked).toBe(issued)
  }).pipe(Effect.provide(runner))
}))))
