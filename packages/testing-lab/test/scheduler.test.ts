import { expect, test } from "vitest"
import { BunContext } from "@effect/platform-bun"
import { DateTime, Deferred, Effect, Exit, Fiber, Layer, Option, Schema } from "effect"
import { initializeDatabase } from "../src/database"
import { InfrastructureFailure, RunRequest } from "../src/domain"
import { planRun } from "../src/catalog"
import { LeaseStore } from "../src/lease"
import { LeaseStoreLive } from "../src/lease-store"
import { type Machine, type MachineAllocator } from "../src/machines"
import { RunStore, runStoreLayer } from "../src/run-store"
import { MachineProviders, Scheduler, schedulerLayer, WorkerRunner } from "../src/scheduler"
import { WorkStoreLive, type TargetResult } from "../src/work-store"
import { ProcessExecutorLive } from "../src/process"
import { temporaryDatabase } from "./postgres"

const fixture = (mode: "success" | "ambiguous" | "cancel" | "cleanup-timeout") => Effect.scoped(Effect.gen(function* () {
  const database = yield* temporaryDatabase
  const stores = Layer.mergeAll(database, runStoreLayer(100).pipe(Layer.provide(database)), WorkStoreLive.pipe(Layer.provide(database)), LeaseStoreLive.pipe(Layer.provide(database)))
  return yield* Effect.gen(function* () {
    yield* initializeDatabase
    const leases = yield* LeaseStore
    const runs = yield* RunStore
    const started = yield* Deferred.make<void>()
    const inventory: Machine[] = []
    let allocations = 0, removals = 0
    const allocator: MachineAllocator = {
      inventory: () => Effect.sync(() => [...inventory]),
      ensure: (lease, target) => Effect.gen(function* () {
        expect((yield* leases.list()).some(l => l.state.leaseId === lease.leaseId && l.state._tag === "Allocating")).toBe(true)
        const machine: Machine = { provider: "azure", id: "fixture-vm", name: lease.resourceName,
          tags: { schemaVersion: 1, runId: lease.runId, leaseId: lease.leaseId, expiresAt: lease.expiresAt } }
        inventory.push(machine); allocations++
        if (mode === "ambiguous") return yield* new InfrastructureFailure({ operation: "fixture-create", message: "Connection failed after allocation" })
        expect(target.id).toBe(lease.targetId)
        return machine
      }),
      release: machine => mode === "cleanup-timeout" ? Effect.never : Effect.sync(() => {
        const index = inventory.findIndex(m => m.tags.leaseId === machine.tags.leaseId)
        if (index !== -1) { inventory.splice(index, 1); removals++ }
      }),
    }
    const runner: WorkerRunner = { run: (machine, job) => Effect.gen(function* () {
      expect((yield* leases.list()).some(l => l.state.leaseId === machine.tags.leaseId && l.state._tag === "Ready")).toBe(true)
      expect(DateTime.toEpochMillis(machine.tags.expiresAt)).toBe(DateTime.toEpochMillis(job.deadline))
      yield* Deferred.succeed(started, undefined)
      if (mode === "cancel") return yield* Effect.never
      const now = new Date().toISOString()
      return { cleanupErrors: [], cases: job.target.cases.map(c => ({ targetId: job.target.target.id, caseId: c.id, harness: c.harness,
        startedAt: now, endedAt: now, evidence: [], outcome: { status: "passed", detail: "Scheduler fixture, not application qualification" } })) } satisfies TargetResult
    }) }
    const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: `fixture-${mode}`, owner: "fixture", trust: "developer",
      input: { kind: "source", digest: "a".repeat(64) }, selection: { kind: "custom", targets: ["ubuntu-24.04-x64-cpu-intel"], suites: ["package"], harnesses: ["pi"] },
      mode: "verify", allowSpark: false, limits: { concurrency: 1, deadlineMinutes: 5, budgetUsd: 10, idleMinutes: 15 } })
    const run = yield* runs.submit(yield* planRun(request))
    yield* Effect.gen(function* () {
      const scheduler = yield* Scheduler
      if (mode === "cancel") {
        const executing = yield* scheduler.next("fixture-worker").pipe(Effect.fork)
        yield* Deferred.await(started)
        yield* runs.cancel(run.state.runId)
        // No explicit fiber interruption: lost authority must stop the worker via heartbeat.
        const exit = yield* Fiber.await(executing).pipe(Effect.timeout("15 seconds"))
        expect(Exit.isFailure(exit)).toBe(true)
        yield* scheduler.reconcile("fixture-janitor")
      } else expect(yield* scheduler.next("fixture-worker")).toBe(true)
      if (mode === "cleanup-timeout") {
        expect(inventory).toHaveLength(1)
        expect((yield* leases.list()).every(l => l.state._tag === "Releasing")).toBe(true)
        expect(Option.isNone(yield* runs.result(run.state.runId))).toBe(true)
        return
      }
      expect(inventory).toEqual([])
      expect(allocations).toBe(1); expect(removals).toBe(1)
      expect((yield* leases.list()).every(l => l.state._tag === "Released")).toBe(true)
      const result = Option.getOrThrow(yield* runs.result(run.state.runId))
      expect(result.cases.every(c => c.outcome.status === (mode === "success" ? "passed" : mode === "ambiguous" ? "blocked" : "cancelled"))).toBe(true)
      if (mode === "ambiguous") expect(result.cases[0]?.outcome.detail).toContain("Connection failed after allocation")
    }).pipe(Effect.provide(schedulerLayer(mode === "cleanup-timeout" ? "100 millis" : "20 minutes").pipe(Layer.provide(Layer.mergeAll(stores,
      Layer.succeed(MachineProviders, { allocators: new Map([["azure" as const, allocator]]) }), Layer.succeed(WorkerRunner, runner))))))
  }).pipe(Effect.provide(stores))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))

test.each(["success", "ambiguous", "cancel", "cleanup-timeout"] as const)("scheduler preserves durable ownership and cleanup for %s", mode => Effect.runPromise(fixture(mode)), 30_000)
