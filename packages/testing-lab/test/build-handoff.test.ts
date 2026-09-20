import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import releasePlan from "../../release/release-plan.json"
import { fileArtifactStore } from "../src/artifact-store"
import { BuildOutput } from "../src/build-output"
import { planRun } from "../src/catalog"
import { Database, initializeDatabase } from "../src/database"
import { Digest, LeaseId, RunRequest } from "../src/domain"
import { InputRegistry, InputRegistryLive } from "../src/inputs"
import { Allocating, LeaseStore } from "../src/lease"
import { LeaseStoreLive } from "../src/lease-store"
import { ProcessExecutorLive } from "../src/process"
import { RunStore, runStoreLayer } from "../src/run-store"
import { sha256 } from "../src/snapshot"
import { WorkStore, WorkStoreLive, WorkResult, type WorkAssignment } from "../src/work-store"
import { WorkerEvidence, WorkerEvidenceLive } from "../src/worker-evidence"
import { WorkerInputs, WorkerInputsLive } from "../src/worker-inputs"
import { WorkerInvocation, WorkerReply } from "../src/worker-protocol"
import { WorkerResults, WorkerResultsLive } from "../src/worker-results"
import { WorkerTickets, WorkerTicketsLive } from "../src/worker-tickets"
import { temporaryDatabase } from "./postgres"

const result = (job: WorkAssignment, output = Option.none<BuildOutput>()) => WorkResult.make({ output, cleanupErrors: [],
  cases: job.target.cases.map(test => ({ targetId: job.claim.targetId, caseId: test.id, harness: test.harness,
    startedAt: new Date().toISOString(), endedAt: new Date().toISOString(), evidence: [],
    outcome: { status: "passed", detail: "Handoff fixture; no native application acceptance" } })),
})

test("a source producer publishes verified packages to separate consumers and never grants source access", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem, database = yield* temporaryDatabase
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-build-handoff-" })
  const storage = fileArtifactStore(join(root, "objects")), base = Layer.merge(database, storage)
  const inputs = InputRegistryLive.pipe(Layer.provide(base)), tickets = WorkerTicketsLive.pipe(Layer.provide(database))
  const services = Layer.mergeAll(base, inputs, tickets, WorkStoreLive.pipe(Layer.provide(base)),
    runStoreLayer(100).pipe(Layer.provide(database)), LeaseStoreLive.pipe(Layer.provide(database)),
    WorkerEvidenceLive.pipe(Layer.provide(Layer.merge(base, tickets))),
    WorkerResultsLive.pipe(Layer.provide(Layer.merge(base, tickets))),
    WorkerInputsLive.pipe(Layer.provide(Layer.merge(inputs, tickets))))
  yield* Effect.gen(function* () {
    yield* initializeDatabase
    // Repeated startup must work after the old key columns have been migrated.
    yield* initializeDatabase
    const db = yield* Database, runs = yield* RunStore, work = yield* WorkStore, registry = yield* InputRegistry
    const credentials = yield* WorkerTickets, uploads = yield* WorkerEvidence, replies = yield* WorkerResults
    const scopedInputs = yield* WorkerInputs, leases = yield* LeaseStore
    const sourceBytes = "unpublished local change"
    const source = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "source", commit: "a".repeat(40), entries: [
      { kind: "file", path: "unpublished.ts", sha256: sha256(sourceBytes), bytes: sourceBytes.length, executable: false },
    ] })
    const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "source-handoff-fixture", owner: "developer",
      input: { kind: "source", digest: sha256(source) }, mode: "verify", trust: "developer", allowSpark: false,
      selection: { kind: "custom", targets: ["ubuntu-24.04-x64-cpu-intel", "fedora-44-x64-cpu-amd"], suites: ["install"], harnesses: ["pi"] },
      limits: { concurrency: 2, deadlineMinutes: 60, budgetUsd: 20, idleMinutes: 15 } })
    for (const value of [source, sourceBytes]) yield* registry.upload(request.owner, sha256(value), Stream.make(new TextEncoder().encode(value)))
    yield* registry.register(request.owner, request.input)
    const plan = yield* planRun(request), run = yield* runs.submit(plan)
    expect(plan.estimatedComputeUsd).toBe(3)
    expect((yield* runs.progress(run.state.runId)).stages.map(stage => stage.state)).toEqual(["Queued", "Queued", "Queued"])
    const build = Option.getOrThrow(yield* work.claim("builder", 60))
    expect(build.work.kind).toBe("build")
    expect(build.target.cases.map(test => test.id)).toEqual(["P1", "P2"])
    expect(Option.isNone(yield* work.claim("early-consumer", 60))).toBe(true)
    const allocate = (job: WorkAssignment) => leases.reserve(new Allocating({ leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`),
      runId: job.claim.runId, workId: job.claim.workId, workFence: job.claim.fence, targetId: job.claim.targetId,
      provider: "azure", resourceName: `fixture-${crypto.randomUUID()}`, expiresAt: job.deadline }), job.claim.worker, 60)
    const producer = yield* allocate(build)
    const active = (yield* runs.progress(run.state.runId)).stages[0]!
    expect(active).toMatchObject({ id: build.claim.workId, kind: "build", state: "Running", attempts: 1 })
    expect(active.leases[0]).toMatchObject({ id: producer.state.leaseId, state: "Allocating" })
    const ticket = yield* credentials.issue(WorkerInvocation.make({ schemaVersion: 1, assignment: build, disposable: true, port: 11279, model: "fixture" }))
    const packageBytes = "exact native package fixture", packageDigest = sha256(packageBytes)
    const artifact = (kind: string, filename: string, extra = {}) => ({ id: filename, kind, host: "linux-x64-gnu", filename, sha256: packageDigest, bytes: packageBytes.length, ...extra })
    const manifest = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "artifacts", release: {
      schemaVersion: 2, version: "0.1.3", acnRevision: 1, rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40),
      artifacts: [artifact("desktop", "magnitude.deb"), artifact("desktop", "magnitude.rpm"), artifact("acn", "acn.tar.gz"),
        artifact("icn-base", "icn.tar.gz", { backend: "cpu", nativeBuild: "fixture", backendModuleAbi: "fixture" })],
    } })
    const output = BuildOutput.make({ sourceDigest: request.input.digest, sourceCommit: "a".repeat(40), artifactDigest: sha256(manifest), artifactHost: "linux-x64-gnu", backend: "cpu" })
    const reply = WorkerReply.make({ schemaVersion: 1, claim: build.claim, result: result(build, Option.some(output)) })
    yield* uploads.upload(ticket.token, sha256(manifest), Buffer.byteLength(manifest), Stream.make(new TextEncoder().encode(manifest)))
    expect((yield* replies.submit(ticket.token, reply).pipe(Effect.either))._tag).toBe("Left")
    yield* uploads.upload(ticket.token, packageDigest, packageBytes.length, Stream.make(new TextEncoder().encode(packageBytes)))
    yield* replies.submit(ticket.token, reply)
    expect(Option.isNone(yield* work.claim("still-early-consumer", 60))).toBe(true)
    const cleanup = yield* leases.release(producer.state.leaseId, "builder-cleanup", 60)
    yield* leases.released({ leaseId: cleanup.state.leaseId, fence: cleanup.fence })
    expect((yield* work.finish(build.claim, { ...reply.result, output: Option.some({ ...output, sourceDigest: Digest.make("b".repeat(64)) }) }).pipe(Effect.either))._tag).toBe("Left")
    yield* work.finish(build.claim, reply.result)
    // The next coordinator instance reconstructs dependencies and exact output from PostgreSQL.
    const restarted = Context.get(yield* Layer.build(WorkStoreLive.pipe(Layer.provide(base))), WorkStore)
    const consumers = [Option.getOrThrow(yield* restarted.claim("consumer-a", 60)), Option.getOrThrow(yield* restarted.claim("consumer-b", 60))]
    expect(consumers.every(job => job.work.kind === "test" && job.input.kind === "artifacts" && job.input.digest === output.artifactDigest)).toBe(true)
    expect(consumers.every(job => job.target.cases.every(test => test.id !== "P1" && test.id !== "P2"))).toBe(true)
    for (const consumer of consumers) {
      const machine = yield* allocate(consumer)
      expect(machine.state.leaseId).not.toBe(producer.state.leaseId)
      expect((yield* leases.list()).find(item => item.state.leaseId === producer.state.leaseId)!.state._tag).toBe("Released")
      const consumerTicket = yield* credentials.issue(WorkerInvocation.make({ schemaVersion: 1, assignment: consumer, disposable: true, port: 11279, model: "fixture" }))
      expect((yield* scopedInputs.read(consumerTicket.token, request.input.digest).pipe(Effect.either))._tag).toBe("Left")
      expect((yield* scopedInputs.read(consumerTicket.token, sha256(sourceBytes)).pipe(Effect.either))._tag).toBe("Left")
      expect(Buffer.concat(Array.from(yield* scopedInputs.read(consumerTicket.token, packageDigest).pipe(Effect.flatMap(Stream.runCollect)))).toString()).toBe(packageBytes)
      const released = yield* leases.release(machine.state.leaseId, "consumer-cleanup", 60)
      yield* leases.released({ leaseId: released.state.leaseId, fence: released.fence })
      yield* restarted.finish(consumer.claim, result(consumer))
    }
    yield* restarted.reconcile()
    const completed = Option.getOrThrow(yield* runs.result(run.state.runId))
    expect(completed.cases).toHaveLength(plan.targets.reduce((sum, target) => sum + target.cases.length, 0))
    expect(completed.cases.every(test => test.outcome.status === "passed")).toBe(true)
    expect(completed.cases.filter(test => test.caseId === "P2").every(test => test.evidence.some(item => item.path === "evidence/producer-receipt.json"))).toBe(true)
    expect((yield* db.query("SELECT * FROM lab_work WHERE run_id=$1", [run.state.runId]))).toHaveLength(3)
    expect((yield* leases.list()).every(item => item.state._tag === "Released")).toBe(true)
  }).pipe(Effect.provide(services))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
