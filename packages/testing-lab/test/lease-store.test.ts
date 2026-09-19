import { expect, test } from "vitest"
import { FileSystem, FetchHttpClient, HttpApp, HttpServer } from "@effect/platform"
import { BunContext, BunHttpServer } from "@effect/platform-bun"
import { ConfigProvider, Context, DateTime, Effect, Layer, Option, Redacted, Schema, Stream } from "effect"
import { join } from "node:path"
import { Database, initializeDatabase } from "../src/database"
import { Allocating, LeaseStore } from "../src/lease"
import { LeaseStoreLive } from "../src/lease-store"
import { LeaseId, RunId, RunRequest, TargetId } from "../src/domain"
import { RunStore, runStoreLayer } from "../src/run-store"
import { planRun } from "../src/catalog"
import { WorkStore, WorkStoreLive, type TargetResult } from "../src/work-store"
import { Principal } from "../src/domain"
import { api, bearerAuthenticator } from "../src/api"
import { LabClient, labClientLayer } from "../src/client"
import { InputRegistry, InputRegistryLive } from "../src/inputs"
import { fileArtifactStore } from "../src/artifact-store"
import { sha256 } from "../src/snapshot"
import { ProcessExecutorLive } from "../src/process"
import { snapshotArtifacts } from "../src/artifact-input"
import releasePlan from "../../release/release-plan.json"
import { temporaryDatabase } from "./postgres"
import { cli } from "../src/cli"

test("PostgreSQL fences stale workers, rolls back transactions and serializes Spark reservations", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-postgres-" })
  const database = yield* temporaryDatabase
  const objects = fileArtifactStore(join(root, "objects")).pipe(Layer.provide(BunContext.layer))
  const inputs = InputRegistryLive.pipe(Layer.provide(Layer.merge(database, objects)))
  const layers = Layer.mergeAll(database, inputs, LeaseStoreLive.pipe(Layer.provide(database)), runStoreLayer(500).pipe(Layer.provide(database)), WorkStoreLive.pipe(Layer.provide(database)))
  yield* Effect.gen(function* () {
    yield* initializeDatabase
    const store = yield* LeaseStore
    const db = yield* Database
    const state = (provider: "azure" | "spark" = "azure") => new Allocating({
      leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), runId: RunId.make(`run-${crypto.randomUUID()}`),
      targetId: TargetId.make("ubuntu-24.04-x64-cpu-intel"), provider, resourceName: `lab-${crypto.randomUUID()}`,
      expiresAt: DateTime.unsafeMake(Date.now() + 60_000),
    })
    const first = yield* store.reserve(state(), "worker-a", 30)
    const claim = { leaseId: first.state.leaseId, fence: first.fence }
    expect((yield* store.ready(claim)).state._tag).toBe("Ready")
    expect((yield* store.heartbeat(claim, 30)).fence).toBe(first.fence)
    const cleanup = yield* store.release(first.state.leaseId, "janitor", 30)
    expect(cleanup.fence).toBe(first.fence + 1)
    expect(cleanup.state._tag).toBe("Releasing")
    expect(yield* store.release(first.state.leaseId, "other-janitor", 30).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { _tag: "LeaseConflict" } })
    expect(yield* store.heartbeat(claim, 30).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { _tag: "StaleLease" } })
    expect(yield* store.ready(claim).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { _tag: "StaleLease" } })
    expect((yield* store.released({ leaseId: claim.leaseId, fence: cleanup.fence })).state._tag).toBe("Released")
    expect((yield* store.release(claim.leaseId, "janitor-again", 30)).fence).toBe(cleanup.fence)

    const results = yield* Effect.all([store.reserve(state("spark"), "a", 30).pipe(Effect.either),
      store.reserve(state("spark"), "b", 30).pipe(Effect.either)], { concurrency: 2 })
    expect(results.filter(r => r._tag === "Right")).toHaveLength(1)
    expect(results.filter(r => r._tag === "Left")).toMatchObject([{ left: { _tag: "LeaseConflict" } }])
    const before = yield* store.list()
    yield* db.transaction(tx => tx.query("DELETE FROM lab_leases").pipe(Effect.zipRight(Effect.fail("deliberate rollback")))).pipe(Effect.either)
    expect((yield* store.list()).length).toBe(before.length)
    const expired = yield* store.reserve(state(), "expired-worker", 30)
    yield* db.query("UPDATE lab_leases SET claim_expires_at=clock_timestamp()-interval '1 second' WHERE lease_id=$1", [expired.state.leaseId])
    expect(yield* store.ready({ leaseId: expired.state.leaseId, fence: expired.fence }).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { _tag: "StaleLease" } })
    const rescued = yield* store.release(expired.state.leaseId, "janitor", 30)
    expect(rescued.state._tag).toBe("Releasing")
    yield* db.query("UPDATE lab_leases SET expires_at=clock_timestamp()-interval '1 second' WHERE lease_id=$1", [rescued.state.leaseId])
    const cleanupClaim = { leaseId: rescued.state.leaseId, fence: rescued.fence }
    expect((yield* store.heartbeat(cleanupClaim, 30)).state._tag).toBe("Releasing")
    expect((yield* store.released(cleanupClaim)).state._tag).toBe("Released")

    const runs = yield* RunStore
    const manifest = '{"schemaVersion":1,"kind":"source","commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","entries":[]}'
    const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "persistent-submission", owner: "developer",
      input: { kind: "source", digest: sha256(manifest) }, selection: { kind: "profile", profile: "quick", target: "macos-26-arm64-metal-apple-silicon" },
      mode: "verify", trust: "developer", allowSpark: false, limits: { concurrency: 4, deadlineMinutes: 60, budgetUsd: 100, idleMinutes: 15 } })
    const registry = yield* InputRegistry
    yield* registry.upload(request.owner, request.input.digest, Stream.make(new TextEncoder().encode(manifest)))
    yield* registry.register(request.owner, request.input)
    const plan = yield* planRun(request)
    const submissions = yield* Effect.all([runs.submit(plan), runs.submit(plan)], { concurrency: 2 })
    expect(submissions[0].state.runId).toBe(submissions[1].state.runId)
    const runId = submissions[0].state.runId
    expect((yield* db.query("SELECT * FROM lab_work WHERE run_id=$1", [runId])).length).toBe(plan.targets.length)
    expect((yield* db.query("SELECT * FROM lab_events WHERE run_id=$1", [runId])).length).toBe(1)
    expect(yield* runs.submit({ ...plan, request: { ...request, mode: "iterate" } }).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { _tag: "AdmissionRejected" } })
    expect((yield* runs.cancel(runId)).state._tag).toBe("Cancelling")
    expect((yield* runs.cancel(runId)).state._tag).toBe("Cancelling")
    expect((yield* db.query("SELECT fence::text FROM lab_work WHERE run_id=$1", [runId]))[0]?.fence).toBe("2")
    const unaffordable = { ...plan, request: { ...request, idempotencyKey: RunRequest.fields.idempotencyKey.make("over-budget-request") }, estimatedComputeUsd: 501 }
    expect(yield* runs.submit(unaffordable).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { _tag: "AdmissionRejected" } })
    expect((yield* db.query("SELECT * FROM lab_runs")).length).toBe(1)
    const work = yield* WorkStore
    const workRequest = { ...request, idempotencyKey: RunRequest.fields.idempotencyKey.make("work-claim-request"), limits: { ...request.limits, concurrency: 1 } }
    const workPlan = yield* planRun(workRequest)
    const workRun = yield* runs.submit(workPlan)
    const assignments = yield* Effect.all([work.claim("worker-a", 30), work.claim("worker-b", 30)], { concurrency: 2 })
    const assigned = assignments.filter(Option.isSome)
    expect(assigned).toHaveLength(1)
    const assignment = assigned[0]!.value
    expect(assignment.claim.runId).toBe(workRun.state.runId)
    yield* work.heartbeat(assignment.claim, 30)
    const evidenceBytes = new TextEncoder().encode("private test trace")
    const evidenceDigest = sha256(evidenceBytes)
    yield* fs.writeFile(join(root, "objects", evidenceDigest), evidenceBytes)
    const targetResult: TargetResult = { cleanupErrors: [], cases: assignment.target.cases.map(c => ({
      targetId: assignment.claim.targetId, caseId: c.id, harness: c.harness,
      startedAt: new Date().toISOString(), endedAt: new Date().toISOString(), evidence: [{ path: "evidence/trace.zip", sha256: evidenceDigest, bytes: evidenceBytes.length }],
      outcome: { status: "blocked", detail: "Database fixture; no product execution" },
    })) }
    expect(yield* work.finish(assignment.claim, { ...targetResult, cases: targetResult.cases.slice(1) }).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { _tag: "InvalidResult" } })
    yield* work.finish(assignment.claim, targetResult)
    expect(yield* work.finish(assignment.claim, targetResult).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { _tag: "StaleWork" } })
    expect((yield* db.query("SELECT ended_at FROM lab_attempts WHERE run_id=$1", [assignment.claim.runId]))[0]?.ended_at).toBeInstanceOf(Date)
    yield* work.reconcile()
    expect((yield* runs.get(assignment.claim.runId)).state._tag).toBe("Finished")
    expect(Option.getOrThrow(yield* runs.result(assignment.claim.runId)).cases).toHaveLength(targetResult.cases.length)
    expect((yield* runs.get(runId)).state._tag).toBe("Finished")
    const retryRun = yield* runs.submit(yield* planRun({ ...workRequest, idempotencyKey: RunRequest.fields.idempotencyKey.make("retry-request-001") }))
    const lost = Option.getOrThrow(yield* work.claim("lost-worker", 30))
    yield* db.query("UPDATE lab_work SET claim_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=$1", [retryRun.state.runId])
    yield* work.reconcile()
    const retried = Option.getOrThrow(yield* work.claim("new-worker", 30))
    expect(retried.claim.fence).toBe(lost.claim.fence + 1)
    expect(yield* work.finish(lost.claim, targetResult).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { _tag: "StaleWork" } })
    yield* db.query("UPDATE lab_work SET claim_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=$1", [retryRun.state.runId])
    yield* work.reconcile()
    expect(Option.getOrThrow(yield* runs.result(retryRun.state.runId)).cases.every(c => c.outcome.status === "blocked")).toBe(true)
    expect((yield* db.query("SELECT * FROM lab_attempts WHERE run_id=$1", [retryRun.state.runId])).length).toBe(2)
    const auth = bearerAuthenticator([{ token: Redacted.make("a".repeat(40)), principal: yield* Schema.decodeUnknown(Principal)({ owner: "developer", trust: "developer" }) },
      { token: Redacted.make("b".repeat(40)), principal: yield* Schema.decodeUnknown(Principal)({ owner: "outsider", trust: "untrusted-ci" }) }])
    const http = yield* Effect.acquireRelease(Effect.sync(() => HttpApp.toWebHandlerLayer(api, Layer.mergeAll(database, inputs, objects,
      runStoreLayer(500).pipe(Layer.provide(database)), auth))), http => Effect.promise(() => http.dispose()))
    const fetch = (path: string, token: string) => Effect.promise(() => http.handler(new Request(`http://localhost${path}`, { headers: { authorization: `Bearer ${token}` } })))
    expect((yield* fetch("/v1/targets", "incorrect")).status).toBe(401)
    expect((yield* fetch("/v1/targets", "a".repeat(40))).status).toBe(200)
    expect((yield* fetch(`/v1/runs/${runId}`, "b".repeat(40))).status).toBe(403)
    const response = yield* fetch(`/v1/runs/${runId}/results`, "a".repeat(40))
    expect(response.status).toBe(200)
    const body = yield* Effect.promise(() => response.json())
    expect(body).toMatchObject({ runId, cases: expect.any(Array) })
    const evidenceUrl = `/v1/runs/${assignment.claim.runId}/evidence/${evidenceDigest}`
    const trace = yield* fetch(evidenceUrl, "a".repeat(40))
    expect(trace.status).toBe(200)
    expect(yield* Effect.promise(() => trace.text())).toBe("private test trace")
    expect((yield* fetch(evidenceUrl, "b".repeat(40))).status).toBe(403)
    expect((yield* fetch(`/v1/runs/${runId}/evidence/${evidenceDigest}`, "a".repeat(40))).status).toBe(404)
    expect((yield* fetch(`/v1/runs/${assignment.claim.runId}/evidence/${sha256("absent")}`, "a".repeat(40))).status).toBe(404)
    const apiServices = Layer.mergeAll(database, inputs, objects, runStoreLayer(500).pipe(Layer.provide(database)), auth)
    const serverContext = yield* Layer.build(BunHttpServer.layer({ hostname: "127.0.0.1", port: 0 }))
    const server = Context.get(serverContext, HttpServer.HttpServer)
    yield* server.serve(api.pipe(Effect.provide(apiServices))).pipe(Effect.provide(serverContext))
    if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP test server")
    const downloaded = join(root, "downloads", "trace.zip")
    const download = cli(["evidence", "--run", assignment.claim.runId, "--digest", evidenceDigest, "--output", downloaded]).pipe(
      Effect.withConfigProvider(ConfigProvider.fromMap(new Map([["LAB_URL", `http://127.0.0.1:${server.address.port}`], ["LAB_TOKEN", "a".repeat(40)]]))),
    )
    yield* download
    expect(yield* fs.readFileString(downloaded)).toBe("private test trace")
    expect((yield* download.pipe(Effect.either))._tag).toBe("Left")
    expect(yield* fs.readDirectory(join(root, "downloads"))).toEqual(["trace.zip"])
    yield* Effect.gen(function* () {
      const client = yield* LabClient
      expect(Buffer.concat(Array.from(yield* client.evidence(assignment.claim.runId, evidenceDigest).pipe(Stream.runCollect))).toString()).toBe("private test trace")
      expect((yield* client.identity()).owner).toBe("developer")
      expect((yield* client.targets()).length).toBe(44)
      expect((yield* client.plan(request)).targets.length).toBe(1)
      expect((yield* client.get(runId)).state._tag).toBe("Finished")
      expect(Option.getOrThrow(yield* client.result(runId)).cases.every(c => c.outcome.status === "cancelled")).toBe(true)
      const uploadedSource = '{"schemaVersion":1,"kind":"source","commit":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","entries":[]}'
      const uploadedInput = { kind: "source" as const, digest: sha256(uploadedSource) }
      expect(yield* client.registerInput(uploadedInput).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { status: 403 } })
      expect(yield* client.missing([uploadedInput.digest])).toEqual([uploadedInput.digest])
      yield* client.upload(uploadedInput.digest, Stream.make(new TextEncoder().encode(uploadedSource)))
      expect(yield* client.missing([uploadedInput.digest])).toEqual([])
      yield* client.registerInput(uploadedInput)
      expect(yield* client.upload(sha256("wrong"), Stream.make(new TextEncoder().encode("different"))).pipe(Effect.either)).toMatchObject({ _tag: "Left" })
      const packageBytes = "private unpublished package"
      const artifactDirectory = join(root, "local-artifacts")
      yield* fs.makeDirectory(artifactDirectory)
      yield* fs.writeFileString(join(artifactDirectory, "Magnitude.dmg"), packageBytes)
      yield* fs.writeFileString(join(artifactDirectory, "release.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))({
        schemaVersion: 2, version: "0.1.3", acnRevision: 1, rpc: releasePlan.rpc, plugins: [],
        tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40), artifacts: [{ id: "desktop-darwin-arm64", kind: "desktop", host: "darwin-arm64",
          filename: "Magnitude.dmg", bytes: Buffer.byteLength(packageBytes), sha256: sha256(packageBytes) }],
      }))
      const prepared = yield* snapshotArtifacts(join(artifactDirectory, "release.json"), join(root, "local-objects"))
      const artifactInput = { kind: "artifacts" as const, digest: prepared.digest }
      yield* client.upload(prepared.digest, Stream.make(new TextEncoder().encode(prepared.json)))
      expect(yield* client.registerInput(artifactInput).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { status: 403 } })
      yield* client.upload(sha256(packageBytes), Stream.make(new TextEncoder().encode(packageBytes)))
      yield* client.registerInput(artifactInput)
      const artifactRun = yield* client.submit({ ...request, input: artifactInput,
        idempotencyKey: RunRequest.fields.idempotencyKey.make("http-artifact-request") })
      expect((yield* client.get(artifactRun.state.runId)).state.plan.request.input).toEqual(artifactInput)
      yield* client.cancel(artifactRun.state.runId)
      const submission = yield* client.submit({ ...request, idempotencyKey: RunRequest.fields.idempotencyKey.make("http-client-request") })
      expect((yield* client.result(submission.state.runId))._tag).toBe("None")
      expect((yield* client.cancel(submission.state.runId)).state._tag).toBe("Cancelling")
    }).pipe(Effect.provide(labClientLayer(`http://127.0.0.1:${server.address.port}`, Effect.succeed(Redacted.make("a".repeat(40)))).pipe(Layer.provide(FetchHttpClient.layer))))
  }).pipe(Effect.provide(layers))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))), 60_000)
