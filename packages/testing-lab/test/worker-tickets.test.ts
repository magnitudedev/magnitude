import { FetchHttpClient, FileSystem, HttpClient, HttpClientRequest, HttpServer } from "@effect/platform"
import { BunContext, BunHttpServer } from "@effect/platform-bun"
import { Context, Effect, Layer, Option, Redacted, Schema, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { planRun } from "../src/catalog"
import { Database, initializeDatabase } from "../src/database"
import { InfrastructureFailure, RunRequest } from "../src/domain"
import { Fence } from "../src/lease"
import { ProcessExecutorLive } from "../src/process"
import { RunStore, runStoreLayer } from "../src/run-store"
import { WorkStore, WorkStoreLive } from "../src/work-store"
import { WorkerInvocation, WorkerReply } from "../src/worker-protocol"
import { WorkerResults, WorkerResultsLive } from "../src/worker-results"
import { WorkerEvidence, WorkerEvidenceLimits, WorkerEvidenceLive } from "../src/worker-evidence"
import { WorkerClient, workerClientLayer } from "../src/worker-client"
import { WorkerTickets, WorkerTicketsLive } from "../src/worker-tickets"
import { workerApi } from "../src/worker-api"
import { WorkerInputsLive } from "../src/worker-inputs"
import { InputRegistry, InputRegistryLive } from "../src/inputs"
import { fileArtifactStore } from "../src/artifact-store"
import { sha256 } from "../src/snapshot"
import { temporaryDatabase } from "./postgres"
import releasePlan from "../../release/release-plan.json"

test("worker credentials are durable, attempt-bound and immediately invalidated by lost authority", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const database = yield* temporaryDatabase
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-worker-inputs-" })
  const storage = fileArtifactStore(join(root, "objects"))
  const inputs = InputRegistryLive.pipe(Layer.provide(Layer.merge(database, storage)))
  const ticketService = WorkerTicketsLive.pipe(Layer.provide(database))
  const services = Layer.mergeAll(database, inputs, runStoreLayer(1000).pipe(Layer.provide(database)), WorkStoreLive.pipe(Layer.provide(database)), ticketService,
    WorkerResultsLive.pipe(Layer.provide(Layer.merge(database, ticketService))),
    WorkerEvidenceLive.pipe(Layer.provide(Layer.mergeAll(database, ticketService, storage))),
    WorkerInputsLive.pipe(Layer.provide(Layer.merge(inputs, ticketService))))
  yield* Effect.gen(function* () {
    yield* initializeDatabase
    const db = yield* Database, runs = yield* RunStore, work = yield* WorkStore, tickets = yield* WorkerTickets
    const server = yield* HttpServer.HttpServer
    yield* server.serve(workerApi)
    if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP server")
    const url = `http://127.0.0.1:${server.address.port}/v1/worker/assignment`
    const http = yield* HttpClient.HttpClient
    const registry = yield* InputRegistry
    const receipts = yield* WorkerResults
    const evidence = yield* WorkerEvidence
    const payload = new TextEncoder().encode("Assigned source content")
    const unrelated = new TextEncoder().encode("Unrelated private upload owned by the same developer")
    const sourceManifest = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "source", commit: "a".repeat(40),
      entries: [{ kind: "file", path: "source.txt", sha256: sha256(payload), bytes: payload.byteLength, executable: false }] })
    const artifactManifest = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "artifacts", release: {
      schemaVersion: 2, version: "0.1.3", acnRevision: 1, rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40),
      artifacts: [{ id: "desktop-linux-x64-gnu", kind: "desktop", host: "linux-x64-gnu", filename: "Magnitude.deb", bytes: payload.length, sha256: sha256(payload) }],
    } })
    const objectUrl = (digest: string) => url.replace("/assignment", `/objects/${digest}`)
    expect((yield* http.get(url)).status).toBe(401)
    for (const mode of ["revoked", "cancelled", "expired-claim", "expired-run", "reassigned", "finished", "artifacts"] as const) {
      const manifest = mode === "artifacts" ? artifactManifest : sourceManifest
      const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: `ticket-${mode}`, owner: "fixture", trust: "developer",
        input: { kind: mode === "artifacts" ? "artifacts" : "source", digest: sha256(manifest) }, selection: { kind: "custom", targets: ["ubuntu-24.04-x64-cpu-intel"], suites: ["package"], harnesses: ["pi"] },
        mode: "verify", allowSpark: false, limits: { concurrency: 1, deadlineMinutes: 5, budgetUsd: 10, idleMinutes: 15 } })
      const run = yield* runs.submit(yield* planRun(request))
      yield* registry.upload(request.owner, sha256(payload), Stream.make(payload))
      yield* registry.upload(request.owner, sha256(unrelated), Stream.make(unrelated))
      yield* registry.upload(request.owner, sha256(manifest), Stream.make(new TextEncoder().encode(manifest)))
      yield* registry.register(request.owner, request.input)
      const assignment = Option.getOrThrow(yield* work.claim(`worker-${mode}`, 60))
      const invocation = WorkerInvocation.make({ schemaVersion: 1, assignment, disposable: true, port: 11279, model: "fixture" })
      const forged = { ...invocation, assignment: { ...assignment, target: { ...assignment.target, cases: [] } } }
      expect((yield* tickets.issue(forged).pipe(Effect.either))._tag).toBe("Left")
      const ticket = yield* tickets.issue(invocation)
      const guest = Context.get(yield* Layer.build(workerClientLayer(url.replace("/v1/worker/assignment", ""), ticket.token)), WorkerClient)
      expect(yield* guest.assignment).toEqual(invocation)
      expect(Buffer.concat(Array.from(yield* guest.download(sha256(payload)).pipe(Stream.runCollect))).toString("utf8")).toBe(new TextDecoder().decode(payload))
      expect((yield* guest.download(sha256(unrelated)).pipe(Stream.runDrain, Effect.either))._tag).toBe("Left")
      if (mode === "revoked") {
        let consumed = false
        const tooLarge = yield* evidence.upload(ticket.token, sha256(unrelated), WorkerEvidenceLimits.objectBytes + 1,
          Stream.fromEffect(Effect.sync(() => { consumed = true; return unrelated }))).pipe(Effect.either)
        expect(tooLarge._tag === "Left" && tooLarge.left._tag).toBe("InvalidResult")
        expect(consumed).toBe(false)
        const short = yield* evidence.upload(ticket.token, sha256(unrelated), unrelated.length + 1, Stream.make(unrelated)).pipe(Effect.either)
        expect(short._tag).toBe("Left")
        const interrupted = yield* evidence.upload(ticket.token, sha256(unrelated), unrelated.length,
          Stream.fail(new InfrastructureFailure({ operation: "fixture-stream", message: "Interrupted upload fixture" }))).pipe(Effect.either)
        expect(interrupted._tag).toBe("Left")
        expect((yield* db.query("SELECT 1 FROM lab_worker_objects WHERE run_id=$1", [assignment.claim.runId]))).toHaveLength(0)
        yield* db.query(`INSERT INTO lab_worker_objects(run_id,target_id,fence,digest,bytes,upload_id,upload_expires_at)
          SELECT $1,$2,$3,repeat(md5(n::text),2),$4,md5(n::text)::uuid,clock_timestamp()+interval '15 minutes'
          FROM generate_series(1,16) n`, [assignment.claim.runId, assignment.claim.targetId, assignment.claim.fence, WorkerEvidenceLimits.objectBytes])
        const full = yield* evidence.upload(ticket.token, sha256(unrelated), unrelated.length, Stream.make(unrelated)).pipe(Effect.either)
        expect(full._tag === "Left" && full.left._tag).toBe("InvalidResult")
        yield* db.query("DELETE FROM lab_worker_objects WHERE run_id=$1 AND state='Uploading'", [assignment.claim.runId])
      }
      expect(yield* tickets.authorize(ticket.token)).toEqual(invocation)
      const response = yield* http.get(url, { headers: { authorization: `Bearer ${Redacted.value(ticket.token)}` } })
      expect(response.status).toBe(200)
      expect(response.headers["cache-control"]).toBe("no-store")
      expect(yield* Schema.decodeUnknown(WorkerInvocation)(yield* response.json)).toEqual(invocation)
      const auth = { headers: { authorization: `Bearer ${Redacted.value(ticket.token)}` } }
      const inputResponse = yield* http.get(objectUrl(sha256(manifest)), auth)
      expect(inputResponse.status).toBe(200)
      expect(yield* inputResponse.text).toBe(manifest)
      const fileResponse = yield* http.get(objectUrl(sha256(payload)), auth)
      expect(fileResponse.status).toBe(200)
      expect(yield* fileResponse.text).toBe(new TextDecoder().decode(payload))
      expect((yield* http.get(objectUrl(sha256(unrelated)), auth)).status).toBe(401)
      const now = new Date().toISOString()
      let reply = WorkerReply.make({ schemaVersion: 1, claim: assignment.claim, result: { cleanupErrors: [], cases: assignment.target.cases.map(test => ({
        targetId: assignment.claim.targetId, caseId: test.id, harness: test.harness, startedAt: now, endedAt: now, evidence: [],
        outcome: { status: "passed", detail: "Result transport fixture, not native acceptance" },
      })) } })
      const send = (value: typeof WorkerReply.Type) => Effect.gen(function* () {
        const json = yield* Schema.encode(Schema.parseJson(WorkerReply))(value)
        return yield* http.execute(HttpClientRequest.post(url.replace("/assignment", "/result"), auth).pipe(HttpClientRequest.bodyText(json, "application/json")))
      })
      expect(Option.isNone(yield* receipts.read(assignment.claim))).toBe(true)
      expect((yield* send({ ...reply, claim: { ...reply.claim, fence: Fence.make(reply.claim.fence + 1) } })).status).toBe(409)
      expect((yield* send({ ...reply, result: { ...reply.result, cases: reply.result.cases.slice(1) } })).status).toBe(409)
      const unverified = { ...reply, result: { ...reply.result, cases: reply.result.cases.map(test => ({ ...test, evidence: [{ path: "evidence/missing.json", sha256: sha256(unrelated), bytes: unrelated.length }] })) } }
      expect((yield* send(unverified)).status).toBe(409)
      const upload = (bytes: Uint8Array) => http.execute(HttpClientRequest.put(url.replace("/assignment", `/evidence/${sha256(unrelated)}`), auth).pipe(
        HttpClientRequest.bodyUint8Array(bytes), HttpClientRequest.setHeader("content-length", String(bytes.length))))
      expect((yield* upload(new TextEncoder().encode("corrupt"))).status).toBe(500)
      expect((yield* db.query("SELECT 1 FROM lab_worker_objects WHERE run_id=$1", [assignment.claim.runId]))).toHaveLength(0)
      yield* guest.upload(sha256(unrelated), unrelated.length, Stream.make(unrelated))
      expect((yield* upload(unrelated)).status).toBe(204)
      reply = unverified
      yield* guest.submit(reply)
      expect((yield* send(reply)).status).toBe(204)
      expect(Option.getOrThrow(yield* receipts.read(assignment.claim))).toEqual(reply)
      expect((yield* send({ ...reply, result: { ...reply.result, cleanupErrors: ["Changed reply"] } })).status).toBe(409)
      expect((yield* tickets.issue(invocation).pipe(Effect.either))._tag).toBe("Left")
      expect((yield* tickets.authorize(Redacted.make("x".repeat(43))).pipe(Effect.either))._tag).toBe("Left")
      const rows = yield* db.query("SELECT token_digest,invocation FROM lab_worker_tickets WHERE ticket_id=$1", [ticket.id])
      expect(rows[0]?.token_digest).not.toBe(Redacted.value(ticket.token))
      expect(rows[0]?.invocation).not.toContain(Redacted.value(ticket.token))
      // Reconstruct the service over the same database: credentials survive coordinator service restart.
      expect(yield* Effect.flatMap(WorkerTickets, service => service.authorize(ticket.token)).pipe(Effect.provide(WorkerTicketsLive))).toEqual(invocation)
      if (mode === "revoked") {
        const revokedUpload = yield* evidence.upload(ticket.token, sha256(unrelated), unrelated.length,
          Stream.concat(Stream.make(unrelated), Stream.drain(Stream.fromEffect(tickets.revoke(ticket.id))))).pipe(Effect.either)
        expect(revokedUpload._tag === "Left" && revokedUpload.left._tag).toBe("WorkerAccessDenied")
      }
      if (mode === "artifacts") yield* tickets.revoke(ticket.id)
      if (mode === "cancelled") yield* runs.cancel(run.state.runId)
      if (mode === "expired-claim") yield* db.query("UPDATE lab_work SET claim_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=$1", [run.state.runId])
      if (mode === "expired-run") yield* db.query("UPDATE lab_runs SET deadline=clock_timestamp()-interval '1 second' WHERE run_id=$1", [run.state.runId])
      if (mode === "reassigned") yield* db.query("UPDATE lab_work SET fence=fence+1,worker='replacement' WHERE run_id=$1", [run.state.runId])
      if (mode === "finished") {
        const now = new Date().toISOString()
        yield* work.finish(assignment.claim, { cleanupErrors: [], cases: assignment.target.cases.map(test => ({ targetId: assignment.claim.targetId, caseId: test.id, harness: test.harness,
          startedAt: now, endedAt: now, evidence: [], outcome: { status: "passed", detail: "Credential lifecycle fixture, not native acceptance" } })) })
      }
      const denied = yield* tickets.authorize(ticket.token).pipe(Effect.either)
      expect(denied._tag === "Left" && denied.left._tag).toBe("WorkerAccessDenied")
      const rejected = yield* http.get(url, { headers: { authorization: `Bearer ${Redacted.value(ticket.token)}` } })
      expect(rejected.status).toBe(401)
      expect(yield* rejected.json).toEqual({ error: "WorkerAccessDenied" })
      expect((yield* http.get(objectUrl(sha256(payload)), auth)).status).toBe(401)
      expect((yield* send(reply)).status).toBe(401)
      expect((yield* upload(unrelated)).status).toBe(401)
      const guestDenied = yield* guest.assignment.pipe(Effect.either)
      expect(guestDenied._tag === "Left" && guestDenied.left.status).toBe(401)
    }
  }).pipe(Effect.provide(services))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive, FetchHttpClient.layer, BunHttpServer.layer({ hostname: "127.0.0.1", port: 0 })]))))
