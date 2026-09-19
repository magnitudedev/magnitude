import releasePlan from "../../release/release-plan.json"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { DateTime, Effect, Layer, Schema, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { planRun } from "../src/catalog"
import { RunId, RunRequest } from "../src/domain"
import { GuestExecutor } from "../src/guest-executor"
import { Fence } from "../src/lease"
import { deliverOutwardWorkerResult, runOutwardWorker } from "../src/outward-worker"
import { sha256 } from "../src/snapshot"
import { WorkerApiError, WorkerClient } from "../src/worker-client"
import { WorkerInvocation, WorkerReply } from "../src/worker-protocol"

for (const mode of ["success", "delivery-error", "corrupt-input", "wrong-claim", "revoked", "saved-claim", "saved-invocation", "missing-reply", "oversized-reply", "saved-evidence", "delivery-revoked", "delivery-revoked-upload"] as const) test(`outward guest preserves single execution and evidence for ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const parent = yield* fs.makeTempDirectoryScoped({ prefix: "lab-outward-" })
  const root = join(parent, "attempt")
  const payload = new TextEncoder().encode("source fixture")
  const report = new TextEncoder().encode("native executor fixture evidence")
  const manifest = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "source", commit: "a".repeat(40),
    entries: [{ kind: "file", path: "file.ts", sha256: sha256(payload), bytes: payload.length, executable: false }] })
  const oldPackage = "previous installed package"
  const baseline = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "artifacts", release: {
    schemaVersion: 2, version: "0.1.2", acnRevision: 1, rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.2", sourceCommit: "b".repeat(40),
    artifacts: [{ id: "desktop-linux-x64", kind: "desktop", host: "linux-x64-gnu", filename: "magnitude.deb", bytes: oldPackage.length, sha256: sha256(oldPackage) }],
  } })
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "outward-guest-test", owner: "fixture", trust: "developer", mode: "verify", allowSpark: false,
    input: { kind: "source", digest: sha256(manifest) }, updateFrom: { kind: "artifacts", digest: sha256(baseline) }, selection: { kind: "custom", targets: ["ubuntu-24.04-x64-cpu-intel"], suites: ["package"], harnesses: ["pi"] },
    limits: { concurrency: 1, deadlineMinutes: 5, budgetUsd: 10, idleMinutes: 15 } })
  const plan = yield* planRun(request)
  const invocation = WorkerInvocation.make({ schemaVersion: 1, disposable: true, port: 11279, model: "fixture", assignment: { plan, target: plan.targets[0]!,
    claim: { runId: RunId.make("run-00000000-0000-0000-0000-000000000001"), targetId: plan.targets[0]!.target.id, fence: Fence.make(1), worker: "fixture" }, deadline: DateTime.unsafeMake(Date.now() + 60000) } })
  const recovery = ["delivery-error", "saved-claim", "saved-invocation", "missing-reply", "oversized-reply", "saved-evidence", "delivery-revoked", "delivery-revoked-upload"].includes(mode)
  let executed = 0, uploads = 0, submissions = 0, cleaned = 0, started = false, downloads = 0, recovering = false, redeliveryStarted = false
  const client = Layer.succeed(WorkerClient, {
    assignment: Effect.suspend(() => (mode === "revoked" && started || mode === "delivery-revoked" && recovering || mode === "delivery-revoked-upload" && redeliveryStarted) ? Effect.fail(new WorkerApiError({ status: 401, message: "Revoked fixture" })) : Effect.succeed(invocation)),
    download: digest => { downloads++; return Stream.make(digest === request.input.digest ? new TextEncoder().encode(manifest) : digest === sha256(baseline) ? new TextEncoder().encode(baseline) : digest === sha256(oldPackage) ? new TextEncoder().encode(oldPackage) : mode === "corrupt-input" ? new TextEncoder().encode("wrong bytes") : payload) },
    upload: (digest, size, content) => Effect.gen(function* () {
      uploads++
      if (mode === "delivery-revoked-upload" && recovering) {
        redeliveryStarted = true
        return yield* Effect.never.pipe(Effect.ensuring(Effect.sync(() => { cleaned++ })))
      }
      const bytes = Buffer.concat(Array.from(yield* content.pipe(Stream.runCollect)))
      expect(digest).toBe(sha256(report)); expect(size).toBe(report.length); expect(bytes.toString()).toBe(new TextDecoder().decode(report))
    }).pipe(Effect.mapError(() => new WorkerApiError({ status: 0, message: "Fixture upload failed" }))),
    submit: () => Effect.suspend(() => { submissions++; return recovery && !recovering ? Effect.fail(new WorkerApiError({ status: 500, message: "Delivery fixture failed" })) : Effect.void }),
  })
  const executor = Layer.succeed(GuestExecutor, { run: (received, directory) => Effect.gen(function* () {
    executed++; started = true
    expect(received).toEqual(invocation)
    expect(yield* fs.readFileString(join(directory, "objects", sha256(baseline)))).toBe(baseline)
    expect(yield* fs.readFileString(join(directory, "objects", sha256(oldPackage)))).toBe(oldPackage)
    expect(yield* fs.readFileString(join(directory, "objects", sha256(payload)))).toBe(new TextDecoder().decode(payload))
    if (mode === "revoked") return yield* Effect.never.pipe(Effect.ensuring(Effect.sync(() => { cleaned++ })))
    yield* fs.writeFile(join(directory, "objects", sha256(report)), report)
    const now = new Date().toISOString()
    return WorkerReply.make({ schemaVersion: 1, claim: { ...received.assignment.claim, fence: Fence.make(mode === "wrong-claim" ? 2 : 1) }, result: { cleanupErrors: [], cases: received.assignment.target.cases.map(test => ({
      targetId: received.assignment.claim.targetId, caseId: test.id, harness: test.harness, startedAt: now, endedAt: now,
      outcome: { status: "passed", detail: "Orchestration fixture, not native acceptance" }, evidence: [{ path: "evidence/result.txt", sha256: sha256(report), bytes: report.length }],
    })) } })
  }).pipe(Effect.orDie) })
  const run = runOutwardWorker({ root, pollMs: 20 }).pipe(Effect.provide([client, executor]))
  const outcome = yield* run.pipe(Effect.either)
  expect(outcome._tag).toBe(mode === "success" ? "Right" : "Left")
  expect(executed).toBe(mode === "corrupt-input" ? 0 : 1)
  expect(uploads).toBe(mode === "success" || recovery ? 1 : 0)
  expect(submissions).toBe(mode === "success" || recovery ? 1 : 0)
  expect(cleaned).toBe(mode === "revoked" ? 1 : 0)
  if (mode === "success" || recovery) {
    const saved = yield* Schema.decodeUnknown(Schema.parseJson(WorkerReply))(yield* fs.readFileString(join(root, "reply.json")))
    expect(saved.claim).toEqual(invocation.assignment.claim)
    expect((yield* run.pipe(Effect.either))._tag).toBe("Left")
    expect(executed).toBe(1)
    if (recovery) {
      if (mode === "saved-claim") yield* fs.writeFileString(join(root, "reply.json"), yield* Schema.encode(Schema.parseJson(WorkerReply))({ ...saved, claim: { ...saved.claim, fence: Fence.make(2) } }))
      if (mode === "saved-invocation") yield* fs.writeFileString(join(root, "invocation.json"), yield* Schema.encode(Schema.parseJson(WorkerInvocation))({ ...invocation, model: "different-model" }))
      if (mode === "missing-reply") yield* fs.remove(join(root, "reply.json"))
      if (mode === "oversized-reply") yield* fs.writeFileString(join(root, "reply.json"), " ".repeat(16 * 1024 * 1024 + 1))
      if (mode === "saved-evidence") yield* fs.writeFile(join(root, "objects", sha256(report)), new Uint8Array(report.length))
      const initialDownloads = downloads
      recovering = true
      // No GuestExecutor is provided: recovery cannot accidentally execute native tests.
      const delivered = yield* deliverOutwardWorkerResult({ root, pollMs: 20 }).pipe(Effect.provide(client), Effect.either)
      expect(delivered._tag).toBe(mode === "delivery-error" ? "Right" : "Left")
      expect(executed).toBe(1)
      expect(downloads).toBe(initialDownloads)
      if (mode === "delivery-revoked-upload") expect(cleaned).toBe(1)
      expect(submissions).toBe(mode === "delivery-error" ? 2 : 1)
    }
  }
})).pipe(Effect.provide(BunContext.layer))))
