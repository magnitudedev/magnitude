import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { DateTime, Effect, Layer, Schema, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { planRun } from "../src/catalog"
import { RunId, RunRequest } from "../src/domain"
import { GuestExecutor } from "../src/guest-executor"
import { Fence } from "../src/lease"
import { runOutwardWorker } from "../src/outward-worker"
import { sha256 } from "../src/snapshot"
import { WorkerApiError, WorkerClient } from "../src/worker-client"
import { WorkerInvocation, WorkerReply } from "../src/worker-protocol"

for (const mode of ["success", "delivery-error", "corrupt-input", "wrong-claim", "revoked"] as const) test(`outward guest preserves single execution and evidence for ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const parent = yield* fs.makeTempDirectoryScoped({ prefix: "lab-outward-" })
  const root = join(parent, "attempt")
  const payload = new TextEncoder().encode("source fixture")
  const report = new TextEncoder().encode("native executor fixture evidence")
  const manifest = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "source", commit: "a".repeat(40),
    entries: [{ kind: "file", path: "file.ts", sha256: sha256(payload), bytes: payload.length, executable: false }] })
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "outward-guest-test", owner: "fixture", trust: "developer", mode: "verify", allowSpark: false,
    input: { kind: "source", digest: sha256(manifest) }, selection: { kind: "custom", targets: ["ubuntu-24.04-x64-cpu-intel"], suites: ["package"], harnesses: ["pi"] },
    limits: { concurrency: 1, deadlineMinutes: 5, budgetUsd: 10, idleMinutes: 15 } })
  const plan = yield* planRun(request)
  const invocation = WorkerInvocation.make({ schemaVersion: 1, disposable: true, port: 11279, model: "fixture", assignment: { plan, target: plan.targets[0]!,
    claim: { runId: RunId.make("run-00000000-0000-0000-0000-000000000001"), targetId: plan.targets[0]!.target.id, fence: Fence.make(1), worker: "fixture" }, deadline: DateTime.unsafeMake(Date.now() + 60000) } })
  let executed = 0, uploads = 0, submissions = 0, cleaned = 0, started = false
  const client = Layer.succeed(WorkerClient, {
    assignment: Effect.suspend(() => mode === "revoked" && started ? Effect.fail(new WorkerApiError({ status: 401, message: "Revoked fixture" })) : Effect.succeed(invocation)),
    download: digest => Stream.make(digest === request.input.digest ? new TextEncoder().encode(manifest) : mode === "corrupt-input" ? new TextEncoder().encode("wrong bytes") : payload),
    upload: (digest, size, content) => Effect.gen(function* () {
      uploads++
      const bytes = Buffer.concat(Array.from(yield* content.pipe(Stream.runCollect)))
      expect(digest).toBe(sha256(report)); expect(size).toBe(report.length); expect(bytes.toString()).toBe(new TextDecoder().decode(report))
    }).pipe(Effect.mapError(() => new WorkerApiError({ status: 0, message: "Fixture upload failed" }))),
    submit: () => Effect.suspend(() => { submissions++; return mode === "delivery-error" ? Effect.fail(new WorkerApiError({ status: 500, message: "Delivery fixture failed" })) : Effect.void }),
  })
  const executor = Layer.succeed(GuestExecutor, { run: (received, directory) => Effect.gen(function* () {
    executed++; started = true
    expect(received).toEqual(invocation)
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
  expect(uploads).toBe(mode === "success" || mode === "delivery-error" ? 1 : 0)
  expect(submissions).toBe(mode === "success" || mode === "delivery-error" ? 1 : 0)
  expect(cleaned).toBe(mode === "revoked" ? 1 : 0)
  if (mode === "success" || mode === "delivery-error") {
    const saved = yield* Schema.decodeUnknown(Schema.parseJson(WorkerReply))(yield* fs.readFileString(join(root, "reply.json")))
    expect(saved.claim).toEqual(invocation.assignment.claim)
    expect((yield* run.pipe(Effect.either))._tag).toBe("Left")
    expect(executed).toBe(1)
  }
})).pipe(Effect.provide(BunContext.layer))))
