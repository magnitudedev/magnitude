import { Option } from "effect"
import { UpdateConfiguration } from "@magnitudedev/release/hosted-update"
import { TestWork } from "../src/execution-plan"
import { WorkId } from "../src/work-identity"
import releasePlan from "../../release/release-plan.json"
import { DateTime, Effect, Layer, Redacted, Schema, Stream } from "effect"
import { expect, test } from "vitest"
import { planRun } from "../src/catalog"
import { InfrastructureFailure, RunId, RunRequest } from "../src/domain"
import { ArtifactInput, InputDenied, InputRegistry } from "../src/inputs"
import { Fence } from "../src/lease"
import { sha256 } from "../src/snapshot"
import { WorkerInputs, WorkerInputsLive } from "../src/worker-inputs"
import { WorkerInvocation } from "../src/worker-protocol"
import { WorkerAccessDenied, WorkerTickets } from "../src/worker-tickets"

for (const transient of [false, true]) test(`manifest cache coalesces downloads without caching authority or transient failures (${transient})`, () => Effect.runPromise(Effect.gen(function* () {
  const content = new TextEncoder().encode("Assigned bytes")
  const digest = sha256(content)
  const baseManifest = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "artifacts", release: {
    schemaVersion: 2, version: "0.1.3", acnRevision: 1, rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40),
    artifacts: [{ id: "desktop-linux-x64", kind: "desktop", host: "linux-x64-gnu", filename: "magnitude.deb", bytes: content.length, sha256: sha256(content) }],
  } })
  const oldBytes = "old package"
  const baseline = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "artifacts", release: {
    schemaVersion: 2, version: "0.1.2", acnRevision: 1, rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.2", sourceCommit: "b".repeat(40),
    artifacts: [{ id: "desktop-linux-x64", kind: "desktop", host: "linux-x64-gnu", filename: "magnitude.deb", bytes: oldBytes.length, sha256: sha256(oldBytes) }],
  } })
  const current = yield* Schema.decodeUnknown(Schema.parseJson(ArtifactInput))(baseManifest)
  const previous = yield* Schema.decodeUnknown(Schema.parseJson(ArtifactInput))(baseline)
  const privateAuthority = "private fixture authority bytes"
  const fixturePackage = "isolated updater candidate"
  const manifest = yield* Schema.encode(Schema.parseJson(ArtifactInput))(ArtifactInput.make({ ...current, updateAcceptance: Option.some({
    sourceDigest: sha256("source"), configuration: yield* Schema.decodeUnknown(UpdateConfiguration)({ origin: "https://127.0.0.1:23456", acceptance: true, keyId: "fixture", publicKey: "fixture public key",
      artifactDelivery: { _tag: "PrivateAcceptance", origin: "https://127.0.0.1:23456" } }),
    authority: { sha256: sha256(privateAuthority), bytes: privateAuthority.length }, previous: previous.release,
    candidate: { ...current.release, artifacts: [{ ...current.release.artifacts[0]!, sha256: sha256(fixturePackage), bytes: fixturePackage.length }] },
  }) }))
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "worker-cache-test", owner: "owner", trust: "developer",
    input: { kind: "artifacts", digest: sha256(manifest) }, updateFrom: { kind: "artifacts", digest: sha256(baseline) }, selection: { kind: "profile", profile: "quick", target: "ubuntu-24.04-x64-cpu-intel" }, mode: "verify", allowSpark: false,
    limits: { concurrency: 1, deadlineMinutes: 60, budgetUsd: 10, idleMinutes: 15 } })
  // Exercise the worker boundary with an already-admitted graph, independently of current admission policy.
  const plan = { ...yield* planRun({ ...request, updateFrom: Option.none() }), request }
  const invocation = WorkerInvocation.make({ schemaVersion: 1, disposable: true, port: 11279, model: "fixture", assignment: { plan, work: TestWork.make({ kind: "test", id: WorkId.make(`test:${plan.targets[0]!.target.id}`), target: plan.targets[0]!, producer: Option.none() }), input: plan.request.input, target: plan.targets[0]!,
    claim: { runId: RunId.make("run-00000000-0000-0000-0000-000000000001"), targetId: plan.targets[0]!.target.id, workId: WorkId.make(`test:${plan.targets[0]!.target.id}`), fence: Fence.make(1), worker: "fixture" }, deadline: DateTime.unsafeMake(Date.now() + 60000) } })
  let authorized = true, manifestReads = 0, checks = 0, baselineAllowed = true
  const tickets = Layer.succeed(WorkerTickets, { issue: () => Effect.dieMessage("Not used"), withAuthority: () => Effect.dieMessage("Not used"), revoke: () => Effect.void, authorize: () => Effect.suspend(() => {
    checks++
    return authorized ? Effect.succeed(invocation) : Effect.fail(new WorkerAccessDenied({}))
  }) })
  const inputs = Layer.succeed(InputRegistry, { require: (_owner, input) => input.kind === "artifacts" && !baselineAllowed ? Effect.fail(new InputDenied({})) : Effect.void, register: () => Effect.void, missing: () => Effect.succeed([]), upload: () => Effect.void,
    read: (_owner, key) => Effect.gen(function* () {
      if (key === request.input.digest) {
        manifestReads++
        if (transient && manifestReads === 1) return yield* new InfrastructureFailure({ operation: "fixture-read", message: "Transient manifest read failure" })
        yield* Effect.yieldNow()
        return Stream.make(new TextEncoder().encode(manifest))
      }
      if (key === sha256(privateAuthority)) return Stream.make(new TextEncoder().encode(privateAuthority))
      if (key === sha256(fixturePackage)) return Stream.make(new TextEncoder().encode(fixturePackage))
      return Stream.make(key === sha256(baseline) ? new TextEncoder().encode(baseline) : key === sha256(oldBytes) ? new TextEncoder().encode(oldBytes) : content)
    }) })
  yield* Effect.gen(function* () {
    const service = yield* WorkerInputs
    const token = Redacted.make("fixture-token")
    if (transient) expect((yield* service.read(token, digest).pipe(Effect.either))._tag).toBe("Left")
    yield* Effect.forEach(Array.from({ length: 100 }), () => service.read(token, digest).pipe(Effect.flatMap(Stream.runCollect)), { concurrency: 16 })
    expect(manifestReads).toBe(transient ? 2 : 1)
    expect(checks).toBeGreaterThanOrEqual(200)
    expect(Buffer.concat(Array.from(yield* service.read(token, sha256(baseline)).pipe(Effect.flatMap(Stream.runCollect)))).toString()).toBe(baseline)
    expect(Buffer.concat(Array.from(yield* service.read(token, sha256(oldBytes)).pipe(Effect.flatMap(Stream.runCollect)))).toString()).toBe(oldBytes)
    expect((yield* service.read(token, sha256("unrelated package")).pipe(Effect.either))._tag).toBe("Left")
    expect(Buffer.concat(Array.from(yield* service.read(token, sha256(privateAuthority)).pipe(Effect.flatMap(Stream.runCollect)))).toString()).toBe(privateAuthority)
    expect(Buffer.concat(Array.from(yield* service.read(token, sha256(fixturePackage)).pipe(Effect.flatMap(Stream.runCollect)))).toString()).toBe(fixturePackage)
    baselineAllowed = false
    expect((yield* service.read(token, sha256(oldBytes)).pipe(Effect.either))._tag).toBe("Left")
    authorized = false
    expect((yield* service.read(token, digest).pipe(Effect.either))._tag).toBe("Left")
    expect((yield* service.read(token, sha256(privateAuthority)).pipe(Effect.either))._tag).toBe("Left")
    expect(manifestReads).toBe(transient ? 2 : 1)
  }).pipe(Effect.provide(WorkerInputsLive.pipe(Layer.provide(Layer.merge(tickets, inputs)))))
})))
