import { WorkId } from "../src/work-identity"
import { Option } from "effect"
import { TestWork } from "../src/execution-plan"
import releasePlan from "../../release/release-plan.json"
import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { DateTime, Effect, Layer, Schema, Stream } from "effect"
import { dirname, join } from "node:path"
import { fileArtifactStore } from "../src/artifact-store"
import { planRun } from "../src/catalog"
import { initializeDatabase } from "../src/database"
import { InfrastructureFailure, LeaseId, RunId, RunRequest } from "../src/domain"
import { InputRegistry, InputRegistryLive } from "../src/inputs"
import { Fence } from "../src/lease"
import { LocalMachine, type WorkerTransport } from "../src/machines"
import { command, ProcessExecutor, ProcessExecutorLive } from "../src/process"
import { WorkerRunner } from "../src/scheduler"
import { sha256 } from "../src/snapshot"
import { WorkAssignment } from "../src/work-store"
import { WorkerInvocation, WorkerReply } from "../src/worker-protocol"
import { transportWorkerRunner, WorkerTransports } from "../src/worker-runner"
import { temporaryDatabase } from "./postgres"
import { WorkerDiagnostic } from "../src/worker-diagnostics"

for (const mode of ["success", "wrong-claim", "missing-case", "corrupt-evidence", "foreign-owner", "untrusted-local", "shared-disposable", "native-exit", "unpack-exit", "invalid-protocol"] as const) {
  test(`transported workers reject unauthorized or mismatched results: ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const processes = yield* ProcessExecutor
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-worker-runner-" })
    const database = yield* temporaryDatabase
    const objectLayer = fileArtifactStore(join(root, "coordinator-objects"))
    const registry = InputRegistryLive.pipe(Layer.provide(Layer.merge(database, objectLayer)))
    yield* Effect.gen(function* () {
      yield* initializeDatabase
      const inputs = yield* InputRegistry
      const bytes = "owned input file"
      const wire = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "source", commit: "a".repeat(40),
        entries: [{ kind: "file", path: "test.txt", sha256: sha256(bytes), bytes: bytes.length, executable: false }] })
      const oldBytes = "old native package"
      const baseline = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "artifacts", release: {
        schemaVersion: 2, version: "0.1.2", acnRevision: 1, rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.2", sourceCommit: "b".repeat(40),
        artifacts: [{ id: "desktop-darwin-arm64", kind: "desktop", host: "darwin-arm64", filename: "Magnitude.dmg", bytes: oldBytes.length, sha256: sha256(oldBytes) }],
      } })
      const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, owner: "developer", trust: mode === "untrusted-local" ? "untrusted-ci" : "developer", idempotencyKey: `transport-${mode}`,
        input: { kind: "source", digest: sha256(wire) }, updateFrom: { kind: "artifacts", digest: sha256(baseline) }, selection: { kind: "profile", profile: "quick", target: "macos-15-arm64-cpu-apple-silicon" }, mode: "verify", allowSpark: false,
        limits: { concurrency: 1, deadlineMinutes: 5, budgetUsd: 25, idleMinutes: 15 } })
      yield* inputs.upload(request.owner, sha256(bytes), Stream.make(new TextEncoder().encode(bytes)))
      yield* inputs.upload(request.owner, sha256(wire), Stream.make(new TextEncoder().encode(wire)))
      yield* inputs.register(request.owner, request.input)
      yield* inputs.upload(request.owner, sha256(oldBytes), Stream.make(new TextEncoder().encode(oldBytes)))
      yield* inputs.upload(request.owner, sha256(baseline), Stream.make(new TextEncoder().encode(baseline)))
      yield* inputs.register(request.owner, { kind: "artifacts", digest: sha256(baseline) })
      // Transport can still consume an already-admitted graph; new admission separately
      // rejects historical updates while their complete journey is unavailable.
      const current = { ...request, updateFrom: Option.none() }
      const admitted = yield* planRun(mode === "foreign-owner" ? { ...current, owner: RunRequest.fields.owner.make("someone-else") } : current)
      const plan = { ...admitted, request: { ...admitted.request, updateFrom: request.updateFrom } }
      const assignment = WorkAssignment.make({ claim: { runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: plan.targets[0]!.target.id, workId: WorkId.make(`test:${plan.targets[0]!.target.id}`), fence: Fence.make(1), worker: "transport-fixture" },
        plan, work: TestWork.make({ kind: "test", id: WorkId.make(`test:${plan.targets[0]!.target.id}`), target: plan.targets[0]!, producer: Option.none() }), input: plan.request.input, target: plan.targets[0]!, deadline: DateTime.unsafeMake(Date.now() + 60_000) })
      const machine = LocalMachine.make({ provider: "local", root: join(root, "guest"), tags: { schemaVersion: 1, runId: assignment.claim.runId,
        leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), expiresAt: assignment.deadline } })
      let executions = 0, uploads = 0
      const evidence = "fixture evidence"
      const transport: WorkerTransport = {
        upload: (_machine, local, remote) => Effect.gen(function* () { uploads++; yield* fs.makeDirectory(dirname(remote), { recursive: true }); yield* fs.copyFile(local, remote) })
          .pipe(Effect.mapError(e => new InfrastructureFailure({ operation: "fixture-upload", message: e.message }))),
        download: (_machine, remote, local) => fs.copyFile(remote, local).pipe(Effect.mapError(e => new InfrastructureFailure({ operation: "fixture-download", message: e.message }))),
        execute: (_machine, _executable, args) => Effect.gen(function* () {
          if (_executable === "/usr/bin/tar") return mode === "unpack-exit" ? { exitCode: 2, stdout: "", stderr: "input extraction failed" }
            : yield* command(_executable, args).pipe(Effect.provideService(ProcessExecutor, processes))
          executions++
          const invocation = yield* fs.readFileString(args[0]!).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(WorkerInvocation))))
          expect(invocation.assignment.claim).toEqual(assignment.claim)
          expect(yield* fs.readFileString(join(dirname(args[0]!), "objects", sha256(baseline)))).toBe(baseline)
          expect(yield* fs.readFileString(join(dirname(args[0]!), "objects", sha256(oldBytes)))).toBe(oldBytes)
          expect(yield* fs.readFileString(join(dirname(args[0]!), "objects", sha256(bytes)))).toBe(bytes)
          if (mode === "invalid-protocol") return { exitCode: 0, stdout: "unexpected browser logging token=private-token", stderr: "" }
          if (mode === "native-exit") return { exitCode: 42, stdout: "worker started", stderr: "native startup failed https://example.test/file?sig=private-capability token=private-token" }
          const output = mode === "corrupt-evidence" ? "corrupt evidence" : evidence
          yield* fs.writeFileString(join(dirname(args[0]!), "objects", sha256(evidence)), output)
          const now = new Date().toISOString()
          const result = { output: Option.none(), cleanupErrors: [], cases: assignment.target.cases.map(c => ({ targetId: assignment.claim.targetId, caseId: c.id, harness: c.harness,
            startedAt: now, endedAt: now, outcome: { status: "passed" as const, detail: "Transport fixture, not product acceptance" },
            evidence: [{ path: "fixture.txt", sha256: sha256(evidence), bytes: evidence.length }] })) }
          if (mode === "missing-case") result.cases.pop()
          const reply = { schemaVersion: 1 as const, claim: mode === "wrong-claim" ? { ...assignment.claim, fence: Fence.make(2) } : assignment.claim, result }
          return { exitCode: 0, stdout: yield* Schema.encode(Schema.parseJson(WorkerReply))(reply), stderr: "" }
        }).pipe(Effect.mapError(e => new InfrastructureFailure({ operation: "fixture-execute", message: e.message }))),
      }
      const result = yield* Effect.flatMap(WorkerRunner, runner => runner.run(machine, assignment)).pipe(Effect.either,
        Effect.provide(transportWorkerRunner([{ provider: "local", artifactHost: "darwin-arm64", executable: "fixture", args: [], root: root,
          disposable: mode === "shared-disposable", port: 11279, model: "fixture" }]).pipe(Layer.provide(Layer.succeed(WorkerTransports, { transports: new Map([["local" as const, transport]]) })))))
      expect(result._tag).toBe(mode === "success" ? "Right" : "Left")
      expect(executions).toBe(["foreign-owner", "untrusted-local", "shared-disposable", "unpack-exit"].includes(mode) ? 0 : 1)
      if (["foreign-owner", "untrusted-local", "shared-disposable"].includes(mode)) expect(uploads).toBe(0)
      else expect(uploads).toBe(1)
      expect(yield* inputs.missing(request.owner, [sha256(evidence)])).toEqual(mode === "success" ? [] : [sha256(evidence)])
      if (mode === "native-exit" || mode === "unpack-exit" || mode === "invalid-protocol") {
        expect(result._tag).toBe("Left")
        if (result._tag !== "Left") throw new Error("Expected native execution failure")
        const items = Option.getOrThrow(result.left.evidence)
        expect(items).toHaveLength(1)
        // The transport's scoped staging directory is gone when run returns.
        // Diagnostics must survive in coordinator storage and be owner-readable.
        const chunks = yield* (yield* inputs.read(request.owner, items[0]!.sha256)).pipe(Stream.runCollect)
        const wire = Buffer.concat(Array.from(chunks)).toString("utf8")
        expect(sha256(wire)).toBe(items[0]!.sha256)
        expect(Buffer.byteLength(wire)).toBe(items[0]!.bytes)
        const diagnostic = yield* Schema.decodeUnknown(Schema.parseJson(WorkerDiagnostic))(wire)
        expect(diagnostic.runId).toBe(assignment.claim.runId)
        expect(diagnostic.output).toContain(mode === "native-exit" ? "native startup failed" : mode === "invalid-protocol" ? "unexpected browser logging" : "input extraction failed")
        expect(diagnostic.output).not.toContain("private-capability")
        expect(diagnostic.output).not.toContain("private-token")
      }
    }).pipe(Effect.provide(Layer.mergeAll(database, registry, objectLayer)))
  })).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))), 30_000)
}
