import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { DateTime, Deferred, Effect, Fiber, Layer, Schema, Stream } from "effect"
import { join } from "node:path"
import releasePlan from "../../release/release-plan.json"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { ArtifactInput } from "../src/inputs"
import { runCandidateWorker } from "../src/candidate-worker"
import { planRun } from "../src/catalog"
import { AssertionFailure, InfrastructureFailure, RunId, RunRequest } from "../src/domain"
import { Fence } from "../src/lease"
import { HostInspector } from "../src/host-inspector"
import { Installer } from "../src/installer"
import { ProcessExecutor } from "../src/process"
import { SourceBuilder } from "../src/source-builder"
import { sha256 } from "../src/snapshot"
import { WorkAssignment, validateTargetResult } from "../src/work-store"

for (const mode of ["success", "wrong-version", "corrupt", "cleanup-failure", "cancel", "defect", "source-success", "source-compile-failure", "source-package-failure"] as const) test(`artifact worker preserves case results and cleanup for ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-artifact-worker-" })
  const bytes = new TextEncoder().encode("fixture installer")
  const artifactJson = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "artifacts", release: {
    schemaVersion: 2, version: "0.1.3", acnRevision: 1, rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40),
    artifacts: [{ id: "desktop-darwin-arm64", kind: "desktop", host: "darwin-arm64", filename: "Magnitude.dmg", bytes: bytes.length, sha256: sha256(bytes) }],
  } })
  const artifactInput = yield* Schema.decodeUnknown(Schema.parseJson(ArtifactInput))(artifactJson)
  const isSource = mode.startsWith("source-")
  const json = isSource ? yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "source", commit: "a".repeat(40), entries: [] }) : artifactJson
  let compiled = 0, packaged = 0
  const sourceFailed = mode === "source-compile-failure" || mode === "source-package-failure"
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "worker-artifact-test", owner: "developer",
    input: { kind: isSource ? "source" : "artifacts", digest: sha256(json) }, selection: { kind: "profile", profile: "quick", target: "macos-15-arm64-metal-apple-silicon" },
    mode: "verify", trust: "developer", allowSpark: false, limits: { concurrency: 1, deadlineMinutes: 60, budgetUsd: 100, idleMinutes: 15 } })
  const plan = yield* planRun(request)
  const selected = { ...plan.targets[0]!, cases: plan.targets[0]!.cases.filter(c => ["P1", "P2", "P5", "I1", "I2", "C1"].includes(c.id)) }
  const assignment = WorkAssignment.make({ claim: { runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: selected.target.id, fence: Fence.make(1), worker: "fixture" },
    plan, target: selected, deadline: DateTime.unsafeMake(Date.now() + 60_000) })
  const started = yield* Deferred.make<void>()
  let installed = 0, removed = 0
  let nativeState: string | undefined
  const program = Effect.gen(function* () {
    const objects = yield* ArtifactStore
    yield* objects.put(sha256(json), Stream.make(new TextEncoder().encode(json)))
    yield* objects.put(sha256(bytes), Stream.make(bytes))
    if (mode === "corrupt") yield* fs.writeFileString(join(root, "objects", sha256(bytes)), "changed bytes")
    const execution = runCandidateWorker(assignment, { root: join(root, "attempt"), port: 11279, model: "fixture", environment: {} })
    if (mode === "cancel") {
      const fiber = yield* Effect.fork(execution)
      yield* Deferred.await(started)
      yield* Fiber.interrupt(fiber)
      expect(installed).toBe(1)
      expect(removed).toBe(1)
      return
    }
    const result = yield* execution
    yield* validateTargetResult(selected, result)
    expect(result.cases.find(c => c.caseId === "P5")!.outcome.status).toBe("blocked")
    expect(result.cases.find(c => c.caseId === "C1")!.outcome.status).toBe((mode === "corrupt" || sourceFailed) ? "blocked" : (mode === "wrong-version" || mode === "defect") ? "failed" : "passed")
    if (mode === "wrong-version" || mode === "defect") {
      const failed = result.cases.find(c => c.caseId === "C1")!
      expect(failed.evidence.some(e => e.path.endsWith("C1-shared-failure.json"))).toBe(true)
    }
    expect(installed).toBe((mode === "corrupt" || sourceFailed) ? 0 : 1)
    expect(removed).toBe(installed)
    if (isSource) {
      const compileOutcome = result.cases.find(c => c.caseId === "P1")!.outcome
      if (mode === "source-compile-failure" && compileOutcome.status === "failed") expect(compileOutcome.category).toBe("build")
      expect(compiled).toBe(1)
      expect(packaged).toBe(mode === "source-compile-failure" ? 0 : 1)
      expect(result.cases.find(c => c.caseId === "P1")!.outcome.status).toBe(mode === "source-compile-failure" ? "failed" : "passed")
      expect(result.cases.find(c => c.caseId === "P2")!.outcome.status).toBe(mode === "source-compile-failure" ? "blocked" : mode === "source-package-failure" ? "failed" : "passed")
    }
    if (nativeState && process.platform !== "win32") {
      expect(Buffer.byteLength(join(nativeState, "application.sock"))).toBeLessThanOrEqual(103)
      expect(yield* fs.exists(nativeState)).toBe(false)
    }
    expect(result.cleanupErrors.length).toBe(mode === "cleanup-failure" ? 1 : 0)
    if (installed > 0 && mode !== "defect") expect(result.cases.find(c => c.caseId === "C1")!.evidence.some(item => item.path.startsWith("evidence/cli/"))).toBe(true)
    for (const item of result.cases.flatMap(c => c.evidence)) expect(yield* objects.exists(item.sha256)).toBe(true)
  })
  yield* program.pipe(Effect.provide([
    fileArtifactStore(join(root, "objects")),
    Layer.succeed(SourceBuilder, { prepare: () => Effect.gen(function* () {
      const compile = yield* Effect.cached(Effect.suspend(() => { compiled++; return mode === "source-compile-failure" ? Effect.fail(new AssertionFailure({ message: "Compiler fixture failed" })) : Effect.succeed({ detail: "Compiled fixture", evidence: [] }) }))
      const packageStage = yield* Effect.cached(Effect.suspend(() => { packaged++; return mode === "source-package-failure" ? Effect.fail(new AssertionFailure({ message: "Packager fixture failed" })) : Effect.succeed({ input: artifactInput, evidence: [] }) }))
      return { compile, package: packageStage, evidence: () => [] }
    }) }),
    Layer.succeed(HostInspector, { inspect: () => Effect.succeed({ os: "macos", version: "15.5", build: "fixture", arch: "arm64", cpuVendor: "Apple", cpuName: "fixture", machineModel: "fixture", memoryBytes: 1024, gpus: [] }) }),
    Layer.succeed(Installer, { install: candidate => Effect.sync(() => { installed++; return { candidate, root: "fixture", executable: "fixture", cli: "fixture", packageVersion: "0.1.3" } }),
      uninstall: () => Effect.gen(function* () { removed++; if (mode === "cleanup-failure") return yield* new InfrastructureFailure({ operation: "fixture-cleanup", message: "Failed cleanup fixture" }) }) }),
    Layer.succeed(ProcessExecutor, { run: spec => { nativeState = spec.env.MAGNITUDE_DESKTOP_STATE_DIR; return mode === "cancel" ? Deferred.succeed(started, undefined).pipe(Effect.zipRight(Effect.never))
      : mode === "defect" ? Effect.dieMessage("Injected command defect") : Effect.succeed({ exitCode: 0, stdout: mode === "wrong-version" ? "0.0.0" : "0.1.3", stderr: "" }) } }),
  ]))
})).pipe(Effect.provide(BunContext.layer))))
