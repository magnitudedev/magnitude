import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { DateTime, Effect, Layer, Option, Schema, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { fileArtifactStore, ArtifactStore } from "../src/artifact-store"
import { runBuildWorker } from "../src/build-worker"
import { planRun, targets } from "../src/catalog"
import { AssertionFailure, RunId, RunRequest } from "../src/domain"
import { planExecution } from "../src/execution-plan"
import { HostInspector } from "../src/host-inspector"
import { ArtifactInput } from "../src/inputs"
import { Fence } from "../src/lease"
import { SourceBuilder } from "../src/source-builder"
import { sha256 } from "../src/snapshot"
import { WorkAssignment } from "../src/work-store"
import releasePlan from "../../release/release-plan.json"

for (const failure of ["none", "compile", "package"] as const) test(`the build-only guest reports ${failure} without an installer or consumer state`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem, root = yield* fs.makeTempDirectoryScoped({ prefix: "build-worker-" })
  const source = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "source", commit: "a".repeat(40), entries: [] })
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, owner: "fixture", idempotencyKey: "build-only-fixture", input: { kind: "source", digest: sha256(source) },
    mode: "verify", trust: "developer", allowSpark: false, selection: { kind: "custom", targets: ["ubuntu-24.04-x64-cuda-a10"], suites: ["install"], harnesses: ["pi"] },
    limits: { concurrency: 1, deadlineMinutes: 60, budgetUsd: 10, idleMinutes: 15 } })
  const plan = yield* planRun(request), work = (yield* planExecution(request, plan.targets, targets))[0]!
  const assignment = WorkAssignment.make({ claim: { runId: RunId.make(`run-${crypto.randomUUID()}`), workId: work.id, targetId: work.target.target.id, worker: "fixture", fence: Fence.make(1) },
    plan, work, input: request.input, target: work.target, deadline: DateTime.unsafeMake(Date.now() + 60000) })
  let compiled = 0, packaged = 0
  const build = Layer.succeed(SourceBuilder, { prepare: (_source, digest, target, backend) => Effect.gen(function* () {
    expect(digest).toBe(request.input.digest); expect(target.backend).toBe("cpu"); expect(backend).toBe("cuda")
    const input = yield* Schema.decodeUnknown(ArtifactInput)({ schemaVersion: 1, kind: "artifacts", release: { schemaVersion: 2, version: "0.1.3", acnRevision: 1,
      rpc: releasePlan.rpc, plugins: [], sourceCommit: "a".repeat(40), tag: "fixture", artifacts: [{ id: "desktop", kind: "desktop", host: "linux-x64-gnu", filename: "magnitude.deb", sha256: sha256("package"), bytes: 7 }] } })
    return { evidence: () => [], compile: Effect.suspend(() => { compiled++; return failure === "compile"
      ? Effect.fail(new AssertionFailure({ message: "Compiler fixture failed" })) : Effect.succeed({ detail: "Compiled fixture", evidence: [] }) }),
      package: Effect.suspend(() => { packaged++; return failure === "package" ? Effect.fail(new AssertionFailure({ message: "Packager fixture failed" }))
        : Effect.succeed({ input, digest: sha256("fixture manifest"), evidence: [] }) }),
    }
  }).pipe(Effect.orDie) })
  yield* Effect.gen(function* () {
    yield* (yield* ArtifactStore).put(request.input.digest, Stream.make(new TextEncoder().encode(source)))
    const result = yield* runBuildWorker(assignment)
    expect(compiled).toBe(1); expect(packaged).toBe(failure === "compile" ? 0 : 1)
    expect(result.cases.map(test => test.outcome.status)).toEqual(failure === "compile" ? ["failed", "blocked"] : failure === "package" ? ["passed", "failed"] : ["passed", "passed"])
    expect(Option.isSome(result.output)).toBe(failure === "none")
    expect(result.cases.every(test => test.evidence.some(item => item.path === "evidence/build-source.json"))).toBe(true)
  }).pipe(Effect.provide([fileArtifactStore(join(root, "objects")), build, Layer.succeed(HostInspector, { inspect: target => Effect.succeed({ os: target.os, version: target.version, arch: target.arch,
    build: "fixture", cpuVendor: "Intel", cpuName: "fixture", machineModel: "fixture", memoryBytes: 1024, gpus: [] }) })]))
})).pipe(Effect.provide(BunContext.layer))))
