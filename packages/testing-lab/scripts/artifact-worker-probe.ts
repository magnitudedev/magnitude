import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, DateTime, Effect, Layer, Schema, Stream } from "effect"
import { join } from "node:path"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { snapshotArtifacts } from "../src/artifact-input"
import { runArtifactWorker } from "../src/artifact-worker"
import { planRun } from "../src/catalog"
import { RunId, RunRequest } from "../src/domain"
import { HostInspectorLive } from "../src/host-inspector"
import { nativeInstaller } from "../src/installer"
import { Fence } from "../src/lease"
import { ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { TargetResult, WorkAssignment } from "../src/work-store"

// Deliberately limited package/install/CLI smoke; not full target qualification.
BunRuntime.runMain(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const root = yield* Config.string("LAB_WORKER_ROOT")
  const target = yield* Config.string("LAB_WORKER_TARGET")
  const manifest = yield* Config.string("LAB_WORKER_MANIFEST")
  const disposable = yield* Config.boolean("LAB_WORKER_DISPOSABLE").pipe(Config.withDefault(false))
  const objects = join(root, "objects")
  const input = yield* snapshotArtifacts(manifest, objects)
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: crypto.randomUUID(), owner: "local-worker-probe",
    input: { kind: "artifacts", digest: input.digest }, selection: { kind: "profile", profile: "quick", target },
    mode: "verify", trust: "developer", allowSpark: false, limits: { concurrency: 1, deadlineMinutes: 15, budgetUsd: 25, idleMinutes: 15 } })
  const original = yield* planRun(request)
  const selected = { ...original.targets[0]!, cases: original.targets[0]!.cases.filter(c => ["P1", "P2", "I1", "I2", "C1"].includes(c.id)) }
  const plan = { ...original, targets: [selected] }
  const assignment = WorkAssignment.make({ claim: { runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: selected.target.id, fence: Fence.make(1), worker: "local-worker-probe" },
    plan, target: selected, deadline: DateTime.unsafeMake(Date.now() + 15 * 60_000) })
  const environment = Object.fromEntries(["PATH", "TMPDIR", "USER", "LOGNAME", "SystemRoot", "TEMP", "APPDATA", "LOCALAPPDATA"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const result = yield* Effect.gen(function* () {
    const store = yield* ArtifactStore
    yield* store.put(input.digest, Stream.make(new TextEncoder().encode(input.json)))
    return yield* runArtifactWorker(assignment, { root: join(root, "attempt"), port: 11279, model: "qwen3.5-4b:gguf:q4", environment })
  }).pipe(Effect.provide([fileArtifactStore(objects), HostInspectorLive, nativeInstaller({ disposable, root: join(root, "installation"), environment })]))
  yield* fs.writeFileString(join(root, "result.json"), yield* Schema.encode(Schema.parseJson(TargetResult))(result))
  if (result.cases.some(c => c.outcome.status !== "passed") || result.cleanupErrors.length) process.exitCode = 1
}).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
