import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, DateTime, Effect, Layer, Option, Schema, Stream } from "effect"
import { join } from "node:path"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { snapshotArtifacts } from "../src/artifact-input"
import { runCandidateWorker } from "../src/candidate-worker"
import { planRun } from "../src/catalog"
import { RunId, RunRequest } from "../src/domain"
import { HostInspectorLive } from "../src/host-inspector"
import { nativeInstaller } from "../src/installer"
import { Fence } from "../src/lease"
import { ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { TargetResult, WorkAssignment } from "../src/work-store"
import { configuredHarnessTools } from "../src/harnesses/suite"

// Package/install/CLI smoke with optional baseline qualification; not a native updater or full target qualification.
BunRuntime.runMain(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const root = yield* Config.string("LAB_WORKER_ROOT")
  const target = yield* Config.string("LAB_WORKER_TARGET")
  const manifest = yield* Config.string("LAB_WORKER_MANIFEST")
  const disposable = yield* Config.boolean("LAB_WORKER_DISPOSABLE").pipe(Config.withDefault(false))
  const harnesses = yield* Config.boolean("LAB_WORKER_HARNESSES").pipe(Config.withDefault(false))
  const generation = (yield* Config.boolean("LAB_WORKER_GENERATION").pipe(Config.withDefault(false))) || harnesses
  const objects = join(root, "objects")
  const input = yield* snapshotArtifacts(manifest, objects)
  const baselinePath = yield* Config.option(Config.string("LAB_WORKER_BASELINE"))
  const baseline = yield* Option.match(baselinePath, { onNone: () => Effect.succeed(Option.none<Effect.Effect.Success<ReturnType<typeof snapshotArtifacts>>>()),
    onSome: path => snapshotArtifacts(path, objects).pipe(Effect.map(Option.some)) })
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: crypto.randomUUID(), owner: "local-worker-probe",
    input: { kind: "artifacts", digest: input.digest }, ...(Option.isSome(baseline) ? { updateFrom: { kind: "artifacts", digest: baseline.value.digest } } : {}),
    selection: Option.isSome(baseline) || generation ? { kind: "custom", targets: [target], suites: ["package", "install", "app", "endpoint", "harness", "cli", "update", "uninstall"], harnesses: harnesses ? ["pi", "opencode", "hermes"] : ["pi"] } : { kind: "profile", profile: "quick", target },
    mode: "verify", trust: "developer", allowSpark: false, limits: { concurrency: 1, deadlineMinutes: generation ? 45 : 15, budgetUsd: 25, idleMinutes: 15 } })
  const original = yield* planRun(request)
  const ids = Option.isSome(baseline) ? ["P1", "P2", "P4", "I1", "I2", "U1", "C1", "X1"]
    : generation ? ["P1", "P2", "P4", "I1", "I2", "I3", "A1", "A2", "A3", ...(harnesses ? ["A5"] : []), "E1", "E6", "E2", "E3", "E4", ...(harnesses ? ["H1", "H2", "H5", "C4"] : []), "C1", "X1"] : ["P1", "P2", "P4", "I1", "I2", "C1"]
  const selected = { ...original.targets[0]!, cases: ids.flatMap(id => original.targets[0]!.cases.filter(c => c.id === id)) }
  const plan = { ...original, targets: [selected] }
  const assignment = WorkAssignment.make({ claim: { runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: selected.target.id, fence: Fence.make(1), worker: "local-worker-probe" },
    plan, target: selected, deadline: DateTime.unsafeMake(Date.now() + (generation ? 45 : 15) * 60_000) })
  yield* fs.writeFileString(join(root, "assignment.json"), yield* Schema.encode(Schema.parseJson(WorkAssignment))(assignment), { flag: "wx", mode: 0o600 })
  const environment = Object.fromEntries(["PATH", "TMPDIR", "USER", "LOGNAME", "SystemRoot", "TEMP", "APPDATA", "LOCALAPPDATA"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const result = yield* Effect.gen(function* () {
    const store = yield* ArtifactStore
    yield* store.put(input.digest, Stream.make(new TextEncoder().encode(input.json)))
    if (Option.isSome(baseline)) yield* store.put(baseline.value.digest, Stream.make(new TextEncoder().encode(baseline.value.json)))
    return yield* runCandidateWorker(assignment, { root: join(root, "attempt"), port: 11279, model: "qwen3.5-4b:gguf:q4", environment })
  }).pipe(Effect.provide([fileArtifactStore(objects), HostInspectorLive, configuredHarnessTools, nativeInstaller({ disposable, root: join(root, "installation"), environment })]))
  yield* fs.writeFileString(join(root, "result.json"), yield* Schema.encode(Schema.parseJson(TargetResult))(result))
  if (result.cases.some(c => c.outcome.status !== "passed") || result.cleanupErrors.length) process.exitCode = 1
}).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
