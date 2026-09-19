import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { join, resolve } from "node:path"
import { snapshotArtifacts } from "../src/artifact-input"
import { fileArtifactStore } from "../src/artifact-store"
import { planRun } from "../src/catalog"
import { initializeDatabase } from "../src/database"
import { InfrastructureFailure, RunRequest, RunResult } from "../src/domain"
import { InputRegistry, InputRegistryLive } from "../src/inputs"
import { LeaseStoreLive } from "../src/lease-store"
import { MachineAllocator, WorkerTransport } from "../src/machines"
import { localAllocator, localTransport } from "../src/providers/local"
import { ProcessExecutorLive } from "../src/process"
import { RunStore, runStoreLayer } from "../src/run-store"
import { assertRuntime } from "../src/runtime"
import { MachineProviders, Scheduler, schedulerLayer } from "../src/scheduler"
import { WorkStoreLive } from "../src/work-store"
import { transportWorkerRunner, WorkerTransports } from "../src/worker-runner"
import { temporaryDatabase } from "../test/postgres"

// Actual scheduler, PostgreSQL, subprocess and installer. Preserve the entire quick profile, including blocked cases.
BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const root = yield* Config.string("LAB_WORKER_ROOT").pipe(Effect.map(resolve))
  const manifest = yield* Config.string("LAB_WORKER_MANIFEST")
  const target = yield* Config.string("LAB_WORKER_TARGET")
  const localObjects = join(root, "upload-objects")
  const snapshot = yield* snapshotArtifacts(manifest, localObjects)
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: crypto.randomUUID(), owner: "scheduler-native-probe",
    input: { kind: "artifacts", digest: snapshot.digest }, selection: { kind: "profile", profile: "quick", target }, mode: "verify", trust: "developer", allowSpark: false,
    limits: { concurrency: 1, deadlineMinutes: 15, budgetUsd: 25, idleMinutes: 15 } })
  const original = yield* planRun(request)
  const selected = { ...original.targets[0]!, target: { ...original.targets[0]!.target, provider: "local" as const } }
  const plan = { ...original, targets: [selected] }
  const database = yield* temporaryDatabase
  const objects = fileArtifactStore(join(root, "coordinator-objects"))
  const inputs = InputRegistryLive.pipe(Layer.provide(Layer.merge(database, objects)))
  const stores = Layer.mergeAll(database, inputs, runStoreLayer(100).pipe(Layer.provide(database)), WorkStoreLive.pipe(Layer.provide(database)), LeaseStoreLive.pipe(Layer.provide(database)))
  const allocator = Context.get(yield* Layer.build(localAllocator(join(root, "workers"))), MachineAllocator)
  const transport = Context.get(yield* Layer.build(localTransport), WorkerTransport)
  const runner = transportWorkerRunner([{ provider: "local", artifactHost: selected.target.artifactHost, executable: process.execPath,
    args: [resolve(import.meta.dir, "../src/worker-entry.ts")], root: root, disposable: false, port: 11279, model: "qwen3.5-4b:gguf:q4" }]).pipe(
    Layer.provide(Layer.succeed(WorkerTransports, { transports: new Map([["local" as const, transport]]) })))
  const scheduler = schedulerLayer().pipe(Layer.provide(runner), Layer.provide(Layer.succeed(MachineProviders, { allocators: new Map([["local" as const, allocator]]) })))
  yield* Effect.gen(function* () {
    yield* initializeDatabase
    const registry = yield* InputRegistry
    for (const digest of snapshot.digests) yield* registry.upload(request.owner, digest, fs.stream(join(localObjects, digest)).pipe(Stream.mapError(e => new InfrastructureFailure({ operation: "probe-upload", message: e.message }))))
    yield* registry.upload(request.owner, snapshot.digest, Stream.make(new TextEncoder().encode(snapshot.json)))
    yield* registry.register(request.owner, request.input)
    const runs = yield* RunStore
    const admitted = yield* runs.submit(plan)
    yield* Effect.flatMap(Scheduler, scheduler => scheduler.next("native-local-worker")).pipe(Effect.provide(scheduler))
    const result = Option.getOrThrow(yield* runs.result(admitted.state.runId))
    yield* fs.writeFileString(join(root, "result.json"), yield* Schema.encode(Schema.parseJson(RunResult))(result))
    if ((yield* allocator.inventory()).length !== 0) return yield* new InfrastructureFailure({ operation: "probe-cleanup", message: "Local allocation remains after scheduler completed" })
    if (result.cases.some(c => c.outcome.status !== "passed") || result.cleanupErrors.length) process.exitCode = 1
  }).pipe(Effect.provide(stores))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
