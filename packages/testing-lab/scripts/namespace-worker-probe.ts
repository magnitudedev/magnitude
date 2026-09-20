import { Option } from "effect"
import { TestWork } from "../src/execution-plan"
import { WorkId } from "../src/work-identity"
import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Context, DateTime, Effect, Layer, Schema, Stream } from "effect"
import { join, resolve } from "node:path"
import { snapshotArtifacts } from "../src/artifact-input"
import { fileArtifactStore } from "../src/artifact-store"
import { planRun } from "../src/catalog"
import { initializeDatabase } from "../src/database"
import { InfrastructureFailure, RunRequest } from "../src/domain"
import { InputRegistry, InputRegistryLive } from "../src/inputs"
import { Fence } from "../src/lease"
import { MachineAllocator, WorkerTransport } from "../src/machines"
import { namespaceAllocator, namespaceTransport } from "../src/providers/namespace"
import { ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { WorkerRunner } from "../src/scheduler"
import { TargetResult, WorkAssignment } from "../src/work-store"
import { transportWorkerRunner, WorkerTransports } from "../src/worker-runner"
import { temporaryDatabase } from "../test/postgres"

// Exercise transport on an explicitly selected, already tagged disposable lab machine.
// This probe takes ownership of cleanup; it never adopts an unrelated Namespace devbox.
BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const root = resolve(yield* Config.string("LAB_WORKER_ROOT"))
  const manifest = yield* Config.string("LAB_WORKER_MANIFEST")
  const target = yield* Config.string("LAB_WORKER_TARGET")
  const name = yield* Config.string("LAB_NAMESPACE_NAME")
  const cli = yield* Config.string("LAB_NAMESPACE_CLI")
  const executable = yield* Config.string("LAB_GUEST_BUN")
  const entry = yield* Config.string("LAB_GUEST_ENTRY")
  const allocator = Context.get(yield* Layer.build(namespaceAllocator(cli, [])), MachineAllocator)
  const selected = (yield* allocator.inventory()).find(machine => machine.provider === "namespace" && machine.name === name)
  if (!selected || selected.provider !== "namespace") return yield* new InfrastructureFailure({ operation: "namespace-probe", message: "Explicit tagged lab machine not found" })
  const machine = yield* Effect.acquireRelease(Effect.succeed(selected), machine => allocator.release(machine).pipe(Effect.orDie))
  const transport = Context.get(yield* Layer.build(namespaceTransport(cli)), WorkerTransport)
  const localObjects = join(root, "upload-objects")
  const snapshot = yield* snapshotArtifacts(manifest, localObjects)
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: crypto.randomUUID(), owner: "namespace-native-probe",
    input: { kind: "artifacts", digest: snapshot.digest }, selection: { kind: "profile", profile: "quick", target }, mode: "verify", trust: "developer", allowSpark: false,
    limits: { concurrency: 1, deadlineMinutes: 15, budgetUsd: 25, idleMinutes: 15 } })
  const plan = yield* planRun(request)
  if (plan.targets[0]!.target.provider !== "namespace") return yield* new InfrastructureFailure({ operation: "namespace-probe", message: "Probe requires a Namespace target" })
  const assignment = WorkAssignment.make({ claim: { runId: machine.tags.runId, targetId: plan.targets[0]!.target.id, workId: WorkId.make(`test:${plan.targets[0]!.target.id}`), fence: Fence.make(1), worker: "namespace-native-probe" },
    plan, work: TestWork.make({ kind: "test", id: WorkId.make(`test:${plan.targets[0]!.target.id}`), target: plan.targets[0]!, producer: Option.none() }), input: plan.request.input, target: plan.targets[0]!, deadline: DateTime.unsafeMake(Math.min(Date.now() + 15 * 60_000, DateTime.toEpochMillis(machine.tags.expiresAt))) })
  const database = yield* temporaryDatabase
  const objects = fileArtifactStore(join(root, "coordinator-objects"))
  const inputs = InputRegistryLive.pipe(Layer.provide(Layer.merge(database, objects)))
  const runner = transportWorkerRunner([{ provider: "namespace", artifactHost: plan.targets[0]!.target.artifactHost, executable,
    args: [entry], root: "/Users/runner/lab/worker", disposable: true, port: 11279, model: "qwen3.5-4b:gguf:q4" }]).pipe(
    Layer.provide(Layer.succeed(WorkerTransports, { transports: new Map([["namespace" as const, transport]]) })))
  yield* Effect.gen(function* () {
    yield* initializeDatabase
    const registry = yield* InputRegistry
    for (const digest of snapshot.digests) yield* registry.upload(request.owner, digest, fs.stream(join(localObjects, digest)).pipe(Stream.mapError(e => new InfrastructureFailure({ operation: "probe-upload", message: e.message }))))
    yield* registry.upload(request.owner, snapshot.digest, Stream.make(new TextEncoder().encode(snapshot.json)))
    yield* registry.register(request.owner, request.input)
    const result = yield* Effect.flatMap(WorkerRunner, worker => worker.run(machine, assignment)).pipe(Effect.provide(runner))
    yield* fs.writeFileString(join(root, "result.json"), yield* Schema.encode(Schema.parseJson(TargetResult))(result))
    if (result.cases.some(c => c.outcome.status !== "passed") || result.cleanupErrors.length) process.exitCode = 1
  }).pipe(Effect.provide(Layer.merge(database, inputs)))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
