import { FetchHttpClient, FileSystem, HttpServer } from "@effect/platform"
import { BunContext, BunHttpServer, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Option, Schema, Stream } from "effect"
import { join } from "node:path"
import { snapshotArtifacts } from "../src/artifact-input"
import { fileArtifactStore } from "../src/artifact-store"
import { planRun } from "../src/catalog"
import { initializeDatabase } from "../src/database"
import { AssertionFailure, RunRequest } from "../src/domain"
import { InputRegistry, InputRegistryLive } from "../src/inputs"
import { runOutwardWorker } from "../src/outward-worker"
import { ProcessExecutorLive } from "../src/process"
import { RunStore, runStoreLayer } from "../src/run-store"
import { assertRuntime } from "../src/runtime"
import { WorkStore, WorkStoreLive } from "../src/work-store"
import { workerApi } from "../src/worker-api"
import { workerClientLayer } from "../src/worker-client"
import { GuestExecutorLive } from "../src/worker-entry"
import { WorkerEvidenceLive } from "../src/worker-evidence"
import { WorkerInputsLive } from "../src/worker-inputs"
import { WorkerInvocation, WorkerReply } from "../src/worker-protocol"
import { WorkerResults, WorkerResultsLive } from "../src/worker-results"
import { WorkerTickets, WorkerTicketsLive } from "../src/worker-tickets"
import { temporaryDatabase } from "../test/postgres"

const program = Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const manifest = yield* Config.string("LAB_PROBE_MANIFEST")
  const fs = yield* FileSystem.FileSystem
  if (yield* fs.exists(root)) return yield* new AssertionFailure({ message: "Probe needs a fresh root" })
  yield* fs.makeDirectory(root, { recursive: true, mode: 0o700 })
  const prepared = yield* snapshotArtifacts(manifest, join(root, "inputs"))
  const database = yield* temporaryDatabase
  const storage = fileArtifactStore(join(root, "server-objects"))
  const registry = InputRegistryLive.pipe(Layer.provide(Layer.merge(database, storage)))
  const tickets = WorkerTicketsLive.pipe(Layer.provide(database))
  const services = Layer.mergeAll(database, registry, tickets, runStoreLayer(100).pipe(Layer.provide(database)), WorkStoreLive.pipe(Layer.provide(database)),
    WorkerInputsLive.pipe(Layer.provide(Layer.merge(registry, tickets))),
    WorkerEvidenceLive.pipe(Layer.provide(Layer.mergeAll(database, tickets, storage))),
    WorkerResultsLive.pipe(Layer.provide(Layer.merge(database, tickets))))
  yield* Effect.gen(function* () {
    yield* initializeDatabase
    const inputs = yield* InputRegistry
    const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "native-outward-probe", owner: "local-probe", trust: "developer", mode: "verify", allowSpark: false,
      input: { kind: "artifacts", digest: prepared.digest }, selection: { kind: "custom", targets: ["macos-15-arm64-metal-apple-silicon"], suites: ["package", "install"], harnesses: ["pi"] },
      limits: { concurrency: 1, deadlineMinutes: 10, budgetUsd: 10, idleMinutes: 15 } })
    for (const digest of prepared.digests) yield* inputs.upload(request.owner, digest, fs.stream(join(root, "inputs", digest)).pipe(Stream.orDie))
    yield* inputs.upload(request.owner, prepared.digest, Stream.make(new TextEncoder().encode(prepared.json)))
    yield* inputs.register(request.owner, request.input)
    yield* (yield* RunStore).submit(yield* planRun(request))
    const work = yield* WorkStore
    const assignment = Option.getOrThrow(yield* work.claim("native-local-probe", 3600))
    const invocation = WorkerInvocation.make({ schemaVersion: 1, assignment, disposable: false, port: 11429, model: "qwen3.5-4b:gguf:q4" })
    const ticket = yield* (yield* WorkerTickets).issue(invocation)
    const server = yield* HttpServer.HttpServer
    yield* server.serve(workerApi)
    if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP server")
    const reply = yield* runOutwardWorker({ root: join(root, "guest"), pollMs: 1000 }).pipe(Effect.provide([
      workerClientLayer(`http://127.0.0.1:${server.address.port}`, ticket.token), GuestExecutorLive,
    ]))
    const received = Option.getOrThrow(yield* (yield* WorkerResults).read(assignment.claim))
    if (!Schema.equivalence(WorkerReply)(reply, received)) return yield* new AssertionFailure({ message: "Received worker result differs from guest reply" })
    yield* fs.writeFileString(join(root, "outward-report.json"), yield* Schema.encode(Schema.parseJson(WorkerReply))(received), { mode: 0o600 })
    yield* work.finish(assignment.claim, received.result)
    yield* work.reconcile()
    if (received.result.cleanupErrors.length || ["P3", "I1", "I2", "I3", "I5"].some(id => received.result.cases.find(test => test.caseId === id)?.outcome.status !== "passed")) {
      return yield* new AssertionFailure({ message: "Native outward probe did not pass its required exercised cases; inspect outward-report.json" })
    }
  }).pipe(Effect.provide(services))
}))
BunRuntime.runMain(program.pipe(Effect.provide([BunContext.layer, ProcessExecutorLive, FetchHttpClient.layer, BunHttpServer.layer({ hostname: "127.0.0.1", port: 0 })])))
