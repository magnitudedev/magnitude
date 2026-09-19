import { FetchHttpClient, FileSystem, HttpClient, HttpServer } from "@effect/platform"
import { BunContext, BunHttpServer, BunRuntime } from "@effect/platform-bun"
import { Config, Context, Effect, Layer, Option, Schema, Scope, Stream } from "effect"
import { join } from "node:path"
import { snapshotArtifacts } from "../src/artifact-input"
import { fileArtifactStore } from "../src/artifact-store"
import { planRun } from "../src/catalog"
import { initializeDatabase } from "../src/database"
import { AssertionFailure, LeaseId, RunRequest } from "../src/domain"
import { LocalMachine } from "../src/machines"
import { outwardWorkerRunner, type WorkerBootstrap, WorkerBootstraps } from "../src/outward-runner"
import { WorkerRunner } from "../src/scheduler"
import { InputRegistry, InputRegistryLive } from "../src/inputs"
import { deliverOutwardWorkerResult, runOutwardWorker } from "../src/outward-worker"
import { ProcessExecutor, ProcessExecutorLive } from "../src/process"
import { RunStore, runStoreLayer } from "../src/run-store"
import { assertRuntime } from "../src/runtime"
import { WorkStore, WorkStoreLive } from "../src/work-store"
import { workerApi } from "../src/worker-api"
import { WorkerApiError, WorkerClient, workerClientLayer } from "../src/worker-client"
import { GuestExecutorLive } from "../src/worker-entry"
import { WorkerEvidenceLive } from "../src/worker-evidence"
import { WorkerInputsLive } from "../src/worker-inputs"
import { WorkerReply } from "../src/worker-protocol"
import { WorkerResults, WorkerResultsLive } from "../src/worker-results"
import { WorkerTicketsLive } from "../src/worker-tickets"
import { temporaryDatabase } from "../test/postgres"

const program = Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const manifest = yield* Config.string("LAB_PROBE_MANIFEST")
  const recoverDelivery = yield* Config.boolean("LAB_PROBE_DELIVERY_RECOVERY").pipe(Config.withDefault(false))
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
    const server = yield* HttpServer.HttpServer
    yield* server.serve(workerApi)
    if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP server")
    const origin = `http://127.0.0.1:${server.address.port}`
    const scope = yield* Effect.scope
    const guestServices = yield* Effect.context<FileSystem.FileSystem | ProcessExecutor | HttpClient.HttpClient | Scope.Scope>()
    let guestRoot = ""
    const bootstrap = Layer.succeed(WorkerBootstraps, { providers: new Map<"local", WorkerBootstrap>([["local", { start: (_machine, launch) => Effect.gen(function* () {
      guestRoot = launch.root
      const guest = Effect.gen(function* () {
        const client = yield* WorkerClient
        const config = { root: launch.root, pollMs: 1000 }
        if (!recoverDelivery) return yield* runOutwardWorker(config).pipe(Effect.provide(GuestExecutorLive))
        const interrupted = yield* runOutwardWorker(config).pipe(Effect.provide(GuestExecutorLive),
          Effect.provideService(WorkerClient, { ...client, submit: () => Effect.fail(new WorkerApiError({ status: 503, message: "Injected result delivery failure" })) }), Effect.either)
        if (interrupted._tag !== "Left" || interrupted.left._tag !== "WorkerApiError" || interrupted.left.status !== 503) return yield* new AssertionFailure({ message: "Probe did not reach the intended result delivery failure" })
        if (yield* fs.exists(join(launch.root, "installation", "Magnitude.app"))) return yield* new AssertionFailure({ message: "Native installation was not cleaned before result recovery" })
        const reply = yield* deliverOutwardWorkerResult(config)
        yield* fs.writeFileString(join(root, "delivery-recovery.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ passed: Schema.Boolean, recoveredWithoutExecutor: Schema.Boolean, cleanedBeforeDelivery: Schema.Boolean })))({ passed: true, recoveredWithoutExecutor: true, cleanedBeforeDelivery: true }))
        return reply
      })
      yield* guest.pipe(Effect.provide(workerClientLayer(launch.origin, launch.token)), Effect.provide(guestServices), Effect.forkIn(scope))
    }) }]]) })
    const runner = Context.get(yield* Layer.build(outwardWorkerRunner({ origin, pollMs: 100, runtimes: [{ provider: "local", artifactHost: "darwin-arm64", root: join(root, "guest"),
      executable: "bun", args: ["src/outward-worker.ts"], disposable: false, port: 11429, model: "qwen3.5-4b:gguf:q4" }] }).pipe(Layer.provide(bootstrap))), WorkerRunner)
    const machine = LocalMachine.make({ provider: "local", root: join(root, "guest"), tags: { schemaVersion: 1, runId: assignment.claim.runId,
      leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), expiresAt: assignment.deadline } })
    const result = yield* runner.run(machine, assignment)
    const reply = WorkerReply.make({ schemaVersion: 1, claim: assignment.claim, result })
    const received = Option.getOrThrow(yield* (yield* WorkerResults).read(assignment.claim))
    if (!Schema.equivalence(WorkerReply)(reply, received)) return yield* new AssertionFailure({ message: "Received worker result differs from guest reply" })
    yield* fs.writeFileString(join(root, "outward-report.json"), yield* Schema.encode(Schema.parseJson(WorkerReply))(received), { mode: 0o600 })
    yield* work.finish(assignment.claim, received.result)
    yield* work.reconcile()
    if (yield* fs.exists(join(guestRoot, "installation", "Magnitude.app"))) return yield* new AssertionFailure({ message: "Native app remains after outward execution" })
    if (received.result.cleanupErrors.length || ["P3", "I1", "I2", "I3", "I5"].some(id => received.result.cases.find(test => test.caseId === id)?.outcome.status !== "passed")) {
      return yield* new AssertionFailure({ message: "Native outward probe did not pass its required exercised cases; inspect outward-report.json" })
    }
  }).pipe(Effect.provide(services))
}))
BunRuntime.runMain(program.pipe(Effect.provide([BunContext.layer, ProcessExecutorLive, FetchHttpClient.layer, BunHttpServer.layer({ hostname: "127.0.0.1", port: 0 })])))
