import { FileSystem, FetchHttpClient } from "@effect/platform"
import { BunContext, BunHttpServer } from "@effect/platform-bun"
import { ConfigProvider, Effect, Fiber, Layer, Option, Redacted, Schema, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { bearerAuthenticator } from "../src/api"
import { fileArtifactStore } from "../src/artifact-store"
import { LabClient, labClientLayer } from "../src/client"
import { startCoordinator } from "../src/coordinator"
import { OwnerId, Principal, RunRequest } from "../src/domain"
import { ProcessExecutorLive } from "../src/process"
import { MachineProviders, WorkerRunner } from "../src/scheduler"
import { sha256 } from "../src/snapshot"
import { temporaryDatabase } from "./postgres"
import { configuredCoordinator } from "../src/server"
import { Database } from "../src/database"

test("live coordinator admits API inputs, schedules runs and preserves owner isolation", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-coordinator-" })
  const database = yield* temporaryDatabase
  const token = Redacted.make("a".repeat(40)), other = Redacted.make("b".repeat(40))
  const program = Effect.gen(function* () {
    const service = yield* startCoordinator({ instance: "coordinator-test", concurrency: 2, accountBudgetUsd: 100, pollMs: 20, reconcileMs: 50 })
    const running = yield* Effect.forkScoped(service.run)
    if (service.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP address")
    const origin = `http://127.0.0.1:${service.address.port}`
    const ownerClient = labClientLayer(origin, Effect.succeed(token)).pipe(Layer.provide(FetchHttpClient.layer))
    const source = '{"schemaVersion":1,"kind":"source","commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","entries":[]}'
    const input = { kind: "source" as const, digest: sha256(source) }
    const id = yield* Effect.gen(function* () {
      const client = yield* LabClient
      expect((yield* client.identity()).owner).toBe("owner")
      yield* client.upload(input.digest, Stream.make(new TextEncoder().encode(source)))
      yield* client.registerInput(input)
      const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "http-coordinator-run", owner: "owner", input,
        selection: { kind: "profile", profile: "quick", target: "ubuntu-24.04-x64-cpu-intel" }, mode: "verify", trust: "developer", allowSpark: false,
        limits: { concurrency: 1, deadlineMinutes: 5, budgetUsd: 5, idleMinutes: 1 } })
      const accepted = yield* client.submit(request)
      const id = accepted.state.runId
      expect((yield* client.submit(request)).state.runId).toBe(id)
      const result = yield* Effect.gen(function* () {
        for (;;) { const result = yield* client.result(id); if (Option.isSome(result)) return result.value; yield* Effect.sleep("20 millis") }
      }).pipe(Effect.timeout("10 seconds"))
      expect(result.cases.length).toBeGreaterThan(10)
      expect(result.cases.every(test => test.outcome.status === "blocked" && test.outcome.detail.includes("not configured"))).toBe(true)
      return id
    }).pipe(Effect.provide(ownerClient))
    const denied = yield* Effect.flatMap(LabClient, client => client.get(id)).pipe(Effect.either,
      Effect.provide(labClientLayer(origin, Effect.succeed(other)).pipe(Layer.provide(FetchHttpClient.layer))))
    expect(denied._tag === "Left" && denied.left.status).toBe(403)
    yield* Fiber.interrupt(running)
  }).pipe(Effect.provide([database, fileArtifactStore(join(root, "objects")), BunHttpServer.layer({ hostname: "127.0.0.1", port: 0 }),
    bearerAuthenticator([{ token, principal: Principal.make({ owner: OwnerId.make("owner"), trust: "developer" }) }, { token: other, principal: Principal.make({ owner: OwnerId.make("other"), trust: "developer" }) }]),
    Layer.succeed(MachineProviders, { allocators: new Map() }), Layer.succeed(WorkerRunner, { run: () => Effect.dieMessage("No provider may allocate in this test") })]))
  yield* program
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))

test("configured entry point keeps the server alive and rejects weak credentials", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-server-config-" })
  const database = yield* temporaryDatabase
  const rows = yield* Effect.flatMap(Database, db => db.query("SELECT current_setting('unix_socket_directories') AS socket")).pipe(Effect.provide(database))
  const row = yield* Schema.decodeUnknown(Schema.Struct({ socket: Schema.String }))(rows[0])
  const file = join(root, "server.json")
  yield* fs.writeFileString(file, yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ coordinator: { instance: "configured-test", concurrency: 1, accountBudgetUsd: 50, pollMs: 20, reconcileMs: 50 },
    hostname: "127.0.0.1", port: 0, credentials: [{ tokenEnvironment: "TEST_LAB_TOKEN", principal: { owner: "configured", trust: "developer" } }],
    storage: { kind: "file", directory: join(root, "objects") }, runtimes: [] }))
  const values = new Map([["LAB_COORDINATOR_CONFIG", file], ["LAB_DATABASE_URL", `postgresql://lab@localhost/postgres?host=${encodeURIComponent(row.socket)}`], ["TEST_LAB_TOKEN", "short"]])
  const invalid = yield* configuredCoordinator.pipe(Effect.withConfigProvider(ConfigProvider.fromMap(values)), Effect.either)
  expect(invalid._tag === "Left" && invalid.left._tag).toBe("InvalidInput")
  values.set("TEST_LAB_TOKEN", "c".repeat(40))
  const service = yield* configuredCoordinator.pipe(Effect.withConfigProvider(ConfigProvider.fromMap(values)))
  const running = yield* Effect.forkScoped(service.run)
  if (service.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected configured TCP address")
  const identity = yield* Effect.flatMap(LabClient, client => client.identity()).pipe(Effect.provide(
    labClientLayer(`http://127.0.0.1:${service.address.port}`, Effect.succeed(Redacted.make("c".repeat(40)))).pipe(Layer.provide(FetchHttpClient.layer))))
  expect(identity.owner).toBe("configured")
  yield* Fiber.interrupt(running)
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
