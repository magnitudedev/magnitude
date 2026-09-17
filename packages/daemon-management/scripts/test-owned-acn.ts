import { execFile } from "node:child_process"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { resolve, join } from "node:path"
import { Effect, Deferred, Stream, Schema, Option } from "effect"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { makeUnixOwnedChildSpawner } from "../src/desktop-native/owned-child"

class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const run = Effect.scoped(Effect.gen(function* () {
  const mode = yield* Schema.decodeUnknown(Schema.Literal("shutdown", "service-crash", "engine-crash"))(process.argv[2] ?? "shutdown")
  const root = resolve(import.meta.dir, "../../..")
  const dataDir = yield* Effect.acquireRelease(
    Effect.tryPromise({ try: () => mkdtemp(join(tmpdir(), "magnitude-owned-acn-")), catch: error => new AcceptanceFailed({ message: String(error) }) }),
    path => Effect.promise(() => rm(path, { recursive: true, force: true })),
  )
  const spawner = yield* makeUnixOwnedChildSpawner
  const child = yield* spawner.spawn({
    executable: process.execPath,
    arguments: [join(root, "packages/acn/src/binary.ts"), "serve", "--data-dir", dataDir, "--port", "11101"],
    environment: {
      ...process.env,
      MAGNITUDE_NATIVE_HOST: join(root, `packages/daemon-management/dist/native/${process.platform}-${process.arch}/desktop-host.node`),
      MAGNITUDE_ICN_PATH: join(root, "inference/target/development/installation.json"),
      MAGNITUDE_OTEL: "0",
    },
  })
  const ready = yield* Deferred.make<void, AcceptanceFailed>()
  yield* child.events.pipe(Stream.runForEach(event => Effect.gen(function* () {
    if (event._tag === "Booted") {
      if (event.pid !== child.identity.pid) return yield* new AcceptanceFailed({ message: "Wrong child identity" })
      yield* Effect.log(`Booted owned ACN ${event.pid}; authorizing Start`)
      yield* child.send({ _tag: "Start" })
    } else {
      yield* Effect.log(`ACN: ${event.health.state._tag}`)
      if (event.health.state._tag === "Ready") yield* Deferred.succeed(ready, undefined)
    }
  })), Effect.catchAll(error => Deferred.fail(ready, new AcceptanceFailed({ message: String(error) }))), Effect.forkScoped)
  yield* Effect.raceFirst(
    Deferred.await(ready),
    child.exit.pipe(Effect.flatMap(code => child.diagnosticTail.pipe(Effect.flatMap(tail => Effect.fail(new AcceptanceFailed({ message: `ACN exited ${code}: ${tail}` })))))),
  ).pipe(Effect.timeout("3 minutes"), Effect.tapErrorCause(() => child.diagnosticTail.pipe(Effect.flatMap(Effect.logError))))
  const response = yield* Effect.tryPromise({ try: () => fetch("http://127.0.0.1:11101/health"), catch: error => new AcceptanceFailed({ message: String(error) }) })
  if (response.status !== 200) return yield* new AcceptanceFailed({ message: `Health status ${response.status}` })
  const table = yield* Effect.async<string, AcceptanceFailed>(resume => {
    execFile("/bin/ps", ["-axo", "pid=,ppid="], (error, stdout) => resume(error
      ? Effect.fail(new AcceptanceFailed({ message: error.message })) : Effect.succeed(stdout)))
  })
  const children = table.trim().split("\n").map(row => row.trim().split(/\s+/).map(Number))
    .filter(row => row[1] === child.identity.pid)
  if (children.length !== 1) return yield* new AcceptanceFailed({ message: `Expected one private ICN, found ${children.length}` })
  const groups = yield* ProcessGroupController
  const icnIdentity = yield* groups.inspect(children[0]![0]!)
  if (Option.isNone(icnIdentity)) return yield* new AcceptanceFailed({ message: "Private ICN disappeared" })
  if (mode === "shutdown") yield* child.send({ _tag: "Shutdown" })
  else process.kill(mode === "service-crash" ? child.identity.pid : icnIdentity.value.pid, "SIGKILL")
  yield* child.exit.pipe(Effect.timeout("15 seconds"))
  yield* child.stop
  if (!(yield* groups.waitForGroupExit({ leader: icnIdentity.value }, "5 seconds"))) {
    // Failure cleanup is restricted to the exact private child identified above.
    yield* groups.stop({ leader: icnIdentity.value })
    return yield* new AcceptanceFailed({ message: "ICN descendants survived owner termination" })
  }
  yield* Effect.log(`PASS: real ACN/ICN ${mode}, complete service and engine group cleanup`)

})).pipe(Effect.provideService(ProcessGroupController, ProcessGroupControllerLive))
Effect.runPromise(run).catch(error => { console.error(String(error)); process.exitCode = 1 })
