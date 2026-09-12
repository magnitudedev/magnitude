import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { resolve, join } from "node:path"
import { Effect, Schema, Stream } from "effect"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { makeUnixOwnedChildSpawner, OwnedChildSpawner } from "../src/desktop-native/owned-child"
import { makeOwnedService } from "../src/desktop-native/owned-service"

class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const run = Effect.scoped(Effect.gen(function* () {
  const root = resolve(import.meta.dir, "../../..")
  const dataDir = yield* Effect.acquireRelease(
    Effect.tryPromise({ try: () => mkdtemp(join(tmpdir(), "magnitude-supervisor-")), catch: error => new AcceptanceFailed({ message: String(error) }) }),
    path => Effect.promise(() => rm(path, { recursive: true, force: true })),
  )
  const spawner = yield* makeUnixOwnedChildSpawner
  const service = yield* makeOwnedService({
    executable: process.execPath,
    arguments: [join(root, "packages/acn/src/binary.ts"), "serve", "--data-dir", dataDir, "--port", "11101"],
    environment: {
      ...process.env,
      MAGNITUDE_NATIVE_HOST: join(root, `packages/daemon-management/dist/native/${process.platform}-${process.arch}/desktop-host.node`),
      MAGNITUDE_ICN_PATH: join(root, "inference/target/development/installation.json"),
      MAGNITUDE_OTEL: "0",
    },
  }, 1).pipe(Effect.provideService(OwnedChildSpawner, spawner))
  yield* service.changes.pipe(Stream.runForEach(state => Effect.log(`Supervisor: ${state._tag}`)), Effect.forkScoped)
  const first = yield* service.awaitReady.pipe(Effect.timeout("3 minutes"))
  yield* Effect.sync(() => process.kill(first.pid, "SIGKILL"))
  // Observe the loss before waiting for the replacement; a previous Ready value is not recovery.
  yield* service.changes.pipe(Stream.filter(state => state._tag !== "Ready"), Stream.take(1), Stream.runDrain, Effect.timeout("10 seconds"))
  const recovered = yield* service.awaitReady.pipe(Effect.timeout("3 minutes"))
  if (recovered.id === first.id || recovered.pid === first.pid) return yield* new AcceptanceFailed({ message: "Supervisor reused the crashed service" })
  yield* Effect.all([service.shutdown, service.shutdown], { concurrency: "unbounded" })
  if ((yield* service.state)._tag !== "Stopped") return yield* new AcceptanceFailed({ message: "Quit did not reach Stopped" })
  const groups = yield* ProcessGroupController
  for (const pid of [first.pid, recovered.pid]) {
    if ((yield* groups.inspect(pid))._tag !== "None") return yield* new AcceptanceFailed({ message: `Owned service ${pid} survived Quit` })
  }
  yield* Effect.log("PASS: real supervisor startup, crash recovery, concurrent Quit, no surviving service")
})).pipe(Effect.provideService(ProcessGroupController, ProcessGroupControllerLive))
Effect.runPromise(run).catch(error => { console.error(String(error)); process.exitCode = 1 })
