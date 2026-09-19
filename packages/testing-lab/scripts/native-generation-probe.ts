import { FetchHttpClient, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Option, Schema, Stream } from "effect"
import { dirname, join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { executionTelemetry, NativeExecution } from "../src/execution-telemetry"
import { GenerationExecution, observeGeneration } from "../src/generation-evidence"
import { assertRuntime } from "../src/runtime"
import { AssertionFailure, Digest, InfrastructureFailure } from "../src/domain"
import { verifiedArtifactContent } from "../src/artifact-store"

/** Exercise real native completion through a packaged UI; an explicit development installation
 * makes this a diagnostic, not acceptance of the package's released inference artifacts. */
BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const installation = yield* Config.string("LAB_PROBE_ICN_INSTALLATION")
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const port = yield* Config.integer("LAB_PROBE_PORT").pipe(Config.withDefault(11339))
  const state = yield* fs.makeTempDirectoryScoped({ ...(process.platform === "win32" ? {} : { directory: "/tmp" }), prefix: "ml-execution-" })
  const collector = yield* executionTelemetry()
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "SystemRoot", "TEMP", "APPDATA", "LOCALAPPDATA"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const observations: (typeof GenerationExecution.Type)[] = []
  const cleanup: string[] = []
  const result = yield* Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.host()
    yield* desktop.ready()
    yield* desktop.search(model)
    yield* desktop.load(model)
    const execution = yield* observeGeneration(`http://127.0.0.1:${port}`, model, collector)
    observations.push(execution)
    if (Option.isNone(execution.native.modules) || execution.native.modules.value.length === 0) {
      return yield* new AssertionFailure({ message: "Native completion has no loaded module observation" })
    }
    for (const module of execution.native.modules.value) {
      yield* verifiedArtifactContent(Digest.make(module.sha256), fs.stream(join(dirname(installation), "backends", module.name)).pipe(
        Stream.mapError(() => new InfrastructureFailure({ operation: "module-observation", message: "Cannot read observed module backing file" }))), module.bytes).pipe(Stream.runDrain)
    }
    yield* desktop.screenshot("native-generation")
  }).pipe(Effect.provide(playwrightDesktop({ executable, profile: join(root, "profile"), evidence: join(root, "evidence"), port,
    environment: { ...environment, MAGNITUDE_DESKTOP_STATE_DIR: state, MAGNITUDE_ICN_PATH: installation,
      MAGNITUDE_OTEL_ENDPOINT: collector.endpoint },
  }, undefined, detail => cleanup.push(detail))), Effect.either)
  const collection = yield* collector.observations.pipe(Effect.mapError(error => error.message), Effect.either)
  yield* fs.writeFileString(join(root, "native-generation.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({
    passed: Schema.Boolean, detail: Schema.String, observations: Schema.Array(GenerationExecution), cleanup: Schema.Array(Schema.String),
    collection: Schema.Either({ left: Schema.String, right: Schema.Array(NativeExecution) }),
  })))({ passed: result._tag === "Right" && cleanup.length === 0,
    detail: result._tag === "Right" ? "Real public generation correlated to native target-model allocations and already-loaded module backing files; admitted package inference acquisition and full backend qualification remain unverified" : String(result.left),
    observations, cleanup, collection }))
  if (result._tag === "Left") return yield* result.left
  if (cleanup.length) return yield* new InfrastructureFailure({ operation: "desktop-cleanup", message: "Native generation probe cleanup failed" })
})).pipe(Effect.provide(Layer.merge(BunContext.layer, FetchHttpClient.layer))))
