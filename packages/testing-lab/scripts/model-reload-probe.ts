import { FetchHttpClient, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { EndpointTests, endpointTests, Generation } from "../src/suites/endpoint"
import { bundledCliTests, CliTests } from "../src/suites/cli"
import { ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"

BunRuntime.runMain(Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const cli = yield* Config.string("LAB_PROBE_BUNDLED_CLI")
  const version = yield* Config.string("LAB_PROBE_VERSION")
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const toolsPath = yield* Config.string("LAB_PROBE_TOOLS_PATH")
  const port = yield* Config.integer("LAB_PROBE_PORT").pipe(Config.withDefault(11319))
  const fs = yield* FileSystem.FileSystem
  const state = yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-connections-" })
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.PATH = `${toolsPath}:${environment.PATH ?? ""}`
  environment.MAGNITUDE_DESKTOP_STATE_DIR = state
  environment.MAGNITUDE_DEV_DATA_DIR = join(root, "profile")
  environment.MAGNITUDE_DEV_PORT = String(port)
  environment.MAGNITUDE_SHELL_ENV_INHERITED = "1"
  const generations: Generation[] = []
  const result = yield* Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.host()
    yield* desktop.ready()
    const tests = yield* CliTests
    yield* tests.version
    yield* desktop.search(model)
    yield* desktop.load(model)
    const endpoint = yield* EndpointTests
    generations.push(yield* endpoint.generate)
    yield* tests.reloadModel
    generations.push(yield* endpoint.generate)
  }).pipe(Effect.provide(Layer.mergeAll(playwrightDesktop({ executable, profile: join(root, "profile"), evidence: join(root, "evidence"), port, environment }), bundledCliTests({ executable: cli, version, model, evidence: join(root, "cli-evidence"), environment }), endpointTests(`http://127.0.0.1:${port}`, model).pipe(Layer.provide(FetchHttpClient.layer)))), Effect.either)
  yield* fs.writeFileString(join(root, "reload-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ passed: Schema.Boolean, detail: Schema.String, generations: Schema.Array(Generation) })))({ generations, passed: result._tag === "Right", detail: result._tag === "Right" ? "Generation succeeded before and after CLI stop/unload/reload; backend allocation is not attested by this diagnostic" : String(result.left) }))
  if (result._tag === "Left") return yield* result.left
}).pipe(Effect.scoped, Effect.provide(Layer.merge(BunContext.layer, ProcessExecutorLive))))
