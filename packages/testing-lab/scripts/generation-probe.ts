import { FetchHttpClient, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { EndpointTests, endpointTests, type Generation } from "../src/suites/endpoint"
import { AssertionFailure } from "../src/domain"
import { assertRuntime } from "../src/runtime"

const Report = Schema.Struct({ model: Schema.String, appVersion: Schema.String, checks: Schema.Array(Schema.Struct({ name: Schema.String, passed: Schema.Boolean, detail: Schema.String })) })
const run = Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const cached = yield* Config.boolean("LAB_PROBE_CACHED").pipe(Config.withDefault(false))
  const toolCases = yield* Config.integer("LAB_PROBE_TOOL_CASES").pipe(Config.withDefault(1),
    Effect.flatMap(Schema.decodeUnknown(Schema.Int.pipe(Schema.between(1, 20)))))
  const fs = yield* FileSystem.FileSystem
  const state = yield* fs.makeTempDirectoryScoped({ ...(process.platform === "win32" ? {} : { directory: "/tmp" }), prefix: "ml-generation-" })
  const checks: { name: string; passed: boolean; detail: string }[] = []
  let appVersion = "unknown"
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "SystemRoot", "TEMP", "APPDATA", "LOCALAPPDATA"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.MAGNITUDE_DESKTOP_STATE_DIR = state
  const program = Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    const endpoint = yield* EndpointTests
    appVersion = yield* desktop.host()
    yield* desktop.ready()
    yield* desktop.search(model)
    yield* desktop.screenshot("model-before-download")
    if (!cached) {
      yield* desktop.download(model)
      yield* desktop.screenshot("model-downloaded")
      checks.push({ name: "Download through the packaged app", passed: true, detail: model })
    }
    yield* desktop.load(model)
    yield* desktop.screenshot("model-loaded")
    checks.push({ name: "Load through the packaged app", passed: true, detail: model })
    const steps: ReadonlyArray<readonly [string, Effect.Effect<Generation | void, AssertionFailure>]> = [
      ["Endpoint discovery", endpoint.discover], ["Nonstreamed generation", endpoint.generate], ["Streaming generation", endpoint.stream],
      ...Array.from({ length: toolCases }, (_, index) => [toolCases === 1 ? "Tool call and follow-up" : `Tool call and follow-up ${index + 1}/${toolCases}`, endpoint.tools] as const),
      ["Invalid requests and subsequent generation", endpoint.invalid], ["Cancellation and subsequent generation", endpoint.cancelAndRetry],
    ]
    for (const [name, test] of steps) {
      const result = yield* test.pipe(Effect.either)
      checks.push({ name, passed: result._tag === "Right", detail: result._tag === "Left" ? result.left.message : typeof result.right === "object" ? result.right.text : "Assertions passed" })
    }
    yield* desktop.navigate("status")
    yield* desktop.screenshot("generation-status")
    yield* fs.writeFileString(join(root, "status.txt"), yield* desktop.text())
  }).pipe(Effect.tapError(() => Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.screenshot("failure").pipe(Effect.ignore)
    yield* fs.writeFileString(join(root, "failure.txt"), yield* desktop.text()).pipe(Effect.ignore)
  })), Effect.provide(Layer.merge(playwrightDesktop({ executable, profile: join(root, "profile"), evidence: join(root, "evidence"), port: 11279, environment }),
    endpointTests("http://127.0.0.1:11279", model).pipe(Layer.provide(FetchHttpClient.layer)))))
  const result = yield* program.pipe(Effect.either)
  if (result._tag === "Left") checks.push({ name: "Application model journey", passed: false, detail: String(result.left) })
  yield* fs.writeFileString(join(root, "generation-report.json"), yield* Schema.encode(Schema.parseJson(Report))({ model, appVersion, checks }))
  if (result._tag === "Left") return yield* result.left
  if (checks.some(c => !c.passed)) return yield* new AssertionFailure({ message: "One or more packaged generation checks failed; see generation-report.json" })
})
BunRuntime.runMain(run.pipe(Effect.scoped, Effect.provide(BunContext.layer)))
