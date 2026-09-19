import { FetchHttpClient, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { EndpointTests, endpointTests } from "../src/suites/endpoint"
import { assertRuntime } from "../src/runtime"
import { challengePresentation } from "../test-support/presentation-challenge"

const run = Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const toolsPath = yield* Config.string("LAB_PROBE_TOOLS_PATH")
  const fs = yield* FileSystem.FileSystem
  const environment: Record<string, string> = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.PATH = `${toolsPath}:${environment.PATH ?? ""}`
  const program = Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    const endpoint = yield* EndpointTests
    yield* desktop.host()
    yield* desktop.ready()
    yield* desktop.theme("dark")
    yield* desktop.theme("light")
    yield* desktop.chrome()
    yield* desktop.search(model)
    yield* desktop.load(model)
    yield* desktop.connect("pi")
    yield* desktop.screenshot("reworded-recolored-reordered-ui")
    yield* desktop.disconnect("pi")
    yield* endpoint.generate
  }).pipe(Effect.provide(Layer.merge(playwrightDesktop({ executable, profile: join(root, "profile"), evidence: join(root, "resilience-evidence"), port: 11279, environment }, challengePresentation),
    endpointTests("http://127.0.0.1:11279", model).pipe(Layer.provide(FetchHttpClient.layer)))))
  const result = yield* program.pipe(Effect.either)
  yield* fs.writeFileString(join(root, "resilience-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ passed: Schema.Boolean, detail: Schema.String })))({
    passed: result._tag === "Right", detail: result._tag === "Right" ? "Changed labels, accessible names, placeholders, colors and visual order; navigation, settings, window lifecycle, model loading, connections and generation passed" : String(result.left),
  }))
  if (result._tag === "Left") return yield* result.left
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
