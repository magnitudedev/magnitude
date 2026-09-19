import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { connectionFixture } from "../src/harnesses/connection-fixture"
import { assertRuntime } from "../src/runtime"

BunRuntime.runMain(Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const toolsPath = yield* Config.string("LAB_PROBE_TOOLS_PATH")
  const port = yield* Config.integer("LAB_PROBE_PORT").pipe(Config.withDefault(11319))
  const fs = yield* FileSystem.FileSystem
  const state = yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-connections-" })
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.PATH = `${toolsPath}:${environment.PATH ?? ""}`
  environment.MAGNITUDE_DESKTOP_STATE_DIR = state
  const fixtures = yield* Effect.forEach(["pi", "opencode", "hermes"] as const, harness => connectionFixture(join(root, "profile", "harness-home"), harness, `http://127.0.0.1:${port}/inference/v1`))
  const result = yield* Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.host()
    yield* desktop.ready()
    for (const fixture of fixtures) yield* fixture.exercise
  }).pipe(Effect.provide(playwrightDesktop({ executable, profile: join(root, "profile"), evidence: join(root, "evidence"), port, environment })), Effect.either)
  yield* fs.writeFileString(join(root, "connections-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ passed: Schema.Boolean, detail: Schema.String })))({ passed: result._tag === "Right", detail: result._tag === "Right" ? "Pi, OpenCode and Hermes connect, refresh, disconnect, reconnect preserved unrelated providers and used the test endpoint" : String(result.left) }))
  if (result._tag === "Left") return yield* result.left
}).pipe(Effect.scoped, Effect.provide(BunContext.layer)))
