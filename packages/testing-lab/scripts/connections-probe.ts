import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Cause, Config, Effect, Exit, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { connectionFixture } from "../src/harnesses/connection-fixture"
import { exerciseConnectionError } from "../src/harnesses/connection-error"
import { AssertionFailure, Harness } from "../src/domain"
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
  environment.MAGNITUDE_DEV_DATA_DIR = join(root, "profile")
  environment.MAGNITUDE_DEV_PORT = String(port)
  environment.PATH = `${toolsPath}:${environment.PATH ?? ""}`
  environment.MAGNITUDE_DESKTOP_STATE_DIR = state
  const fixtures = yield* Effect.forEach(["pi", "opencode", "hermes"] as const, harness => connectionFixture(join(root, "profile", "harness-home"), harness, `http://127.0.0.1:${port}/inference/v1`))
  const checks: { harness: typeof Harness.Type; passed: boolean; detail: string }[] = []
  const cleanupErrors: string[] = []
  const result = yield* Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.host()
    yield* desktop.ready()
    for (const fixture of fixtures) {
      const check = yield* Effect.gen(function* () {
        yield* fixture.exercise
        yield* exerciseConnectionError(join(root, "profile", "harness-home"), fixture.harness)
        yield* fixture.inspect(true)
      }).pipe(Effect.exit)
      checks.push({ harness: fixture.harness, passed: Exit.isSuccess(check), detail: Exit.isSuccess(check)
        ? "Connection lifecycle and malformed-file recovery passed" : Cause.pretty(check.cause) })
    }
  }).pipe(Effect.provide(playwrightDesktop({ mode: "isolated", executable, profile: join(root, "profile"), evidence: join(root, "evidence"), port, environment }, undefined, detail => { cleanupErrors.push(detail) })), Effect.exit)
  const passed = Exit.isSuccess(result) && cleanupErrors.length === 0 && checks.length === fixtures.length && checks.every(check => check.passed)
  const report = Schema.Struct({ passed: Schema.Boolean, checks: Schema.Array(Schema.Struct({ harness: Harness, passed: Schema.Boolean, detail: Schema.String })),
    setupFailure: Schema.String, cleanupErrors: Schema.Array(Schema.String) })
  yield* fs.writeFileString(join(root, "connections-report.json"), yield* Schema.encode(Schema.parseJson(report))({
    passed, checks, setupFailure: Exit.isFailure(result) ? Cause.pretty(result.cause) : "", cleanupErrors,
  }))
  if (!passed) return yield* new AssertionFailure({ message: "Harness connection checks failed; inspect connections-report.json" })
}).pipe(Effect.scoped, Effect.provide(BunContext.layer)))
