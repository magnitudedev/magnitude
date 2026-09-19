import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Cause, Config, Effect, Exit, Schema } from "effect"
import { join } from "node:path"
import { desktopSession } from "../src/desktop-session"
import { AssertionFailure } from "../src/domain"
import { assertRuntime } from "../src/runtime"

BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(root, { recursive: true, mode: 0o700 })
  const state = yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-state-" })
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const cleanupErrors: string[] = []
  const result = yield* Effect.scoped(Effect.gen(function* () {
    const session = yield* desktopSession({ executable, profile: join(root, "profile"), evidence: join(root, "evidence"), port: 11349,
      environment: { ...environment, MAGNITUDE_DESKTOP_STATE_DIR: state } }, detail => { cleanupErrors.push(detail) })
    yield* (yield* session.driver).host()
    for (const theme of ["dark", "light"] as const) {
      yield* (yield* session.driver).theme(theme)
      const restarted = yield* session.restart
      yield* restarted.host()
      yield* restarted.verifyTheme(theme)
      yield* restarted.screenshot(`persisted-${theme}`)
    }
    const driver = yield* session.driver
    yield* driver.ready()
    yield* driver.chrome()
    yield* driver.quit()
    yield* (yield* session.restart).ready()
  })).pipe(Effect.exit)
  const report = { passed: Exit.isSuccess(result) && cleanupErrors.length === 0,
    detail: Exit.isFailure(result) ? Cause.pretty(result.cause) : "Dark and light appearance survived restarts; hide/reopen, clean quit and service readiness after restart passed", cleanupErrors }
  yield* fs.writeFileString(join(root, "report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ passed: Schema.Boolean, detail: Schema.String, cleanupErrors: Schema.Array(Schema.String) })))(report))
  if (!report.passed) return yield* new AssertionFailure({ message: report.detail })
})).pipe(Effect.provide(BunContext.layer)))
