import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Cause, Config, Effect, Exit, Schema } from "effect"
import { join } from "node:path"
import { desktopSession } from "../src/desktop-session"
import { occupyServicePort } from "../src/port-fault"
import { AssertionFailure } from "../src/domain"
import { assertRuntime } from "../src/runtime"

BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(root, { recursive: true, mode: 0o700 })
  const state = yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-state-" })
  const cleanupErrors: string[] = []
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.MAGNITUDE_DEV_DATA_DIR = join(root, "profile")
  environment.MAGNITUDE_DEV_PORT = String(11379)
  const result = yield* Effect.scoped(Effect.gen(function* () {
    const session = yield* desktopSession({ mode: "isolated", executable, profile: join(root, "profile"), evidence: join(root, "evidence"), port: 11379,
      environment: { ...environment, MAGNITUDE_DESKTOP_STATE_DIR: state } }, detail => { cleanupErrors.push(detail) })
    const diagnostic = yield* Effect.scoped(Effect.gen(function* () {
      yield* occupyServicePort(11379)
      const desktop = yield* session.driver
      yield* desktop.host()
      const message = yield* desktop.serviceFailure()
      yield* desktop.screenshot("service-failure")
      yield* session.stop
      return message
    }))
    yield* (yield* session.driver).ready()
    yield* (yield* session.driver).screenshot("service-recovered")
    return diagnostic
  })).pipe(Effect.exit)
  const passed = Exit.isSuccess(result) && cleanupErrors.length === 0
  yield* fs.writeFileString(join(root, "report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ passed: Schema.Boolean, detail: Schema.String, cleanupErrors: Schema.Array(Schema.String) })))({
    passed, detail: Exit.isFailure(result) ? Cause.pretty(result.cause) : result.value, cleanupErrors,
  }))
  if (!passed) return yield* new AssertionFailure({ message: "Service error probe failed; inspect report.json" })
})).pipe(Effect.provide(BunContext.layer)))
