import { assertRuntime } from "../src/runtime"
import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"

const probe = Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const requireReady = yield* Config.boolean("LAB_PROBE_READY").pipe(Config.withDefault(false))
  const fs = yield* FileSystem.FileSystem
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "SystemRoot", "TEMP", "APPDATA", "LOCALAPPDATA"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.MAGNITUDE_DEV_DATA_DIR = join(root, "profile")
  environment.MAGNITUDE_DEV_PORT = String(11279)
  const checks: string[] = []
  const result = yield* Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    checks.push(`Native host bridge reported version ${yield* desktop.host()}`)
    for (const page of ["usage", "settings", "connections", "status"] as const) {
      yield* desktop.navigate(page)
      yield* desktop.screenshot(page.toLowerCase())
      checks.push(`Navigated ${page}`)
    }
    yield* desktop.theme("dark")
    yield* desktop.screenshot("dark")
    yield* desktop.theme("light")
    yield* desktop.screenshot("light")
    checks.push("Changed appearance using Settings")
    yield* desktop.navigate("status")
    yield* fs.writeFileString(join(root, "status.txt"), yield* desktop.text())
    yield* desktop.chrome()
    checks.push("Collapsed sidebar, minimized/restored, closed/reopened window")
    if (requireReady) {
      const readiness = yield* desktop.ready().pipe(Effect.either)
      yield* desktop.screenshot("readiness")
      yield* fs.writeFileString(join(root, "readiness.txt"), yield* desktop.text())
      if (readiness._tag === "Left") return yield* readiness.left
      checks.push("Packaged service reached Ready")
    }
    yield* desktop.navigate("status")
    yield* fs.writeFileString(join(root, "status.txt"), yield* desktop.text())
    if (requireReady) {
      yield* desktop.navigate("catalog")
      yield* desktop.screenshot("catalog")
      yield* fs.writeFileString(join(root, "catalog.txt"), yield* desktop.text())
    }
  }).pipe(Effect.provide(playwrightDesktop({ mode: "isolated", executable, profile: join(root, "profile"), evidence: join(root, "evidence"), port: 11279, environment })), Effect.either)
  const report = Schema.Struct({ checks: Schema.Array(Schema.String), passed: Schema.Boolean, detail: Schema.String })
  yield* fs.writeFileString(join(root, "report.json"), yield* Schema.encode(Schema.parseJson(report))({ checks, passed: result._tag === "Right", detail: result._tag === "Left" ? String(result.left) : "Packaged UI navigation probe completed; generation and installer acceptance are separate" }))
  if (result._tag === "Left") return yield* result.left
})
BunRuntime.runMain(probe.pipe(Effect.provide(BunContext.layer)))
