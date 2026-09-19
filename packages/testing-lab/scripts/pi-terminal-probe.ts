import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Schema } from "effect"
import { dirname, join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { piTerminal, PiTerminalReceipt } from "../src/harnesses/pi-terminal"
import { NativeTerminalDriver } from "../src/terminal"
import { ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { AssertionFailure } from "../src/domain"

BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT"), executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const runtime = yield* Config.string("LAB_TERMINAL_NODE_EXECUTABLE"), pi = yield* Config.string("LAB_PI_EXECUTABLE")
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const port = yield* Config.integer("LAB_PROBE_PORT").pipe(Config.withDefault(11339))
  const fs = yield* FileSystem.FileSystem
  const evidence = yield* fs.makeTempDirectory({ directory: root, prefix: "pi-terminal-" })
  const state = yield* fs.makeTempDirectoryScoped({ prefix: "lab-pi-desktop-" })
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "SystemRoot"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.PATH = `${dirname(runtime)}${process.platform === "win32" ? ";" : ":"}${environment.PATH ?? ""}`
  environment.MAGNITUDE_DEV_DATA_DIR = join(root, "profile")
  environment.MAGNITUDE_DEV_PORT = String(port)
  environment.MAGNITUDE_DESKTOP_STATE_DIR = state
  const cleanup: string[] = []
  const result = yield* Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.host()
    yield* desktop.ready()
    yield* desktop.search(model)
    yield* desktop.load(model)
    yield* desktop.connect("pi")
    const home = join(root, "profile", "harness-home")
    const first = crypto.randomUUID().replaceAll("-", "").slice(0, 8), second = crypto.randomUUID().replaceAll("-", "").slice(0, 8)
    return yield* piTerminal({ runtime, executable: pi, cwd: evidence, evidence: join(evidence, "terminal"),
      environment: { ...environment, HOME: home, USERPROFILE: home, PI_CODING_AGENT_DIR: join(home, ".pi", "agent") }, model, initialModel: model,
      interrupt: { prompt: `First concatenate READY and ${first} without a space and print that word. Then count from 1 to 10000, one number per line. Do not use tools.`, expected: `READY${first}` },
      recovery: { prompt: `Concatenate DONE and ${second} without a space. Reply only with the resulting word. Do not use tools.`, expected: `DONE${second}` },
    }, message => { cleanup.push(message) })
  }).pipe(Effect.provide(playwrightDesktop({ mode: "isolated", executable, profile: join(root, "profile"), evidence: join(evidence, "app"), port, environment })), Effect.either)
  const report = Schema.Struct({ passed: Schema.Boolean, detail: Schema.String, cleanup: Schema.Array(Schema.String),
    receipts: Schema.Array(PiTerminalReceipt) })
  yield* fs.writeFileString(join(evidence, "report.json"), yield* Schema.encode(Schema.parseJson(report))({
    passed: result._tag === "Right" && cleanup.length === 0, cleanup,
    detail: result._tag === "Right" ? "Pi keyboard selection, real generation, native abort receipt, follow-up and normal exit" : String(result.left),
    receipts: result._tag === "Right" ? [result.right] : [],
  }))
  yield* Effect.logInfo(`Pi terminal evidence: ${evidence}`)
  if (result._tag === "Left") return yield* result.left
  if (cleanup.length) return yield* new AssertionFailure({ message: "Pi terminal cleanup failed" })
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive, NativeTerminalDriver.pipe(Layer.provide(BunContext.layer))])))
