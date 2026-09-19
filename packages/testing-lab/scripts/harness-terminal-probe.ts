import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Schema } from "effect"
import { dirname, join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { piTerminal } from "../src/harnesses/pi-terminal"
import { openCodeModelName, openCodeTerminal } from "../src/harnesses/opencode-terminal"
import { HarnessTerminalReceipt } from "../src/harnesses/terminal"
import { NativeTerminalDriver } from "../src/terminal"
import { ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { AssertionFailure } from "../src/domain"

BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT"), executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const harness = yield* Config.literal("pi", "opencode")("LAB_PROBE_HARNESS")
  const runtime = yield* Config.string("LAB_TERMINAL_NODE_EXECUTABLE"), client = yield* Config.string(`LAB_${harness.toUpperCase()}_EXECUTABLE`)
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const port = yield* Config.integer("LAB_PROBE_PORT").pipe(Config.withDefault(11339))
  const fs = yield* FileSystem.FileSystem
  const evidence = yield* fs.makeTempDirectory({ directory: root, prefix: `${harness}-terminal-` })
  const state = yield* fs.makeTempDirectoryScoped({ prefix: "lab-terminal-desktop-" })
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
    yield* desktop.connect(harness)
    const home = join(root, "profile", "harness-home")
    const first = crypto.randomUUID().replaceAll("-", "").slice(0, 8), second = crypto.randomUUID().replaceAll("-", "").slice(0, 8)
    const config = { runtime, executable: client, cwd: evidence, evidence: join(evidence, "terminal"),
      environment: { ...environment, HOME: home, USERPROFILE: home, PI_CODING_AGENT_DIR: join(home, ".pi", "agent"),
        XDG_CONFIG_HOME: join(home, ".config"), XDG_DATA_HOME: join(home, ".local", "share"), XDG_CACHE_HOME: join(home, ".cache"), XDG_STATE_HOME: join(home, ".local", "state") }, model, initialModel: model,
      interrupt: { prompt: `First concatenate READY and ${first} without a space and print that word. Then count from 1 to 10000, one number per line. Do not use tools.`, expected: `READY${first}` },
      recovery: { prompt: `Concatenate DONE and ${second} without a space. Reply only with the resulting word. Do not use tools.`, expected: `DONE${second}` },
    }
    if (harness === "pi") return yield* piTerminal(config, message => { cleanup.push(message) })
    const name = yield* openCodeModelName(home, model)
    return yield* openCodeTerminal({ ...config, modelName: name, initialModelName: name }, message => { cleanup.push(message) })
  }).pipe(Effect.provide(playwrightDesktop({ mode: "isolated", executable, profile: join(root, "profile"), evidence: join(evidence, "app"), port, environment })), Effect.either)
  const report = Schema.Struct({ passed: Schema.Boolean, detail: Schema.String, cleanup: Schema.Array(Schema.String),
    receipts: Schema.Array(HarnessTerminalReceipt) })
  yield* fs.writeFileString(join(evidence, "report.json"), yield* Schema.encode(Schema.parseJson(report))({
    passed: result._tag === "Right" && cleanup.length === 0, cleanup,
    detail: result._tag === "Right" ? `${harness} keyboard selection, real generation, native abort receipt, follow-up and normal exit` : String(result.left),
    receipts: result._tag === "Right" ? [result.right] : [],
  }))
  yield* Effect.logInfo(`${harness} terminal evidence: ${evidence}`)
  if (result._tag === "Left") return yield* result.left
  if (cleanup.length) return yield* new AssertionFailure({ message: "Harness terminal cleanup failed" })
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive, NativeTerminalDriver.pipe(Layer.provide(BunContext.layer))])))
