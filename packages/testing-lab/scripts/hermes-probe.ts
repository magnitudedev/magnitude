import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Option, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { hermes, HermesTurn } from "../src/harnesses/hermes"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { fileFixture } from "../src/harnesses/file-fixture"
import { assertRuntime } from "../src/runtime"
import { AssertionFailure } from "../src/domain"

const run = Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const client = yield* Config.string("LAB_PROBE_HERMES")
  const selectModel = yield* Config.boolean("LAB_PROBE_HERMES_SET_MODEL").pipe(Config.withDefault(false))
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const fs = yield* FileSystem.FileSystem
  const task = yield* fileFixture(root)
  const fixture = task.directory
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.PATH = `${join(client, "..")}:/usr/local/bin:/usr/bin:/bin:${environment.PATH ?? ""}`
  const home = join(root, "profile", "harness-home")
  const turns: HermesTurn[] = []
  const program = Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.host()
    yield* desktop.ready()
    yield* desktop.search(model)
    yield* desktop.load(model)
    yield* desktop.connect("hermes")
    yield* desktop.screenshot("hermes-connected")
    if (selectModel) {
      const bundledCli = yield* Config.string("LAB_PROBE_BUNDLED_CLI")
      const selected = yield* checkedCommand(bundledCli, ["connections", "add", "hermes", "--set-model", model], {
        cwd: Option.some(fixture), env: { ...environment, MAGNITUDE_DEV_DATA_DIR: join(root, "profile"), MAGNITUDE_DEV_PORT: "11279", MAGNITUDE_SHELL_ENV_INHERITED: "1" }, inheritEnv: false,
      })
      yield* fs.writeFileString(join(root, "hermes-model-selection.txt"), selected.stdout)
    }
    const client = yield* hermes({ executable: yield* Config.string("LAB_PROBE_HERMES"), cwd: fixture, model,
      evidence: join(root, selectModel ? "hermes-selected-events" : "hermes-events"), environment: { ...environment, HOME: home, HERMES_HOME: join(home, ".hermes"), XDG_CONFIG_HOME: join(home, ".config"),
        XDG_DATA_HOME: join(home, ".local", "share"), XDG_CACHE_HOME: join(home, ".cache"), XDG_STATE_HOME: join(home, ".local", "state") } })
    turns.push(yield* client.prompt("Reply with exactly HELLO. Do not use tools."))
    if (turns[0]!.text.trim() !== "HELLO") return yield* new AssertionFailure({ message: "Hermes did not return the expected greeting" })
    turns.push(yield* client.prompt("Use the read_file tool to read message.txt, then use the patch tool to replace before with after. Do not change any other file. Reply DONE after the edit.", Option.some(turns[0]!.sessionId)))
    if (!(turns[1]!.tools.includes("read_file") && turns[1]!.tools.includes("patch"))) return yield* new AssertionFailure({ message: "Hermes did not perform the requested read/edit tool sequence" })
    yield* task.verify
    turns.push(yield* client.prompt("What word did you replace before with in our previous turn? Reply with that single word. Do not use tools.", Option.some(turns[0]!.sessionId)))
    if (turns[2]!.text.trim() !== "after") return yield* new AssertionFailure({ message: "Hermes did not retain the previous conversation after process restart" })
    yield* desktop.disconnect("hermes")
    yield* desktop.screenshot("hermes-disconnected")
  }).pipe(Effect.provide(playwrightDesktop({ executable, profile: join(root, "profile"), evidence: join(root, "hermes-evidence"), port: 11279, environment })))
  const outcome = yield* program.pipe(Effect.either)
  yield* fs.writeFileString(join(root, selectModel ? "hermes-selected-report.json" : "hermes-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ turns: Schema.Array(HermesTurn), passed: Schema.Boolean, detail: Schema.String })))({
    turns, passed: outcome._tag === "Right", detail: outcome._tag === "Right" ? "Connected through app, generated, real read/edit, resumed session, disconnected" : String(outcome.left),
  }))
  if (outcome._tag === "Left") return yield* Effect.fail(outcome.left)
})
BunRuntime.runMain(run.pipe(Effect.provide(Layer.merge(BunContext.layer, ProcessExecutorLive))))
