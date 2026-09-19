import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Option, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { openCode, OpenCodeTurn } from "../src/harnesses/opencode"
import { ProcessExecutorLive } from "../src/process"
import { fileFixture } from "../src/harnesses/file-fixture"
import { assertRuntime } from "../src/runtime"
import { AssertionFailure } from "../src/domain"

const run = Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const client = yield* Config.string("LAB_PROBE_OPENCODE")
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const fs = yield* FileSystem.FileSystem
  const task = yield* fileFixture(root, { "opencode.json": yield* Schema.encode(Schema.parseJson(Schema.Struct({ $schema: Schema.Literal("https://opencode.ai/config.json"), permission: Schema.Record({ key: Schema.String, value: Schema.Literal("allow", "deny") }) })))({ $schema: "https://opencode.ai/config.json", permission: { "*": "deny", read: "allow", edit: "allow" } }) })
  const fixture = task.directory
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.PATH = `${join(client, "..")}:/usr/local/bin:/usr/bin:/bin:${environment.PATH ?? ""}`
  const home = join(root, "profile", "harness-home")
  const turns: OpenCodeTurn[] = []
  const program = Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.host()
    yield* desktop.ready()
    yield* desktop.search(model)
    yield* desktop.load(model)
    yield* desktop.connect("opencode")
    yield* desktop.screenshot("opencode-connected")
    const client = yield* openCode({ executable: yield* Config.string("LAB_PROBE_OPENCODE"), cwd: fixture, model,
      evidence: join(root, "opencode-events"), environment: { ...environment, HOME: home, XDG_CONFIG_HOME: join(home, ".config"),
        XDG_DATA_HOME: join(home, ".local", "share"), XDG_CACHE_HOME: join(home, ".cache"), XDG_STATE_HOME: join(home, ".local", "state") } })
    turns.push(yield* client.prompt("Reply with exactly HELLO. Do not use tools."))
    if (turns[0]!.text.trim() !== "HELLO") return yield* new AssertionFailure({ message: "OpenCode did not return the expected greeting" })
    turns.push(yield* client.prompt("Use the read tool to read message.txt, then use the edit tool to replace before with after. Do not change any other file. Reply DONE after the edit.", Option.some(turns[0]!.sessionId)))
    if (!(turns[1]!.tools.includes("read") && turns[1]!.tools.includes("edit"))) return yield* new AssertionFailure({ message: "OpenCode did not perform the requested read/edit tool sequence" })
    yield* task.verify
    turns.push(yield* client.prompt("What word did you replace before with in our previous turn? Reply with that single word. Do not use tools.", Option.some(turns[0]!.sessionId)))
    if (turns[2]!.text.trim() !== "after") return yield* new AssertionFailure({ message: "OpenCode did not retain the previous conversation after process restart" })
    yield* desktop.disconnect("opencode")
    yield* desktop.screenshot("opencode-disconnected")
  }).pipe(Effect.provide(playwrightDesktop({ executable, profile: join(root, "profile"), evidence: join(root, "opencode-evidence"), port: 11279, environment })))
  const outcome = yield* program.pipe(Effect.either)
  yield* fs.writeFileString(join(root, "opencode-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ turns: Schema.Array(OpenCodeTurn), passed: Schema.Boolean, detail: Schema.String })))({
    turns, passed: outcome._tag === "Right", detail: outcome._tag === "Right" ? "Connected through app, generated, real read/edit, resumed session, disconnected" : String(outcome.left),
  }))
  if (outcome._tag === "Left") return yield* Effect.fail(outcome.left)
})
BunRuntime.runMain(run.pipe(Effect.provide(Layer.merge(BunContext.layer, ProcessExecutorLive))))
