import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { piSession, PiTurn } from "../src/harnesses/pi"
import { fileFixture } from "../src/harnesses/file-fixture"
import { assertRuntime } from "../src/runtime"
import { AssertionFailure } from "../src/domain"

const run = Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const pi = yield* Config.string("LAB_PROBE_PI")
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const fs = yield* FileSystem.FileSystem
  const task = yield* fileFixture(root)
  const fixture = task.directory
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.PATH = `${join(pi, "..")}:/usr/local/bin:/usr/bin:/bin:${environment.PATH ?? ""}`
  const turns: PiTurn[] = []
  const program = Effect.scoped(Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.host()
    yield* desktop.ready()
    yield* desktop.search(model)
    yield* desktop.load(model)
    yield* desktop.connect("pi")
    yield* desktop.screenshot("pi-connected")
    const sessionConfig = { executable: pi,
      args: ["--mode", "rpc", "--provider", "magnitude", "--model", model, "--thinking", "off", "--tools", "read,write,edit", "--offline", "--session", join(root, `pi-session-${crypto.randomUUID()}.jsonl`)],
      cwd: fixture, environment: { ...environment, HOME: join(root, "profile", "harness-home"), PI_CODING_AGENT_DIR: join(root, "profile", "harness-home", ".pi", "agent") },
      stdoutLog: join(root, "pi-rpc.jsonl"), stderrLog: join(root, "pi-stderr.log") }
    yield* Effect.scoped(Effect.gen(function* () {
    const session = yield* piSession(sessionConfig, model)
    turns.push(yield* session.prompt("Reply with exactly HELLO. Do not use tools."))
    if (turns[0]!.text.trim() !== "HELLO") return yield* new AssertionFailure({ message: "Pi did not return the expected greeting" })
    turns.push(yield* session.prompt("Use the read tool to read message.txt, then use the edit tool to replace before with after. Do not change any other file. Reply DONE after the edit."))
    if (!(turns[1]!.tools.some(t => t.name === "read") && turns[1]!.tools.some(t => t.name === "edit"))) return yield* new AssertionFailure({ message: "Pi did not perform the requested read/edit tool sequence" })
    yield* task.verify
    }))
    yield* Effect.scoped(Effect.gen(function* () {
      const resumed = yield* piSession({ ...sessionConfig, stdoutLog: join(root, "pi-resume-rpc.jsonl"), stderrLog: join(root, "pi-resume-stderr.log") }, model)
      if ((yield* resumed.state()).sessionId !== turns[0]!.sessionId) return yield* new AssertionFailure({ message: "Pi resumed a different session" })
      turns.push(yield* resumed.prompt("What word did you replace before with in our previous turn? Reply with that single word. Do not use tools."))
      if (turns[2]!.text.trim() !== "after") return yield* new AssertionFailure({ message: "Pi did not retain the previous conversation after process restart" })
    }))
    yield* desktop.disconnect("pi")
    yield* desktop.screenshot("pi-disconnected")
  })).pipe(Effect.provide(playwrightDesktop({ executable, profile: join(root, "profile"), evidence: join(root, "pi-evidence"), port: 11279, environment })))
  const outcome = yield* program.pipe(Effect.either)
  yield* fs.writeFileString(join(root, "pi-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ turns: Schema.Array(PiTurn), passed: Schema.Boolean, detail: Schema.String })))({
    turns, passed: outcome._tag === "Right", detail: outcome._tag === "Right" ? "Connected through app, streamed generation, real read/edit, disconnected" : String(outcome.left),
  }))
  if (outcome._tag === "Left") return yield* outcome.left
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
