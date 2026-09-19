import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { configuredHarnessTools, harnessSuite, HarnessTurn } from "../src/harnesses/suite"
import { ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { AssertionFailure, Harness } from "../src/domain"

BunRuntime.runMain(Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const toolsPath = yield* Config.string("LAB_PROBE_TOOLS_PATH")
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const port = yield* Config.integer("LAB_PROBE_PORT").pipe(Config.withDefault(11329))
  const fs = yield* FileSystem.FileSystem
  const selected = yield* Schema.decodeUnknown(Schema.Array(Harness))((yield* Config.string("LAB_PROBE_HARNESSES").pipe(Config.withDefault("pi,opencode,hermes"))).split(","))
  const suiteRoot = yield* fs.makeTempDirectory({ directory: root, prefix: "suite-" })
  const state = yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-harness-" })
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.PATH = `${toolsPath}:${environment.PATH ?? ""}`
  environment.MAGNITUDE_DESKTOP_STATE_DIR = state
  const results: { harness: Harness; passed: boolean; detail: string; turns: typeof HarnessTurn.Type[] }[] = []
  yield* Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.host()
    yield* desktop.ready()
    yield* desktop.search(model)
    yield* desktop.load(model)
    for (const harness of selected) {
      const turns: typeof HarnessTurn.Type[] = []
      const result = yield* Effect.gen(function* () {
        yield* desktop.connect(harness)
        const suite = yield* harnessSuite(harness, model, join(suiteRoot, harness), join(root, "profile", "harness-home"), environment)
        turns.push(yield* suite.initial)
        turns.push(yield* suite.recall)
        turns.push(yield* suite.tools)
        turns.push(yield* suite.recall)
      }).pipe(Effect.either)
      results.push({ harness, passed: result._tag === "Right", detail: result._tag === "Right" ? "Generation, follow-up, exact tool edit and persisted recall passed" : String(result.left), turns })
    }
  }).pipe(Effect.provide(playwrightDesktop({ executable, profile: join(root, "profile"), evidence: join(root, "suite-ui"), port, environment })))
  yield* fs.writeFileString(join(root, "harness-suite-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Array(Schema.Struct({ harness: Harness, passed: Schema.Boolean, detail: Schema.String, turns: Schema.Array(HarnessTurn) }))))(results))
  if (results.some(result => !result.passed)) return yield* new AssertionFailure({ message: "Harness suite diagnostic failed; inspect harness-suite-report.json" })
}).pipe(Effect.scoped, Effect.provide([BunContext.layer, ProcessExecutorLive, configuredHarnessTools])))
