import { FetchHttpClient, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"
import { bundledCliTests, CliTests } from "../src/suites/cli"
import { EndpointTests, endpointTests } from "../src/suites/endpoint"
import { ProcessExecutorLive } from "../src/process"
import { connectionFixture } from "../src/harnesses/connection-fixture"
import { assertRuntime } from "../src/runtime"
import { AssertionFailure } from "../src/domain"

const run = Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const cli = yield* Config.string("LAB_PROBE_BUNDLED_CLI")
  const version = yield* Config.string("LAB_PROBE_VERSION")
  const toolsPath = yield* Config.string("LAB_PROBE_TOOLS_PATH")
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const fs = yield* FileSystem.FileSystem
  const environment: Record<string, string> = { ...Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "LOCALAPPDATA", "APPDATA", "SystemRoot", "TEMP"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : [])),
    MAGNITUDE_DEV_DATA_DIR: join(root, "profile"), MAGNITUDE_DEV_PORT: "11279", MAGNITUDE_SHELL_ENV_INHERITED: "1" }
  environment.PATH = toolsPath + (process.platform === "win32" ? ";" : ":") + (environment.PATH ?? "")
  const connectionFixtures = yield* Effect.forEach(["pi", "opencode", "hermes"] as const, harness =>
    connectionFixture(join(root, "profile", "harness-home"), harness, "http://127.0.0.1:11279/inference/v1"))
  const checks: { name: string; passed: boolean; detail: string }[] = []
  const program = Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    const tests = yield* CliTests
    const endpoint = yield* EndpointTests
    yield* desktop.host()
    yield* desktop.ready()
    yield* desktop.search(model)
    yield* desktop.load(model)
    const steps = [ ["Version", tests.version], ["Help, hardware, service", tests.inspect], ["Cached pull, stop, reload", tests.modelLifecycle],
      ["Generation after CLI reload", endpoint.generate.pipe(Effect.asVoid)], ["Pi connection lifecycle", tests.connections("pi", connectionFixtures[0]!.inspect)],
      ["OpenCode connection lifecycle", tests.connections("opencode", connectionFixtures[1]!.inspect)], ["Hermes connection lifecycle", tests.connections("hermes", connectionFixtures[2]!.inspect)],
      ["Invalid inputs", tests.invalid], ["Embedded runtime without developer PATH", tests.nativeRuntime] ] as const
    for (const [name, test] of steps) {
      const result = yield* test.pipe(Effect.either)
      checks.push({ name, passed: result._tag === "Right", detail: result._tag === "Right" ? "Passed" : String(result.left) })
    }
  }).pipe(Effect.provide(Layer.mergeAll(playwrightDesktop({ mode: "isolated", executable, profile: join(root, "profile"), evidence: join(root, "cli-ui-evidence"), port: 11279, environment }),
    bundledCliTests({ executable: cli, version, model, evidence: join(root, "cli-evidence"), environment }),
    endpointTests("http://127.0.0.1:11279", model).pipe(Layer.provide(FetchHttpClient.layer)))))
  const result = yield* program.pipe(Effect.either)
  if (result._tag === "Left") checks.push({ name: "Application setup", passed: false, detail: String(result.left) })
  yield* fs.writeFileString(join(root, "cli-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Array(Schema.Struct({ name: Schema.String, passed: Schema.Boolean, detail: Schema.String }))))(checks))
  if (checks.some(c => !c.passed)) return yield* new AssertionFailure({ message: "Bundled CLI acceptance failed; inspect cli-report.json" })
})
BunRuntime.runMain(run.pipe(Effect.provide(Layer.merge(BunContext.layer, ProcessExecutorLive))))
