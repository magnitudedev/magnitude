import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { desktopSession } from "../src/desktop-session"
import { verifyServiceOwnership } from "../src/suites/service"
import { ApplicationIdentity } from "../src/application-identity"
import { bundledCliTests } from "../src/suites/cli"
import { ProcessExecutorLive } from "../src/process"
import { AssertionFailure } from "../src/domain"
import { assertRuntime } from "../src/runtime"

BunRuntime.runMain(Effect.gen(function* () {
  yield* assertRuntime
  const root = yield* Config.string("LAB_PROBE_ROOT")
  const executable = yield* Config.string("LAB_PROBE_EXECUTABLE")
  const cli = yield* Config.string("LAB_PROBE_BUNDLED_CLI")
  const version = yield* Config.string("LAB_PROBE_VERSION")
  const model = yield* Config.string("LAB_PROBE_MODEL_ID")
  const toolsPath = yield* Config.string("LAB_PROBE_TOOLS_PATH")
  const port = yield* Config.integer("LAB_PROBE_PORT").pipe(Config.withDefault(11319))
  const fs = yield* FileSystem.FileSystem
  const state = yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-connections-" })
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  environment.PATH = `${toolsPath}:${environment.PATH ?? ""}`
  environment.MAGNITUDE_DESKTOP_STATE_DIR = state
  environment.MAGNITUDE_DEV_DATA_DIR = join(root, "profile")
  environment.MAGNITUDE_DEV_PORT = String(port)
  environment.MAGNITUDE_SHELL_ENV_INHERITED = "1"
  const cleanupErrors: string[] = []
  let observations: ApplicationIdentity[] = []
  const result = yield* Effect.scoped(Effect.gen(function* () {
    const session = yield* desktopSession({ mode: "isolated", executable, profile: join(root, "profile"), evidence: join(root, "evidence"), port, environment }, detail => { cleanupErrors.push(detail) })
    observations = yield* verifyServiceOwnership(session)
  })).pipe(Effect.provide(bundledCliTests({ executable: cli, version, model, evidence: join(root, "cli-evidence"), environment })), Effect.either)
  yield* fs.writeFileString(join(root, "ownership-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ passed: Schema.Boolean, detail: Schema.String, cleanupErrors: Schema.Array(Schema.String), observations: Schema.Array(ApplicationIdentity) })))({
    cleanupErrors, passed: result._tag === "Right" && cleanupErrors.length === 0, detail: result._tag === "Right" ? "Six repeated starts retained ownership across three app lifetimes; previous service processes exited" : String(result.left), observations,
  }))
  if (result._tag === "Left") return yield* result.left
  if (cleanupErrors.length) return yield* new AssertionFailure({ message: "Ownership probe cleanup failed" })
}).pipe(Effect.scoped, Effect.provide(Layer.merge(BunContext.layer, ProcessExecutorLive))))
