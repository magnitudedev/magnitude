import { Command, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Option, Schema } from "effect"
import { join, resolve } from "node:path"
import { NativeHost, nativeHostLayer } from "../../../daemon-management/src/desktop-native/index"
import { requestApplication } from "../../../daemon-management/src/desktop-native/application-control"

class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const Evidence = Schema.Struct({ version: Schema.String, headlessReady: Schema.Literal(true), queries: Schema.Literal(true), gracefulExit: Schema.Literal(true) })
const run = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const output = resolve(yield* Config.string("MAGNITUDE_INSTALLED_ACCEPTANCE_OUTPUT"))
  const cli = yield* Config.string("MAGNITUDE_INSTALLED_ACCEPTANCE_CLI")
  const addon = yield* Config.string("MAGNITUDE_INSTALLED_ACCEPTANCE_ADDON")
  const version = yield* Config.string("MAGNITUDE_INSTALLED_ACCEPTANCE_VERSION")
  const inference = yield* Config.string("MAGNITUDE_ICN_PATH")
  yield* fs.makeDirectory(output, { recursive: true })
  const stateDirectory = join(output, "profile", "state")
  const environment = { MAGNITUDE_DEV_DATA_DIR: join(output, "profile"), MAGNITUDE_DESKTOP_STATE_DIR: stateDirectory,
    MAGNITUDE_DEV_PORT: "11237", MAGNITUDE_ICN_PATH: inference }
  const native = yield* NativeHost.pipe(Effect.provide(nativeHostLayer(addon)))
  const query = (...args: string[]) => Command.make(cli, ...args).pipe(Command.env(environment), Command.string)
  const initial = yield* query("status")
  if (!/Runtime\s+Stopped/.test(initial) || (yield* query("--version")).trim() !== version) {
    return yield* new AcceptanceFailed({ message: "Expected stopped installation at the selected version" })
  }
  yield* fs.writeFileString(join(output, "initial-status.txt"), initial)
  yield* Effect.acquireUseRelease(
    Command.make(cli, "serve").pipe(Command.env(environment), Command.stdout("inherit"), Command.stderr("inherit"), Command.start),
    serving => Effect.gen(function* () {
      yield* Effect.gen(function* () {
        for (;;) {
          if (!(yield* serving.isRunning)) return yield* new AcceptanceFailed({ message: "Installed server exited before readiness" })
          const status = yield* query("status")
          if (/Runtime\s+Ready/.test(status) && /Owner\s+Headless/.test(status)) {
            yield* fs.writeFileString(join(output, "ready-status.txt"), status)
            break
          }
          yield* Effect.sleep("500 millis")
        }
      }).pipe(Effect.timeout("5 minutes"))
      yield* fs.writeFileString(join(output, "models.txt"), yield* query("models", "status"))
      yield* fs.writeFileString(join(output, "hardware.txt"), yield* query("hardware"))
    }),
    serving => Effect.gen(function* () {
      if (!(yield* serving.isRunning)) return
      const endpoint = yield* native.inspectEndpoint(stateDirectory)
      if (Option.isNone(endpoint)) return yield* new AcceptanceFailed({ message: "Serving owner has no control endpoint" })
      yield* requestApplication(endpoint.value, "Quit")
      if ((yield* serving.exitCode.pipe(Effect.timeout("30 seconds"))) !== 0) {
        return yield* new AcceptanceFailed({ message: "Installed server did not exit gracefully" })
      }
    }).pipe(Effect.orDie),
  )
  const final = yield* query("status")
  if (!/Runtime\s+Stopped/.test(final)) return yield* new AcceptanceFailed({ message: "Installed server remains alive" })
  yield* fs.writeFileString(join(output, "final-status.txt"), final)
  yield* fs.writeFileString(join(output, "result.json"), yield* Schema.encode(Schema.parseJson(Evidence))({ version, headlessReady: true, queries: true, gracefulExit: true }))
  yield* Effect.logInfo("Installed foreground serve and CLI query acceptance passed")
}))
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
