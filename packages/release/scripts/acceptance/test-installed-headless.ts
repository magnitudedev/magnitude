import { Command, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Option, Schema, Stream } from "effect"
import { join, resolve } from "node:path"
import { NativeHost, nativeHostLayer } from "../../../daemon-management/src/desktop-native/index"
import { requestApplication } from "../../../daemon-management/src/desktop-native/application-control"

class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const Evidence = Schema.Struct({ version: Schema.String, engineAcquired: Schema.Boolean, offlineCachedStart: Schema.Boolean, headlessReady: Schema.Literal(true), rankingReady: Schema.Boolean, queries: Schema.Literal(true), gracefulExit: Schema.Literal(true) })
const run = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const output = resolve(yield* Config.string("MAGNITUDE_INSTALLED_ACCEPTANCE_OUTPUT"))
  const profile = resolve(yield* Config.string("MAGNITUDE_INSTALLED_ACCEPTANCE_PROFILE"))
  const result = resolve(yield* Config.string("MAGNITUDE_INSTALLED_ACCEPTANCE_RESULT"))
  const cli = yield* Config.string("MAGNITUDE_INSTALLED_ACCEPTANCE_CLI")
  const addon = yield* Config.string("MAGNITUDE_INSTALLED_ACCEPTANCE_ADDON")
  const version = yield* Config.string("MAGNITUDE_INSTALLED_ACCEPTANCE_VERSION")
  const offline = yield* Config.boolean("MAGNITUDE_INSTALLED_ACCEPTANCE_OFFLINE").pipe(Config.withDefault(false))
  const inference = (yield* Config.option(Config.string("MAGNITUDE_ICN_PATH"))).pipe(Option.filter(path => path.trim().length > 0))
  const diagnosticWrite = (name: string, content: string) => fs.writeFileString(join(output, name), content).pipe(
    Effect.catchAll(error => Effect.logWarning(`Could not write acceptance diagnostic ${name}: ${String(error)}`)),
  )
  yield* fs.makeDirectory(output, { recursive: true }).pipe(Effect.catchAll(error => Effect.logWarning(`Could not create acceptance diagnostics directory: ${String(error)}`)))
  const stateDirectory = join(profile, "state")
  const assessmentDiagnostics = join(output, offline ? "offline-assessment-diagnostics.json" : "assessment-diagnostics.json")
  const environment = { MAGNITUDE_DEV_DATA_DIR: profile, MAGNITUDE_DESKTOP_STATE_DIR: stateDirectory,
    MAGNITUDE_DEV_PORT: "11237", MAGNITUDE_RELEASE_BASE_URL: process.env.MAGNITUDE_RELEASE_BASE_URL ?? "https://github.com/magnitudedev/magnitude/releases/download",
    MAGNITUDE_ACCEPTANCE_ASSESSMENT_DIAGNOSTICS: assessmentDiagnostics,
    RUST_LOG: "magnitude_service_server=info",
    ...Option.match(inference, { onNone: () => ({}), onSome: path => ({ MAGNITUDE_ICN_PATH: path }) }) }
  const cachedManifest = join(profile, "releases", "manifests", version, "magnitude-release.json")
  if (offline && !(yield* fs.exists(cachedManifest))) {
    return yield* new AcceptanceFailed({ message: "Offline start has no cached release manifest" })
  }
  const native = yield* NativeHost.pipe(Effect.provide(nativeHostLayer(addon)))
  const query = (...args: string[]) => Effect.scoped(Effect.gen(function* () {
    const child = yield* Command.make(cli, ...args).pipe(Command.env(environment), Command.stderr("inherit"), Command.start)
    const output = yield* child.stdout.pipe(Stream.decodeText(), Stream.runFold("", (text, chunk) => text + chunk))
    const code = yield* child.exitCode
    if (code !== 0) return yield* new AcceptanceFailed({ message: `Installed command ${args.join(" ")} exited ${code}` })
    return output
  })).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new AcceptanceFailed({
    message: `Installed command ${args.join(" ")} did not finish within 30 seconds`,
  }) }))
  const initial = yield* query("status")
  if (!/Runtime\s+Stopped/.test(initial) || (yield* query("--version")).trim() !== version) {
    return yield* new AcceptanceFailed({ message: "Expected stopped installation at the selected version" })
  }
  yield* diagnosticWrite("initial-status.txt", initial)
  yield* Effect.acquireUseRelease(
    Command.make(cli, "serve").pipe(Command.env(environment), Command.stdout("inherit"), Command.stderr("inherit"), Command.start),
    serving => Effect.gen(function* () {
      // Each phase logs its duration and, on timeout, reports the last status it observed.
      const readinessStarted = Date.now()
      let readinessStatus = "(no status observed)"
      yield* Effect.gen(function* () {
        for (;;) {
          if (!(yield* serving.isRunning)) return yield* new AcceptanceFailed({ message: "Installed server exited before readiness" })
          const status = yield* query("status")
          readinessStatus = status
          if (/Runtime\s+Ready/.test(status) && /Owner\s+Headless/.test(status)) {
            yield* diagnosticWrite("ready-status.txt", status)
            break
          }
          yield* Effect.sleep("500 millis")
        }
      }).pipe(Effect.timeoutFail({ duration: "150 seconds", onTimeout: () => new AcceptanceFailed({
        message: `Installed server was not Ready within 150 seconds; last status:\n${readinessStatus}`,
      }) }))
      yield* Effect.logInfo(`Installed server Ready after ${Math.round((Date.now() - readinessStarted) / 1000)} s`)
      if (Option.isNone(inference) && !(yield* fs.exists(cachedManifest))) {
        return yield* new AcceptanceFailed({ message: "Ready service did not acquire the release into the empty profile" })
      }
      if (Option.isNone(inference)) {
        const assessmentStarted = Date.now()
        let catalogStatus = "(no status observed)"
        const ranking = yield* Effect.gen(function* () {
          for (;;) {
            const status = yield* query("catalog", "status")
            if (status !== catalogStatus) yield* Effect.logInfo(`Catalog status after ${Math.round((Date.now() - assessmentStarted) / 1000)} s:\n${status.trim()}`)
            catalogStatus = status
            const counts = /Assessment: Complete - (\d+) of (\d+) models? assessed/.exec(status)
            if (counts && Number(counts[1]) > 0 && counts[1] === counts[2]) {
              yield* diagnosticWrite("catalog-status.txt", status)
              return yield* query("catalog", "recommendations")
            }
            yield* Effect.sleep("500 millis")
          }
        }).pipe(Effect.timeoutFail({ duration: "16 minutes", onTimeout: () => new AcceptanceFailed({
          message: `Catalog assessment did not complete within 16 minutes; last catalog status:\n${catalogStatus}`,
        }) }), Effect.catchAll(error => Effect.gen(function* () {
          const diagnostics = yield* fs.readFileString(assessmentDiagnostics).pipe(Effect.orElseSucceed(() => "(unavailable)"))
          return yield* new AcceptanceFailed({ message: `${error instanceof AcceptanceFailed ? error.message : String(error)}\nAssessment diagnostics:\n${diagnostics}` })
        })))
        yield* Effect.logInfo(`Catalog assessed after ${Math.round((Date.now() - assessmentStarted) / 1000)} s`)
        if (!ranking.includes("Local model recommendations -") && !ranking.includes("No compatible recommendations are available")) {
          return yield* new AcceptanceFailed({ message: "Catalog completed without a settled recommendation result" })
        }
        yield* diagnosticWrite("recommendations.txt", ranking)
      }
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
  yield* diagnosticWrite("final-status.txt", final)
  const evidence = yield* Schema.encode(Schema.parseJson(Evidence))({ version, engineAcquired: Option.isNone(inference) && !offline, offlineCachedStart: offline, headlessReady: true, rankingReady: Option.isNone(inference), queries: true, gracefulExit: true })
  yield* fs.writeFileString(result, evidence)
  yield* diagnosticWrite("result.json", evidence)
  yield* Effect.logInfo("Installed foreground serve and CLI query acceptance passed")
}))
// Bounded by its phase limits; callers cap the whole process above them.
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
