import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Schedule, Schema } from "effect"
import { join } from "node:path"
import { fileArtifactStore } from "../src/artifact-store"
import { findTarget } from "../src/catalog"
import { HostInspector, HostInspectorLive } from "../src/host-inspector"
import { HostObservation } from "../src/hardware"
import { prepareUpdateConsumer } from "../src/update-consumer"
import { UpdateAcceptance } from "../src/update-acceptance"
import { Installer, nativeInstaller } from "../src/installer"
import { desktopSession } from "../src/desktop-session"
import { ProcessExecutorLive } from "../src/process"
import { verifyUpdateBaseline } from "../src/suites/update"
import { AssertionFailure, Digest, TargetId } from "../src/domain"
import { requestApplication } from "../../daemon-management/src/desktop-native/application-control"
import { assertServiceExited, LabProcessId } from "../src/application-identity"

/** Native updater probe against an explicitly built private pair. Does not claim full source qualification. */
const program = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* Config.string("LAB_UPDATE_PROBE_ROOT")
  const input = yield* Config.string("LAB_UPDATE_PROBE_MANIFEST")
  if (process.platform !== "darwin" || process.arch !== "arm64") return yield* new AssertionFailure({ message: "This local probe owns only an isolated Apple Silicon bundle" })
  const target = yield* findTarget(TargetId.make(yield* Config.string("LAB_UPDATE_PROBE_TARGET")))
  const host = yield* (yield* HostInspector).inspect(target)
  yield* fs.makeDirectory(root, { recursive: true, mode: 0o700 })
  const evidence = yield* fs.makeTempDirectory({ directory: root, prefix: "journey-" })
  yield* fs.writeFileString(join(evidence, "host.json"), yield* Schema.encode(Schema.parseJson(HostObservation))(host))
  const acceptance = yield* fs.readFileString(input).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(UpdateAcceptance))))
  const cleanup: string[] = []
  const result = yield* Effect.scoped(Effect.gen(function* () {
    const { fixture, pair } = yield* prepareUpdateConsumer(acceptance, target, join(evidence, "inputs"))
    const state = yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-up-state-" }).pipe(Effect.flatMap(fs.realPath))
    const endpoint = join(state, "application.sock")
    const profile = join(evidence, "profile"), port = 11439
    const environment = { PATH: process.env.PATH ?? "", HOME: join(evidence, "home"), TMPDIR: process.env.TMPDIR ?? "/tmp", USER: process.env.USER ?? "",
      MAGNITUDE_DEV_DATA_DIR: profile, MAGNITUDE_DEV_PORT: String(port), MAGNITUDE_DESKTOP_STATE_DIR: state,
      NODE_EXTRA_CA_CERTS: fixture.caPath, MAGNITUDE_SHELL_ENV_INHERITED: "1" }
    yield* fs.makeDirectory(environment.HOME, { recursive: true, mode: 0o700 })
    return yield* Effect.gen(function* () {
      const installer = yield* Installer
      const app = yield* Effect.acquireRelease(installer.install(pair.previous), installed => installer.uninstall(installed).pipe(
        Effect.catchAll(error => Effect.sync(() => { cleanup.push(error.message) }))))
      // A replacement launched by the updater is not owned by Playwright's retired process.
      yield* Effect.addFinalizer(() => requestApplication(endpoint, "Quit").pipe(Effect.flatMap(owner =>
        assertServiceExited(LabProcessId.make(owner.service._tag === "Ready" ? owner.service.health.pid : owner.pid)).pipe(
          Effect.retry(Schedule.spaced("200 millis").pipe(Schedule.intersect(Schedule.recurs(50)))))),
        Effect.catchTag("ApplicationControlUnavailable", () => Effect.void),
        Effect.catchAll(error => Effect.sync(() => { cleanup.push(error.message) }))))
      const session = yield* desktopSession({ mode: "isolated", executable: app.executable, profile, port, environment,
        evidence: join(evidence, "desktop") }, detail => { cleanup.push(detail) })
      yield* verifyUpdateBaseline(session, pair.previous.version)
      yield* Effect.logInfo("Baseline installation and persisted settings passed")
      const artifact = { path: pair.update.path, version: pair.candidate.version, target: { os: "darwin" as const, arch: "arm64" as const, package: "mac-zip" as const },
        bytes: pair.update.artifact.bytes, sha256: Digest.make(pair.update.artifact.sha256) }
      yield* fixture.publish(artifact, "Corrupt")
      const driver = yield* session.driver
      yield* driver.updates.action("check")
      yield* driver.updates.wait("Available")
      yield* driver.updates.action("download")
      yield* driver.updates.wait("Failed")
      if ((yield* driver.host()) !== pair.previous.version) return yield* new AssertionFailure({ message: "Corrupt update replaced the running app" })
      yield* Effect.logInfo("Corrupt update was rejected")
      yield* fixture.publish(artifact)
      yield* driver.updates.action("check")
      yield* driver.updates.wait("Available")
      yield* driver.updates.action("download")
      yield* driver.updates.wait("Ready")
      const before = yield* driver.identity()
      yield* driver.restartForUpdate()
      yield* session.stop
      const after = yield* requestApplication(endpoint, "Observe").pipe(Effect.flatMap(owner =>
        owner.service._tag === "Ready" && owner.pid !== before.applicationPid && owner.service.health.pid !== before.servicePid && owner.service.health.version === pair.candidate.version
          ? Effect.succeed(owner) : Effect.fail(new AssertionFailure({ message: "Updater has not launched a new ready owner" }))),
        Effect.retry(Schedule.spaced("500 millis").pipe(Schedule.intersect(Schedule.recurs(120)))))
      yield* fs.writeFileString(join(evidence, "automatic-relaunch.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))(after), { mode: 0o600 })
      yield* requestApplication(endpoint, "Quit")
      yield* assertServiceExited(LabProcessId.make(after.service._tag === "Ready" ? after.service.health.pid : after.pid)).pipe(
        Effect.retry(Schedule.spaced("200 millis").pipe(Schedule.intersect(Schedule.recurs(50)))))
      const reopened = yield* session.driver
      if ((yield* reopened.host()) !== pair.candidate.version) return yield* new AssertionFailure({ message: "Updated application version differs from admitted candidate" })
      yield* reopened.verifyTheme("dark")
      yield* reopened.ready()
      yield* Effect.logInfo("Normal app update, automatic new owner and retained settings passed")
    }).pipe(Effect.provide(nativeInstaller({ disposable: false, root: join(evidence, "installation"), environment })))
  })).pipe(Effect.either)
  const prepared = join(evidence, "profile", "updates", "update.json")
  if (yield* fs.exists(prepared)) yield* fs.copyFile(prepared, join(evidence, "prepared-update.json"))
  yield* fs.writeFileString(join(evidence, "report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ passed: Schema.Boolean, detail: Schema.String, cleanup: Schema.Array(Schema.String) })))({ passed: result._tag === "Right" && cleanup.length === 0,
    detail: result._tag === "Left" ? String(result.left) : "Native update probe passed", cleanup }), { mode: 0o600 })
  yield* Effect.logInfo(`Update evidence: ${evidence}`)
  if (result._tag === "Left") return yield* result.left
  if (cleanup.length) return yield* new AssertionFailure({ message: "Update probe cleanup failed" })
}))
const dependencies = Layer.merge(BunContext.layer, ProcessExecutorLive)
BunRuntime.runMain(Effect.gen(function* () {
  const objects = yield* Config.string("LAB_UPDATE_PROBE_OBJECTS")
  return yield* program.pipe(Effect.provide(Layer.mergeAll(dependencies,
    HostInspectorLive.pipe(Layer.provide(dependencies)), fileArtifactStore(objects).pipe(Layer.provide(dependencies)))))
}))
