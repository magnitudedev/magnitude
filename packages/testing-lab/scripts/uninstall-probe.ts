import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { ReleaseArtifactSchema } from "@magnitudedev/release/contracts"
import { Cause, Config, Effect, Exit, Layer, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import { join } from "node:path"
import { Candidate } from "../src/candidate"
import { findTarget } from "../src/catalog"
import { TargetId, AssertionFailure } from "../src/domain"
import { nativeInstaller } from "../src/installer"
import { ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { HostInspector, HostInspectorLive } from "../src/host-inspector"
import { HostObservation } from "../src/hardware"
import { installationSession } from "../src/installation-session"
import { desktopSession } from "../src/desktop-session"
import { verifyNativeRemoval } from "../src/suites/uninstall"
import { captureRetainedProfile, RetainedProfile, verifyRetainedProfile } from "../src/retained-profile"

const run = Effect.gen(function* () {
  yield* assertRuntime
  const file = yield* Config.string("LAB_INSTALL_PACKAGE")
  const version = yield* Config.string("LAB_INSTALL_VERSION")
  const root = yield* Config.string("LAB_INSTALL_ROOT")
  const target = yield* findTarget(TargetId.make(yield* Config.string("LAB_INSTALL_TARGET")))
  const disposable = yield* Config.boolean("LAB_INSTALL_DISPOSABLE").pipe(Config.withDefault(false))
  const port = yield* Config.integer("LAB_INSTALL_PORT").pipe(Config.withDefault(11309))
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(root, { recursive: true })
  const host = yield* (yield* HostInspector).inspect(target)
  yield* fs.writeFileString(join(root, "host.json"), yield* Schema.encode(Schema.parseJson(HostObservation))(host))
  const hash = createHash("sha256")
  let bytes = 0
  yield* fs.stream(file).pipe(Stream.runForEach(chunk => Effect.sync(() => { hash.update(chunk); bytes += chunk.byteLength })))
  const artifact = yield* Schema.decodeUnknown(ReleaseArtifactSchema)({ id: `desktop-${target.artifactHost}`, kind: "desktop", host: target.artifactHost,
    filename: `Magnitude.${target.packageFormat}`, sha256: hash.digest("hex"), bytes })
  const candidate = Candidate.make({ artifact, version, target, path: file })
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "LOCALAPPDATA", "APPDATA", "SystemRoot", "TEMP"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const cleanupErrors: string[] = []
  const result = yield* Effect.scoped(Effect.gen(function* () {
    const state = yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-state-" })
    const profile = join(root, "profile")
    const installation = yield* installationSession(candidate, detail => { cleanupErrors.push(detail) })
    const first = yield* installation.get
    const session = yield* desktopSession({ executable: first.executable, profile, evidence: join(root, "evidence"), port,
      environment: { ...environment, MAGNITUDE_DESKTOP_STATE_DIR: state } }, detail => { cleanupErrors.push(detail) })
    const driver = yield* session.driver
    yield* driver.host()
    yield* driver.theme("dark")
    yield* driver.quit()
    yield* session.stop
    const retained = yield* captureRetainedProfile(profile)
    yield* installation.remove
    yield* verifyNativeRemoval(first, environment)
    yield* verifyRetainedProfile(profile, retained)
    yield* fs.writeFileString(join(root, "retained-profile.json"), yield* Schema.encode(Schema.parseJson(RetainedProfile))(retained))
    const second = yield* installation.get
    if (second.root !== first.root) return yield* new AssertionFailure({ message: "Reinstall changed the installation path" })
    const reopened = yield* session.driver
    if ((yield* reopened.host()) !== version) return yield* new AssertionFailure({ message: "Reinstalled application version differs from candidate" })
    yield* reopened.verifyTheme("dark")
    yield* reopened.ready()
    yield* reopened.screenshot("reinstalled")
  })).pipe(Effect.provide(nativeInstaller({ disposable, root: join(root, "application"), environment })), Effect.exit)
  const passed = Exit.isSuccess(result) && cleanupErrors.length === 0
  yield* fs.writeFileString(join(root, "uninstall-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ passed: Schema.Boolean, detail: Schema.String, cleanupErrors: Schema.Array(Schema.String) })))({
    passed, detail: Exit.isFailure(result) ? Cause.pretty(result.cause) : "Native removal deleted the payload and CLI, preserved user data, and reinstall retained appearance and reached service readiness; startup registration is not qualified", cleanupErrors,
  }))
  if (!passed) return yield* new AssertionFailure({ message: "Native uninstall/reinstall failed; inspect uninstall-report.json" })
})
BunRuntime.runMain(run.pipe(Effect.provide(HostInspectorLive.pipe(Layer.provideMerge(Layer.merge(BunContext.layer, ProcessExecutorLive))))))
