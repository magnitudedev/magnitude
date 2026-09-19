import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { ReleaseArtifactSchema } from "@magnitudedev/release/contracts"
import { Cause, Config, Effect, Exit, Layer, Option, Schema } from "effect"
import { join, resolve } from "node:path"
import { Candidate } from "../src/candidate"
import { findTarget } from "../src/catalog"
import { desktopSession } from "../src/desktop-session"
import { AssertionFailure, Digest, TargetId } from "../src/domain"
import { HostInspector, HostInspectorLive } from "../src/host-inspector"
import { HostObservation } from "../src/hardware"
import { installationSession } from "../src/installation-session"
import { nativeInstaller } from "../src/installer"
import { command, CommandOutput, ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { extractSource, snapshotSource } from "../src/snapshot"
import { updateFixture } from "../src/update-fixture"

/** Native acquisition gate; installation/relaunch remains a separate probe phase. */
const run = Effect.gen(function* () {
  yield* assertRuntime
  if (process.platform !== "darwin" || process.arch !== "arm64") return yield* new AssertionFailure({ message: "This probe requires an Apple Silicon Mac" })
  const fs = yield* FileSystem.FileSystem
  const root = resolve(yield* Config.string("LAB_UPDATE_PROBE_ROOT"))
  if (yield* fs.exists(root)) return yield* new AssertionFailure({ message: "Update probe requires a fresh root" })
  const target = yield* findTarget(TargetId.make(yield* Config.string("LAB_UPDATE_PROBE_TARGET")))
  const host = yield* (yield* HostInspector).inspect(target)
  yield* fs.makeDirectory(root, { recursive: true, mode: 0o700 })
  yield* fs.writeFileString(join(root, "host.json"), yield* Schema.encode(Schema.parseJson(HostObservation))(host))
  const previousVersion = "0.1.3", nextVersion = "0.1.4"
  const source = yield* snapshotSource(resolve(import.meta.dir, "../../.."), join(root, "objects"))
  const workspace = join(root, "source"), home = join(root, "home")
  yield* fs.makeDirectory(home, { mode: 0o700 })
  yield* extractSource(source.manifest, join(root, "objects"), workspace)
  const environment = Object.fromEntries(["PATH", "TMPDIR", "USER", "LOGNAME"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const buildEnv = { ...environment, HOME: home, XDG_CONFIG_HOME: join(home, ".config"), XDG_CACHE_HOME: join(home, ".cache"), MAGNITUDE_APPLE_DISTRIBUTION: "adhoc" }
  const invoke = (label: string, args: readonly string[], extra: Readonly<Record<string, string>> = {}) => Effect.gen(function* () {
    yield* Effect.logInfo(`Update probe: ${label}`)
    const result = yield* command(process.execPath, args, { cwd: Option.some(workspace), inheritEnv: false, env: { ...buildEnv, ...extra }, timeoutMs: 30 * 60_000, maxOutputBytes: 32 * 1024 * 1024 })
    yield* fs.writeFileString(join(root, `${label}.json`), yield* Schema.encode(Schema.parseJson(CommandOutput))(result))
    if (result.exitCode !== 0) return yield* new AssertionFailure({ message: `${label} failed; inspect ${label}.json` })
  })
  const artifacts = (directory: string) => Effect.gen(function* () {
    const records = (yield* fs.readDirectory(directory)).filter(name => name.endsWith(".artifact.json"))
    return yield* Effect.forEach(records, name => fs.readFileString(join(directory, name)).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(ReleaseArtifactSchema)))))
  })
  const cleanupErrors: string[] = []
  const result = yield* Effect.scoped(Effect.gen(function* () {
    const fixture = yield* updateFixture(root)
    yield* invoke("dependencies", ["install", "--frozen-lockfile"])
    for (const [label, version] of [["previous", previousVersion], ["candidate", nextVersion]] as const) {
      yield* invoke(label, ["packages/release/scripts/acceptance/build-desktop.ts"], {
        MAGNITUDE_UPDATE_ACCEPTANCE_CONFIG: fixture.configPath, MAGNITUDE_ACCEPTANCE_VERSION: version,
        MAGNITUDE_ACCEPTANCE_OUTPUT: join(root, label),
      })
    }
    const previous = yield* artifacts(join(root, "previous/artifacts"))
    const next = yield* artifacts(join(root, "candidate/artifacts"))
    const dmg = previous.find(artifact => artifact.filename.endsWith(".dmg"))
    const zip = next.find(artifact => artifact.filename.endsWith(".zip"))
    if (!dmg || !zip) return yield* new AssertionFailure({ message: "Acceptance pair did not produce its native installer and update ZIP" })
    const candidate = Candidate.make({ artifact: dmg, version: previousVersion, target, path: join(root, "previous/artifacts", dmg.filename) })
    const installed = yield* installationSession(candidate, detail => { cleanupErrors.push(detail) })
    const app = yield* installed.get
    const state = yield* fs.makeTempDirectoryScoped({ prefix: "ml-update-state-" })
    const session = yield* desktopSession({ executable: app.executable, profile: join(root, "profile"), evidence: join(root, "evidence"), port: 11449,
      environment: { ...environment, HOME: home, NODE_EXTRA_CA_CERTS: fixture.caPath, MAGNITUDE_DESKTOP_STATE_DIR: state },
    }, detail => { cleanupErrors.push(detail) })
    const driver = yield* session.driver
    if ((yield* driver.host()) !== previousVersion) return yield* new AssertionFailure({ message: "Previous installed app has the wrong version" })
    yield* driver.theme("dark")
    yield* driver.updates.automatic(false)
    yield* fixture.publish({ path: join(root, "candidate/artifacts", zip.filename), version: nextVersion,
      target: { os: "darwin", arch: "arm64", package: "mac-zip" }, bytes: zip.bytes, sha256: Digest.make(zip.sha256) })
    yield* driver.updates.action("check")
    const available = yield* driver.updates.wait("Available")
    if (!Option.contains(available.version, nextVersion)) return yield* new AssertionFailure({ message: "App offered the wrong update version" })
    yield* driver.updates.action("download")
    const ready = yield* driver.updates.wait("Ready")
    if (!Option.contains(ready.version, nextVersion)) return yield* new AssertionFailure({ message: "App prepared the wrong update version" })
    yield* driver.screenshot("private-update-ready")
    yield* driver.updates.action("discard")
    yield* driver.updates.wait("Idle")
    yield* driver.quit()
    yield* session.stop
  })).pipe(Effect.provide(nativeInstaller({ disposable: false, root: join(root, "installed"), environment: { ...environment, HOME: home } })), Effect.exit)
  const passed = Exit.isSuccess(result) && cleanupErrors.length === 0
  yield* fs.writeFileString(join(root, "update-report.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({
    passed: Schema.Boolean, sourceDigest: Digest, detail: Schema.String, cleanupErrors: Schema.Array(Schema.String),
  })))({ passed, sourceDigest: source.digest, cleanupErrors,
    detail: Exit.isFailure(result) ? Cause.pretty(result.cause) : "Installed older acceptance app checked, downloaded, verified and discarded the newer ZIP over private HTTPS; native replacement/relaunch and production trust are not qualified",
  }))
  if (!passed) return yield* new AssertionFailure({ message: "Native update acquisition failed; inspect update-report.json" })
})
BunRuntime.runMain(run.pipe(Effect.provide(HostInspectorLive.pipe(Layer.provideMerge(Layer.merge(BunContext.layer, ProcessExecutorLive))))))
