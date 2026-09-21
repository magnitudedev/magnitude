import { isWindows } from "../domain"
import { FetchHttpClient, FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Option, Schedule, Schema, Scope } from "effect"
import { join } from "node:path"
import { nativeHostLayer, NativeHost } from "../../../daemon-management/src/desktop-native"
import { updateControlEndpoint } from "../update-control"
import { requestApplication } from "../../../daemon-management/src/desktop-native/application-control"
import { ApplicationSnapshot } from "../../../sdk/src/desktop-host"
import { ApplicationIdentity, assertServiceExited, LabProcessId } from "../application-identity"
import { desktopSession } from "../desktop-session"
import { AssertionFailure, Digest, InfrastructureFailure, Target } from "../domain"
import { installationSession } from "../installation-session"
import { runtimeEnvironment } from "../runtime-release"
import { UpdateAcceptance } from "../update-acceptance"
import { prepareUpdateConsumer } from "../update-consumer"
import { InterruptedUpdateTransfer, UpdateFixtureArtifact } from "../update-fixture"
import { NodeArchiveExtractor } from "../../../release/src/archive"
import { admittedModelFiles, ModelFileReceipt, verifyModelFiles } from "../model-files"
import { UpdatePair } from "../update-pair"
import { EndpointTests, endpointTests, Generation } from "./endpoint"
import { inspectPackageIdentity, PackageIdentity } from "./package"
import { observeUpdatedInstallation } from "./update-installation"
import { PackagePayload, verifyDebPayload } from "./package-payload"
import { verifyRpmPayload } from "./rpm-package-trust"
import { InstalledPayload, installedPayload, verifyInstalledPayload } from "./installed-payload"
import { UpdateBaseline, verifyUpdateBaseline } from "./update"

const fail = (message: string) => new AssertionFailure({ message })
export const UpdateReplacement = Schema.Struct({ before: ApplicationIdentity, after: ApplicationSnapshot, version: Schema.String })
export const UpdateContinuation = Schema.Struct({ version: Schema.String, generation: Generation })
export const UpdateRejection = Schema.Struct({ version: Schema.String, recoveredDownloadVersion: Schema.String,
  interrupted: Schema.optionalWith(InterruptedUpdateTransfer, { as: "Option", exact: true }) })

type Ownership = Effect.Effect.Success<ReturnType<typeof installationSession>>
type RecordEvidence = <A, I>(name: string, schema: Schema.Schema<A, I>, value: A) => Effect.Effect<void, InfrastructureFailure>
/** Owns a single updater journey; the caller closes it before resuming unrelated package operations. */
export const updateJourney = (config: { readonly acceptance: UpdateAcceptance; readonly target: Target; readonly root: string;
  readonly evidence: string; readonly environment: Readonly<Record<string, string>>; readonly port: number; readonly model: string },
  ownership: Ownership, record: RecordEvidence, onCleanupError: (detail: string) => void) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (isWindows(config.target.os) && config.target.arch !== "x64") return yield* new InfrastructureFailure({ operation: "update-journey", message: "Windows update packages require x64" })
  const original = yield* ownership.current
  if (Option.isNone(original)) return yield* fail("Updater journey requires an owned candidate installation")
  const { fixture, pair } = yield* prepareUpdateConsumer(config.acceptance, config.target, join(config.root, "inputs"))
  yield* record("pair", UpdatePair, pair)
  // The Windows owner must create the state directory itself with its protected
  // current-user ACL. A generic temporary directory has inherited ACL entries and
  // is correctly rejected by the native ownership lock.
  const state = isWindows(config.target.os) ? join(config.root, "application-state")
    : yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-up-state-" }).pipe(Effect.flatMap(fs.realPath))
  if (isWindows(config.target.os) && (yield* fs.exists(state))) return yield* new InfrastructureFailure({
    operation: "update-journey", message: "Update application state must begin absent so the native owner can create its private directory",
  })
  const profile = join(config.root, "profile")
  // Load a lease-owned copy: Windows locks loaded DLLs until worker exit.
  // Keep it outside scoped state cleanup; allocation cleanup runs after the worker exits.
  const native = isWindows(config.target.os) ? yield* Effect.gen(function* () {
    const addon = join(config.root, "update-observer.node")
    yield* fs.copyFile(join(original.value.root, "resources", "desktop-host.node"), addon)
    return yield* NativeHost.pipe(Effect.provide(nativeHostLayer(addon)))
  }) : undefined
  const endpoint = updateControlEndpoint(state, isWindows(config.target.os))
  const control = (intent: "Observe" | "Quit") => (native ? endpoint.pipe(Effect.provideService(NativeHost, native))
    : Effect.succeed(join(state, "application.sock"))).pipe(Effect.flatMap(path => requestApplication(path, intent)))
  const environment = yield* runtimeEnvironment(pair.previousRelease, config.target.artifactHost, { ...config.environment,
    MAGNITUDE_DEV_DATA_DIR: profile, MAGNITUDE_DEV_PORT: String(config.port), MAGNITUDE_DESKTOP_STATE_DIR: state,
    NODE_EXTRA_CA_CERTS: fixture.caPath }, [config.acceptance.candidate])
  let handoffAttempted = false, handoffSettled = false
  // Registered before child sessions: stop all app owners before restoring the primary package.
  yield* Effect.addFinalizer(() => Effect.gen(function* () {
    if (handoffAttempted && !handoffSettled) return yield* fail("Update handoff remained unresolved; refusing package mutation")
    const observed = yield* observeUpdatedInstallation(original.value, pair.candidate, environment).pipe(Effect.either)
    if (observed._tag === "Right") yield* ownership.adoptReplacement(observed.right)
    yield* ownership.replace(original.value.candidate)
  }).pipe(Effect.catchAllCause(() => Effect.sync(() => { onCleanupError("Could not restore the primary installation after updater testing") }))))
  const expectedPayload = config.target.packageFormat === "exe" || config.target.packageFormat === "dmg"
    ? Option.some(yield* installedPayload(yield* ownership.replace(pair.candidate))) : Option.none()
  if (Option.isSome(expectedPayload)) yield* record("clean-candidate-payload", InstalledPayload, expectedPayload.value)
  let app = yield* ownership.replace(pair.previous)
  yield* Effect.addFinalizer(() => control("Quit").pipe(Effect.flatMap(owner => owner.service._tag === "Ready"
    ? assertServiceExited(LabProcessId.make(owner.service.health.pid)).pipe(Effect.retry(Schedule.spaced("200 millis").pipe(Schedule.intersect(Schedule.recurs(50))))) : Effect.void),
    Effect.catchTag("ApplicationControlUnavailable", () => Effect.void), Effect.catchAllCause(() => Effect.sync(() => { onCleanupError("Could not stop the automatic update owner") }))))
  const session = yield* desktopSession({ mode: "isolated", executable: app.executable, profile, port: config.port, environment,
    evidence: config.evidence }, onCleanupError)
  const artifact: typeof UpdateFixtureArtifact.Type = { path: pair.update.path, version: pair.candidate.version, bytes: pair.update.artifact.bytes,
    sha256: Digest.make(pair.update.artifact.sha256), target: config.target.os === "macos" ? { os: "darwin", arch: config.target.arch, package: "mac-zip" }
      : isWindows(config.target.os) ? { os: "windows", arch: "x64", package: "windows-exe" }
      : { os: "linux", arch: config.target.arch, package: config.target.packageFormat === "rpm" ? "rpm" : "deb" } }
  const scope = yield* Scope.Scope
  const endpointContext = yield* Layer.buildWithScope(endpointTests(`http://127.0.0.1:${config.port}`, config.model).pipe(Layer.provide(FetchHttpClient.layer)), scope)
  const endpoints = Context.get(endpointContext, EndpointTests)
  const reset = Effect.gen(function* () {
    if (handoffAttempted && !handoffSettled) return yield* new InfrastructureFailure({ operation: "update-restoration", message: "Previous update handoff did not settle; refusing to replace an uncertain installation" })
    yield* session.stop
    const observed = yield* observeUpdatedInstallation(app, pair.candidate, environment).pipe(Effect.either)
    if (observed._tag === "Right") yield* ownership.adoptReplacement(observed.right)
    app = yield* ownership.replace(pair.previous)
    yield* fixture.withdraw
    const driver = yield* session.driver
    yield* driver.ready()
    yield* driver.updates.automatic(false)
    if ((yield* driver.host()) !== pair.previous.version) return yield* fail("Update fault fixture did not restore its baseline")
    return driver
  })
  const download = Effect.gen(function* () {
    const driver = yield* session.driver
    yield* driver.updates.action("check")
    if (!Option.contains((yield* driver.updates.wait("Available")).version, pair.candidate.version)) return yield* fail("App offered a different update version")
    yield* driver.updates.action("download")
    if (!Option.contains((yield* driver.updates.wait("Ready")).version, pair.candidate.version)) return yield* fail("App prepared a different update version")
  })
  const modelFiles = yield* Effect.cached(admittedModelFiles(config.acceptance.candidate, config.target.artifactHost, config.model).pipe(Effect.provide(NodeArchiveExtractor)))
  const baseline = Effect.gen(function* () {
    yield* record("baseline", UpdateBaseline, yield* verifyUpdateBaseline(session, pair.previous.version))
    const driver = yield* session.driver
    yield* driver.search(config.model)
    yield* driver.download(config.model)
    yield* driver.load(config.model)
    yield* record("baseline-generation", Generation, yield* endpoints.generate)
    yield* record("baseline-model-files", ModelFileReceipt, yield* verifyModelFiles(profile, yield* modelFiles))
    yield* record("baseline-package", PackageIdentity, yield* inspectPackageIdentity(app, yield* driver.host(), environment))
  })
  const replacement = Effect.gen(function* () {
    yield* fixture.publish(artifact)
    yield* download
    const driver = yield* session.driver, before = yield* driver.identity()
    handoffAttempted = true
    yield* driver.restartForUpdate()
    yield* session.stop
    // This only observes. Launching the replacement manually cannot make the case pass.
    const after = yield* control("Observe").pipe(Effect.flatMap(owner => Effect.gen(function* () {
      if (owner.pid === before.applicationPid || owner.service._tag !== "Ready") return yield* new InfrastructureFailure({ operation: "update-handoff", message: "Waiting for the automatic replacement owner" })
      handoffSettled = true
      if (owner.service.health.version !== pair.candidate.version) return yield* fail("Updater relaunched the old version; inspect installation failure evidence")
      return owner
    })), Effect.retry({ schedule: Schedule.spaced("500 millis").pipe(Schedule.intersect(Schedule.recurs(120))), while: error => error._tag !== "AssertionFailure" }))
    app = yield* observeUpdatedInstallation(app, pair.candidate, environment)
    yield* ownership.adoptReplacement(app)
    yield* record("replacement", UpdateReplacement, { before, after, version: pair.candidate.version })
    yield* control("Quit")
    yield* Effect.all([assertServiceExited(LabProcessId.make(after.pid)),
      ...(after.service._tag === "Ready" ? [assertServiceExited(LabProcessId.make(after.service.health.pid))] : [])]).pipe(
      Effect.retry(Schedule.spaced("200 millis").pipe(Schedule.intersect(Schedule.recurs(50)))))
  })
  const payload = Effect.gen(function* () {
    const driver = yield* session.driver
    yield* record("updated-package", PackageIdentity, yield* inspectPackageIdentity(app, yield* driver.host(), environment))
    yield* record("updated-payload", PackagePayload, yield* (Option.isSome(expectedPayload) ? verifyInstalledPayload(app, expectedPayload.value)
      : config.target.packageFormat === "rpm" ? verifyRpmPayload(app) : verifyDebPayload(app)))
  })
  const continuation = Effect.gen(function* () {
    const driver = yield* session.driver
    yield* driver.ready()
    yield* driver.verifyTheme("dark")
    yield* record("retained-model-files", ModelFileReceipt, yield* verifyModelFiles(profile, yield* modelFiles))
    yield* driver.search(config.model)
    yield* driver.load(config.model)
    yield* record("continuation", UpdateContinuation, { version: yield* driver.host(), generation: yield* endpoints.generate })
  })
  const rejection = (interrupted: boolean) => Effect.gen(function* () {
    const driver = yield* reset
    yield* fixture.publish(artifact, interrupted ? "Exact" : "Corrupt")
    let transfer: Option.Option<typeof InterruptedUpdateTransfer.Type> = Option.none()
    yield* Effect.scoped(Effect.gen(function* () {
      const fault = interrupted ? Option.some(yield* fixture.interruptDownload) : Option.none()
      yield* driver.updates.action("check")
      yield* driver.updates.wait("Available")
      yield* driver.updates.action("download")
      if (Option.isSome(fault)) {
        yield* driver.updates.wait("Downloading")
        transfer = Option.some(yield* fault.value.started)
        yield* fault.value.cut
      }
      yield* driver.updates.wait("Failed")
    }))
    if ((yield* driver.host()) !== pair.previous.version) return yield* fail("Failed update replaced the running baseline")
    yield* driver.verifyTheme("dark")
    yield* fixture.publish(artifact)
    yield* download
    yield* record(interrupted ? "interruption" : "rejection", UpdateRejection, { version: pair.previous.version, recoveredDownloadVersion: pair.candidate.version, interrupted: transfer })
    yield* driver.updates.action("discard")
    yield* driver.updates.wait("Idle")
  })
  return { baseline, replacement, payload, continuation, corrupt: rejection(false), interrupted: rejection(true) }
})
