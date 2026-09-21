import { isWindows } from "./domain"
import { DebianPackageTrust, inspectDebianPackageTrust } from "./suites/debian-package-trust"
import { FetchHttpClient, FileSystem } from "@effect/platform"
import { Cause, Context, DateTime, Effect, Exit, Layer, Option, Schema, Scope, Stream } from "effect"
import { dirname, isAbsolute, join } from "node:path"
import { ArtifactStore } from "./artifact-store"
import { ArtifactInput, InputManifest } from "./inputs"
import { runtimeEnvironment } from "./runtime-release"
import { prepareApplicationContext } from "./application-context"
import { CaseExecutor, CaseObservation, runCases } from "./case-runner"
import { prepareCandidate, selectInstaller } from "./candidate"
import { captureRemovalProcesses, RemovalProcesses, verifyRemovalProcesses } from "./suites/uninstall-processes"
import { captureRemovalLogin, RemovalLoginEntry, RemovedLoginEntry, verifyRemovedLogin } from "./suites/uninstall-login"
import { RemovalReceipt, verifyNativeRemoval } from "./suites/uninstall"
import { installationSession } from "./installation-session"
import { selectedHarnesses } from "./catalog"
import { captureRetainedProfile, RetainedProfile, verifyRetainedProfile } from "./retained-profile"
import { ApplicationIdentity, assertServiceExited } from "./application-identity"
import { verifyServiceOwnership } from "./suites/service"
import { verifyWorkerRecovery, WorkerRecoveryEvidence } from "./suites/worker-recovery"
import { admittedModelFiles, verifyModelFiles } from "./model-files"
import { verifyDownloadRecovery, DownloadRecoveryEvidence } from "./suites/download-recovery"
import { nativeNetworkFault } from "./network-fault"
import { verifyOfflineRecovery, OfflineRecoveryEvidence } from "./suites/offline-recovery"
import { nativeWorkerFault } from "./worker-fault"
import { InstallationOwnership, verifyInstallationOwnership } from "./suites/installation-ownership"
import { occupyServicePort } from "./port-fault"
import { exerciseConnectionError } from "./harnesses/connection-error"
import { desktopSession } from "./desktop-session"
import { DesktopDriver } from "./desktop-driver"
import { AssertionFailure, Evidence, InfrastructureFailure } from "./domain"
import { HostInspector } from "./host-inspector"
import { attestGeneration, HostObservation } from "./hardware"
import { executionTelemetry, LoadedBackendModule } from "./execution-telemetry"
import { GenerationExecution, observeGeneration } from "./generation-evidence"
import { admittedRuntimeModules, attestRuntimeModules } from "./runtime-modules"
import { NodeArchiveExtractor } from "../../release/src/archive"
import { Installer } from "./installer"
import { command, CommandOutput, ProcessExecutor } from "./process"
import { sha256 } from "./snapshot"
import { publishEvidenceFile } from "./evidence"
import { inspectPackageIdentity, PackageIdentity } from "./suites/package"
import { inspectMacPackageDependencies, MacPackageDependencies } from "./suites/package-dependencies"
import { appleTrustPolicy, ApplePackageTrust, inspectApplePackageTrust } from "./suites/apple-package-trust"
import { windowsTrustPolicy, WindowsPackageTrust, inspectWindowsPackageTrust } from "./suites/windows-package-trust"
import { inspectLinuxPackageDependencies, LinuxPackageDependencies } from "./suites/linux-package-dependencies"
import { inspectWindowsPackageDependencies, WindowsPackageDependencies } from "./suites/windows-package-dependencies"
import { rejectCorruptInstaller } from "./suites/install"
import { connectionFixture, ConnectionReceipt } from "./harnesses/connection-fixture"
import { Harness } from "./domain"
import { harnessSuite, HarnessTools, HarnessTurn } from "./harnesses/suite"
import { hermesTerminal } from "./harnesses/hermes-terminal"
import { piTerminal } from "./harnesses/pi-terminal"
import { openCodeModelName, openCodeTerminal } from "./harnesses/opencode-terminal"
import { HarnessTerminalReceipt } from "./harnesses/terminal"
import { TerminalDriver } from "./terminal"
import { EndpointTests, endpointTests, Generation } from "./suites/endpoint"
import { bundledCliTests, CliTests } from "./suites/cli"
import { CliInterruption, verifyCliInterruption } from "./suites/cli-interruption"
import { WorkAssignment, TargetResult } from "./work-store"
import { prepareUpdateConsumer } from "./update-consumer"
import { prepareUpdatePair, UpdatePair } from "./update-pair"
import { updateJourney } from "./suites/update-journey"
import { inspectRpmPackageTrust, RpmPackageTrust } from "./suites/rpm-package-trust"
import { UpdateBaseline, verifyUpdateBaseline } from "./suites/update"

export const CandidateWorkerConfig = Schema.Struct({ root: Schema.NonEmptyString, port: Schema.Int.pipe(Schema.between(1024, 65535)),
  model: Schema.NonEmptyString, environment: Schema.Record({ key: Schema.String, value: Schema.String }) })
export class CandidateWorkerFailure extends Schema.TaggedError<CandidateWorkerFailure>()("CandidateWorkerFailure", { message: Schema.String, cleanupErrors: Schema.Array(Schema.String) }) {}
const unavailable = (message: string) => new InfrastructureFailure({ operation: "candidate-worker", message })

/** Guest-side execution. The caller supplies scoped objects and a native installer, never cloud credentials. */
export const runCandidateWorker = (assignment: WorkAssignment, config: typeof CandidateWorkerConfig.Type) => Effect.gen(function* () {
  if (DateTime.toEpochMillis(assignment.deadline) <= Date.now()) return yield* unavailable("Worker assignment has expired")
  const fs = yield* FileSystem.FileSystem
  const objects = yield* ArtifactStore
  const inspector = yield* HostInspector
  const installer = yield* Installer
  const processes = yield* ProcessExecutor
  const target = assignment.target.target
  const cleanupErrors: string[] = []
  const diagnostics = new Map<string, typeof Evidence.Type>()
  const harnessSetupEvidence: (typeof Evidence.Type)[] = []
  const backendEvidence: (typeof Evidence.Type)[] = []
  const recoveryEvidence: (typeof Evidence.Type)[] = []
  const offlineEvidence: (typeof Evidence.Type)[] = []
  const downloadEvidence: (typeof Evidence.Type)[] = []
  const evidenceDirectory = join(config.root, "evidence")
  if (yield* fs.exists(config.root)) return yield* unavailable("Candidate worker requires a fresh owned workspace")
  yield* fs.makeDirectory(evidenceDirectory, { recursive: true, mode: 0o700 })
  const scope = yield* Scope.make()
  const evidence = <A, I>(name: string, schema: Schema.Schema<A, I>, value: A) => Effect.gen(function* () {
    const wire = yield* Schema.encode(Schema.parseJson(schema))(value)
    const bytes = new TextEncoder().encode(wire)
    const digest = sha256(bytes)
    yield* objects.put(digest, Stream.make(bytes))
    yield* fs.writeFile(join(evidenceDirectory, name), bytes, { mode: 0o600 })
    return Evidence.make({ path: `evidence/${name}`, sha256: digest, bytes: bytes.byteLength })
  }).pipe(Effect.mapError(error => unavailable(error.message)))
  const program = Effect.gen(function* () {
    const host = yield* inspector.inspect(target)
    const hostEvidence = yield* evidence("host.json", HostObservation, host)
    const input = assignment.input
    let size = 0
    const chunks = yield* objects.get(input.digest).pipe(Stream.tap(chunk => Effect.gen(function* () {
      size += chunk.byteLength
      if (size > 16 * 1024 * 1024) return yield* unavailable("Artifact manifest exceeds 16 MiB")
    })), Stream.runCollect)
    const wire = Buffer.concat(Array.from(chunks))
    if (sha256(wire) !== input.digest) return yield* unavailable("Worker input manifest does not match admitted digest")
    const admitted = yield* Schema.decodeUnknown(Schema.parseJson(InputManifest))(wire.toString("utf8")).pipe(Effect.mapError(() => unavailable("Invalid admitted input manifest")))
    if (admitted.kind !== input.kind) return yield* unavailable("Input kind does not match assignment")
    if (assignment.work.kind !== "test" || admitted.kind !== "artifacts") return yield* unavailable("A clean consumer requires admitted packages; source compilation belongs to a separate producer")
    const manifest = Effect.succeed(admitted)
    const sourceEvidence = yield* evidence("admitted-input.json", InputManifest, admitted)
    const inputEvidence = yield* Effect.cached(Effect.gen(function* () {
      const value = yield* manifest
      yield* selectInstaller(value.release, target)
      return yield* evidence("artifact-input.json", ArtifactInput, value)
    }))
    const application = yield* prepareApplicationContext(config.root, config.port, config.environment).pipe(Effect.provideService(Scope.Scope, scope))
    const applicationEvidence = yield* evidence("application-context.json", Schema.Struct({ mode: Schema.String, profile: Schema.String, port: Schema.Int, harnessHome: Schema.String }),
      { mode: application.mode, profile: application.profile, port: application.port, harnessHome: application.harnessHome })
    const environment = application.environment
    const collector = assignment.target.cases.some(test => test.id === "E6" || test.id === "R4" || test.id === "R3" || test.id === "R2")
      ? Option.some(yield* executionTelemetry().pipe(Effect.provideService(Scope.Scope, scope))) : Option.none()
    const candidateEnvironment = yield* Effect.cached(manifest.pipe(Effect.flatMap(value => runtimeEnvironment(value.release, target.artifactHost,
      Option.isSome(collector) ? { ...environment, MAGNITUDE_OTEL_ENDPOINT: collector.value.endpoint } : environment)),
      Effect.provideService(ArtifactStore, objects), Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(Scope.Scope, scope)))
    const selection = assignment.plan.request.selection
    const harnesses = selectedHarnesses(selection)
    const connections = assignment.target.cases.some(test => test.id === "A5" || test.id === "C4" || test.id === "A7")
      ? yield* Effect.forEach(harnesses, harness => connectionFixture(application.harnessHome, harness, `http://127.0.0.1:${application.port}/inference/v1`)) : []
    const candidate = yield* Effect.cached(manifest.pipe(Effect.flatMap(value => prepareCandidate(value.release, target, join(config.root, "candidate"))),
      Effect.provideService(ArtifactStore, objects), Effect.provideService(FileSystem.FileSystem, fs)))
    const installation = yield* Effect.cached(candidate.pipe(Effect.flatMap(value => installationSession(value,
      detail => { cleanupErrors.push(`Uninstall: ${detail}`) }).pipe(Effect.provideService(Installer, installer), Effect.provideService(Scope.Scope, scope)))))
    const installed = installation.pipe(Effect.flatMap(value => value.get))
    let activeSession: Option.Option<Effect.Effect.Success<ReturnType<typeof desktopSession>>> = Option.none()
    let fixtureCleanupFailed = false
    const session = yield* Effect.cached(Effect.gen(function* () {
      const app = yield* installed
      const value = yield* desktopSession({ mode: application.mode, executable: app.executable, profile: application.profile,
        evidence: join(evidenceDirectory, "desktop"), port: application.port, environment: yield* candidateEnvironment }, detail => { cleanupErrors.push(detail) }).pipe(
        Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(Scope.Scope, scope))
      activeSession = Option.some(value)
      return value
    }))
    const desktop = session.pipe(Effect.flatMap(value => value.driver))
    let removalProcesses: Option.Option<typeof RemovalProcesses.Type> = Option.none()
    let removalLogin: Option.Option<typeof RemovalLoginEntry.Type> = Option.none()
    const loginRemovalSelected = assignment.target.cases.some(test => test.id === "X2")
    const retentionSelected = assignment.target.cases.some(test => test.id === "X3" || test.id === "X4")
    const retainedProfile = yield* Effect.cached(Effect.gen(function* () {
      const driver = yield* desktop
      yield* driver.theme("dark")
      yield* driver.quit()
      yield* (yield* session).stop
      return yield* captureRetainedProfile(application.profile).pipe(Effect.provideService(FileSystem.FileSystem, fs))
    }))
    const cli = yield* Effect.cached(Effect.gen(function* () {
      const app = yield* installed
      const context = yield* Layer.buildWithScope(bundledCliTests({ executable: app.cli, version: (yield* manifest).release.version, model: config.model,
        evidence: join(evidenceDirectory, "cli"), environment: yield* candidateEnvironment }).pipe(Layer.provide(Layer.merge(Layer.succeed(ProcessExecutor, processes), Layer.succeed(FileSystem.FileSystem, fs)))), scope)
      return Context.get(context, CliTests)
    }))
    const endpoint = yield* Effect.cached(Effect.gen(function* () {
      const context = yield* Layer.buildWithScope(endpointTests(`http://127.0.0.1:${application.port}`, config.model).pipe(Layer.provide(FetchHttpClient.layer)), scope)
      return Context.get(context, EndpointTests)
    }))
    const tools = yield* Effect.serviceOption(HarnessTools)
    const harnessSuites = new Map<Harness, ReturnType<typeof harnessSuite>>()
    for (const harness of harnesses) harnessSuites.set(harness, yield* Effect.cached(Effect.gen(function* () {
      const prepared = yield* candidateEnvironment
      if (harness === "hermes") {
        // Hermes's first-run guard requires an explicit default selection even when the
        // provider was connected through the UI. Exercise the product's existing command.
        const selected = yield* command((yield* installed).cli, ["connections", "add", "hermes", "--set-model", config.model], {
          env: prepared, inheritEnv: false, timeoutMs: 60_000,
        })
        harnessSetupEvidence.push(yield* evidence("H1-hermes-model-selection.json", CommandOutput, selected))
        if (selected.exitCode !== 0) return yield* new AssertionFailure({ message: "Bundled CLI could not select the Hermes model; inspect retained setup output" })
      }
      return yield* harnessSuite(harness, config.model, join(evidenceDirectory, "harness", harness), application.harnessHome, prepared)
    })))
    let updaterScope: Option.Option<Scope.CloseableScope> = Option.none()
    let activeUpdateCase = "U1"
    const updaterEvidence = new Map<string, (typeof Evidence.Type)[]>()
    const closeUpdater = Effect.gen(function* () {
      if (Option.isNone(updaterScope)) return
      const owned = updaterScope.value
      updaterScope = Option.none()
      yield* Scope.close(owned, Exit.void)
    })
    const updater = yield* Effect.cached(Effect.gen(function* () {
      if (Option.isNone(admitted.updateAcceptance)) return yield* unavailable("Scheduled updater execution requires an admitted source-built acceptance pair")
      const cleanupStart = cleanupErrors.length
      if (Option.isSome(activeSession)) yield* activeSession.value.stop
      if (cleanupErrors.length !== cleanupStart) {
        fixtureCleanupFailed = true
        return yield* unavailable("Primary application cleanup failed before update journey installation")
      }
      yield* installed
      const owned = yield* Scope.make()
      updaterScope = Option.some(owned)
      // Register after installation ownership: LIFO cleanup must stop the update app
      // and restore the primary package before the outer installer removes that package.
      yield* Scope.addFinalizer(scope, closeUpdater.pipe(Effect.catchAllCause(() => Effect.sync(() => {
        fixtureCleanupFailed = true; cleanupErrors.push("Updater fixture scope did not close cleanly")
      }))))
      return yield* updateJourney({ acceptance: admitted.updateAcceptance.value, target, root: join(config.root, "update-journey"),
        evidence: join(evidenceDirectory, "update-journey"), environment, port: application.port, model: config.model }, yield* installation,
        (name, schema, value) => evidence(`${activeUpdateCase}-${name}.json`, schema, value).pipe(Effect.tap(item => Effect.sync(() => {
          updaterEvidence.set(activeUpdateCase, [...(updaterEvidence.get(activeUpdateCase) ?? []), item])
        })), Effect.asVoid), detail => { fixtureCleanupFailed = true; cleanupErrors.push(`Updater: ${detail}`) }).pipe(Effect.provideService(Scope.Scope, owned))
    }))
    const execute: CaseExecutor["execute"] = test => Effect.gen(function* () {
      if (test.suite !== "update") yield* closeUpdater
      if (fixtureCleanupFailed) return yield* unavailable("Native fixture restoration failed; refusing further operations on an uncertain installation")
      if (test.suite === "update" && Option.isSome(admitted.updateAcceptance) && Option.isNone(assignment.plan.request.updateFrom)) {
        activeUpdateCase = test.id
        const journey = yield* updater
        const operation = { U1: journey.baseline, U2: journey.replacement, U3: journey.payload,
          U4: journey.continuation, U5: journey.corrupt, U6: journey.interrupted }[test.id as "U1" | "U2" | "U3" | "U4" | "U5" | "U6"]
        if (!operation) return yield* unavailable("Unknown updater case")
        yield* operation
        return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence, hostEvidence] })
      }
      switch (test.id as string) {
        case "U1": {
          const baseline = assignment.plan.request.updateFrom
          const acceptance = admitted.updateAcceptance
          if (Option.isNone(baseline) && Option.isNone(acceptance)) return yield* unavailable("Update tests require a source-built acceptance pair or an admitted previous artifact manifest through --update-from")
          // An explicit historical baseline retains precedence; same-source fixtures do not claim migration compatibility.
          const restored = Option.isNone(baseline) && Option.isSome(acceptance)
            ? Option.some(yield* prepareUpdateConsumer(acceptance.value, target, join(config.root, "update-pair")).pipe(Effect.provideService(Scope.Scope, scope))) : Option.none()
          const pair = Option.isSome(restored) ? restored.value.pair
            : yield* prepareUpdatePair(Option.getOrThrow(baseline).digest, admitted.release, target, join(config.root, "update-pair"))
          const pairEvidence = yield* evidence("update-pair.json", UpdatePair, pair)
          // Native package managers have one installation. Suspend the primary journey and restore
          // its exact ownership after this separate baseline/profile fixture has completely closed.
          const cleanupStart = cleanupErrors.length
          if (Option.isSome(activeSession)) yield* activeSession.value.stop
          if (cleanupErrors.length !== cleanupStart) {
            fixtureCleanupFailed = true
            return yield* unavailable("Primary application cleanup failed before update baseline installation")
          }
          const ownership = yield* installation
          const previous = yield* ownership.current
          return yield* Effect.acquireUseRelease(
            Effect.void,
            () => Effect.scoped(Effect.gen(function* () {
              const app = yield* ownership.replace(pair.previous)
              const updateState = isWindows(target.os) ? join(config.root, "update-profile", "state")
                : yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-up-state-" }).pipe(Effect.flatMap(fs.realPath))
              const baselineEnvironment = yield* runtimeEnvironment(pair.previousRelease, target.artifactHost, environment)
              const updateEnvironment = { ...baselineEnvironment, ...(Option.isSome(restored) ? { NODE_EXTRA_CA_CERTS: restored.value.fixture.caPath } : {}), MAGNITUDE_DEV_DATA_DIR: join(config.root, "update-profile"), MAGNITUDE_DEV_PORT: String(application.port), MAGNITUDE_DESKTOP_STATE_DIR: updateState }
              const updateSession = yield* desktopSession({ mode: "isolated", executable: app.executable, profile: updateEnvironment.MAGNITUDE_DEV_DATA_DIR,
                evidence: join(evidenceDirectory, "update-baseline"), port: application.port, environment: updateEnvironment }, detail => { cleanupErrors.push(detail) })
              const observation = yield* verifyUpdateBaseline(updateSession, pair.previous.version)
              const driver = yield* updateSession.driver
              const identity = yield* inspectPackageIdentity(app, yield* driver.host(), updateEnvironment)
              yield* driver.quit()
              yield* updateSession.stop
              yield* assertServiceExited(observation.reopenedOwner.servicePid)
              const profile = yield* captureRetainedProfile(updateEnvironment.MAGNITUDE_DEV_DATA_DIR)
              return CaseObservation.make({ detail: `${Option.isSome(restored) ? "Installed the same-source updater acceptance baseline" : "Installed the admitted previous version"}, verified desktop/service/CLI package identity and persisted theme across an application/service restart`,
                evidence: [pairEvidence, hostEvidence, yield* evidence("update-baseline.json", UpdateBaseline, observation),
                  yield* evidence("update-baseline-package.json", PackageIdentity, identity), yield* evidence("update-baseline-profile.json", RetainedProfile, profile)] })
            })),
            () => Effect.gen(function* () {
              if (cleanupErrors.length !== cleanupStart) {
                fixtureCleanupFailed = true
                return
              }
              yield* (Option.isSome(previous) ? ownership.replace(previous.value.candidate).pipe(Effect.asVoid) : ownership.reset(yield* candidate))
            }).pipe(Effect.catchAllCause(cause => Effect.sync(() => {
              fixtureCleanupFailed = true
              cleanupErrors.push(`Restore candidate after update baseline: ${Cause.pretty(cause)}`)
            }))),
          )
        }
        case "P1": {
          return CaseObservation.make({ detail: `Recorded supplied artifact provenance ${(yield* manifest).release.sourceCommit}; compilation was not executed`, evidence: [yield* inputEvidence, hostEvidence] })
        }
        case "P2":
        case "I1": {
          yield* candidate
          return CaseObservation.make({ detail: "Downloaded the selected native installer and verified its admitted hash and length; no package build was executed",
            evidence: [yield* inputEvidence, sourceEvidence] })
        }
        case "P3": {
          const identity = yield* inspectPackageIdentity(yield* installed, yield* (yield* desktop).host(), environment).pipe(
            Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(ProcessExecutor, processes))
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence, yield* evidence("package-identity.json", PackageIdentity, identity)] })
        }
        case "P4": {
          const release = (yield* manifest).release
          if (!release.artifacts.some(artifact => artifact.kind === "icn-base" && Option.contains(artifact.host, target.artifactHost))) {
            return yield* unavailable("Dependency closure requires the admitted native runtime archives")
          }
          if (isWindows(target.os)) {
            const report = yield* inspectWindowsPackageDependencies(yield* installed, release, environment).pipe(Effect.provide(NodeArchiveExtractor))
            return CaseObservation.make({ detail: "Verified packaged PE import closure in native loader contexts, including delay imports, API-set mappings and signed OS boundaries",
              evidence: [yield* inputEvidence, yield* evidence("package-dependencies.json", WindowsPackageDependencies, report)] })
          }
          if (target.os !== "macos") {
            const report = yield* inspectLinuxPackageDependencies(yield* installed, release).pipe(Effect.provide(NodeArchiveExtractor))
            return CaseObservation.make({ detail: "Verified installed ELF and admitted runtime dependency graphs, symbol versions, OS package owners and native loader paths",
              evidence: [yield* inputEvidence, yield* evidence("package-dependencies.json", LinuxPackageDependencies, report)] })
          }
          const report = yield* inspectMacPackageDependencies(yield* installed, release).pipe(Effect.provide(NodeArchiveExtractor))
          return CaseObservation.make({ detail: "Verified every packaged Mach-O executable/library and selected native runtime dependency graph; OS shared-cache boundaries recorded separately",
            evidence: [yield* inputEvidence, yield* evidence("package-dependencies.json", MacPackageDependencies, report)] })
        }
        case "P5": {
          const production = assignment.plan.request.selection.kind === "profile" && assignment.plan.request.selection.profile === "release"
          const release = (yield* manifest).release
          if (!release.artifacts.some(artifact => artifact.kind === "icn-base" && Option.contains(artifact.host, target.artifactHost))) {
            return yield* unavailable("Complete signature verification requires the admitted native runtime archives")
          }
          if (target.packageFormat === "deb") {
            const receipt = yield* inspectDebianPackageTrust(yield* installed, release, production).pipe(Effect.provide(NodeArchiveExtractor))
            return CaseObservation.make({ detail: "Verified exact installed DEB payload and admitted runtime archives; package is unsigned development output, no publisher trust claimed",
              evidence: [yield* inputEvidence, yield* evidence("P5-debian-package-trust.json", DebianPackageTrust, receipt)] })
          }
          if (target.packageFormat === "rpm") {
            const receipt = yield* inspectRpmPackageTrust(yield* installed, release, production).pipe(Effect.provide(NodeArchiveExtractor))
            return CaseObservation.make({ detail: "Verified installed RPM payload, native package integrity and admitted runtime archives; no production publisher trust claimed",
              evidence: [yield* inputEvidence, yield* evidence("P5-rpm-package-trust.json", RpmPackageTrust, receipt)] })
          }
          if (target.os !== "macos" && !isWindows(target.os)) return yield* unavailable("Native package trust verification is not yet qualified for this platform")
          if (isWindows(target.os)) {
            const policy = yield* windowsTrustPolicy(production, config.environment)
            const receipt = yield* inspectWindowsPackageTrust(yield* installed, release, policy, config.environment).pipe(Effect.provide(NodeArchiveExtractor))
            return CaseObservation.make({ detail: receipt.productionTrusted ? "Verified timestamped Authenticode signatures and expected publishers on installer, application and admitted runtime"
              : "Inspected development Authenticode status; unsigned files recorded explicitly, no production trust claimed",
              evidence: [yield* inputEvidence, yield* evidence("P5-windows-package-trust.json", WindowsPackageTrust, receipt)] })
          }
          const policy = yield* appleTrustPolicy(production, config.environment.LAB_EXPECTED_APPLE_TEAM_ID)
          const receipt = yield* inspectApplePackageTrust(yield* installed, release, policy).pipe(Effect.provide(NodeArchiveExtractor))
          return CaseObservation.make({ detail: receipt.productionTrusted ? "Verified expected publisher, timestamps, notarization and Gatekeeper acceptance"
            : "Verified application and admitted runtime code signatures; development integrity only, no production trust claimed",
            evidence: [yield* inputEvidence, yield* evidence("P5-apple-package-trust.json", ApplePackageTrust, receipt)] })
        }
        case "I2": yield* installed; break
        case "I3": {
          const version = yield* (yield* desktop).host()
          if (version !== (yield* manifest).release.version) return yield* new AssertionFailure({ message: "Installed desktop reports a different version from the admitted candidate" })
          break
        }
        case "I5": yield* rejectCorruptInstaller(yield* candidate).pipe(Effect.provideService(Installer, installer), Effect.provideService(FileSystem.FileSystem, fs)); break
        case "I4": {
          const receipt = yield* verifyInstallationOwnership(yield* installed, yield* desktop).pipe(Effect.provideService(CliTests, yield* cli))
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("I4-installation-ownership.json", InstallationOwnership, receipt)] })
        }
        case "A1": yield* (yield* desktop).ready(); break
        case "A2": yield* (yield* desktop).search(config.model); yield* (yield* desktop).details(config.model); break
        case "A3": yield* (yield* desktop).search(config.model); yield* (yield* desktop).download(config.model); break
        case "A4": {
          yield* (yield* desktop).theme("dark")
          const restarted = yield* (yield* session).restart
          yield* restarted.host()
          yield* restarted.verifyTheme("dark")
          yield* restarted.theme("light")
          const second = yield* (yield* session).restart
          yield* second.host()
          yield* second.verifyTheme("light")
          yield* second.ready()
          break
        }
        case "A5": {
          const driver = yield* desktop
          const receipts = yield* Effect.forEach(connections, connection => connection.exercise.pipe(Effect.provideService(DesktopDriver, driver)))
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence, yield* evidence("connections.json", Schema.Array(ConnectionReceipt), receipts)] })
        }
        case "A6": {
          const driver = yield* desktop
          yield* driver.chrome()
          yield* driver.quit()
          yield* (yield* (yield* session).restart).ready()
          break
        }
        case "A7": {
          const running = yield* session
          yield* running.stop
          const serviceError = yield* Effect.scoped(Effect.gen(function* () {
            yield* occupyServicePort(application.port)
            const failed = yield* desktop
            const message = yield* failed.serviceFailure()
            yield* failed.screenshot("service-failure")
            yield* running.stop
            return message
          }))
          yield* (yield* desktop).ready()
          const driver = yield* desktop
          for (const fixture of connections) {
            yield* fixture.exercise.pipe(Effect.provideService(DesktopDriver, driver))
            yield* exerciseConnectionError(application.harnessHome, fixture.harness).pipe(
              Effect.provideService(DesktopDriver, driver), Effect.provideService(FileSystem.FileSystem, fs))
            yield* fixture.inspect(true)
          }
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("service-error.json", Schema.Struct({ message: Schema.String }), { message: serviceError })] })
        }
        case "E1": yield* (yield* desktop).search(config.model); yield* (yield* desktop).load(config.model); yield* (yield* endpoint).discover; break
        case "E6": {
          if (Option.isNone(collector)) return yield* unavailable("Backend verification requires the scoped native execution collector")
          const expected = yield* admittedRuntimeModules((yield* manifest).release, target.artifactHost).pipe(Effect.provide(NodeArchiveExtractor))
          const observed = yield* observeGeneration(`http://127.0.0.1:${application.port}`, config.model, collector.value).pipe(Effect.provide(FetchHttpClient.layer))
          const receipt = yield* evidence("E6-native-generation.json", GenerationExecution, observed)
          const modules = yield* evidence("E6-admitted-modules.json", Schema.Array(LoadedBackendModule), expected)
          backendEvidence.push(receipt, modules)
          yield* attestRuntimeModules(observed.native, expected)
          yield* attestGeneration(target, host, config.model, observed.native)
          return CaseObservation.make({ detail: "Public generation matched admitted runtime modules and requested target-model allocation", evidence: [yield* inputEvidence, hostEvidence, receipt, modules] })
        }
        case "E2":
        case "E3":
        case "E4":
        case "R1": {
          const tests = yield* endpoint
          const generation = yield* test.id === "E2" ? tests.generate : test.id === "E3" ? tests.stream : test.id === "E4" ? tests.tools : tests.cancelAndRetry
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence(`${test.id}-generation.json`, Generation, generation)] })
        }
        case "R2": {
          if (Option.isNone(collector)) return yield* unavailable("Download recovery requires native execution evidence")
          const release = (yield* manifest).release
          const expected = yield* admittedModelFiles(release, target.artifactHost, config.model).pipe(Effect.provide(NodeArchiveExtractor))
          const modules = yield* admittedRuntimeModules(release, target.artifactHost).pipe(Effect.provide(NodeArchiveExtractor))
          downloadEvidence.push(yield* evidence("R2-admitted-modules.json", Schema.Array(LoadedBackendModule), modules))
          const observe = observeGeneration(`http://127.0.0.1:${application.port}`, config.model, collector.value).pipe(Effect.provide(FetchHttpClient.layer))
          yield* verifyDownloadRecovery(config.model, verifyModelFiles(application.profile, expected).pipe(Effect.provideService(FileSystem.FileSystem, fs)), observe,
            generation => attestRuntimeModules(generation.native, modules).pipe(Effect.zipRight(attestGeneration(target, host, config.model, generation.native))),
            value => evidence(`R2-${value._tag}.json`, DownloadRecoveryEvidence, value).pipe(Effect.tap(item => Effect.sync(() => { downloadEvidence.push(item) })), Effect.asVoid),
            detail => { cleanupErrors.push(`Download interruption restoration: ${detail}`) }).pipe(
              Effect.provideService(DesktopDriver, yield* desktop), Effect.provideService(CliTests, yield* cli), Effect.provide(nativeNetworkFault))
          return CaseObservation.make({ detail: "Interrupted an active UI model download, retried through the UI, verified all admitted file digests and attested generation", evidence: [yield* inputEvidence, hostEvidence] })
        }
        case "R3": {
          if (Option.isNone(collector)) return yield* unavailable("Offline recovery requires native execution evidence")
          const expected = yield* admittedRuntimeModules((yield* manifest).release, target.artifactHost).pipe(Effect.provide(NodeArchiveExtractor))
          offlineEvidence.push(yield* evidence("R3-admitted-modules.json", Schema.Array(LoadedBackendModule), expected))
          const observe = observeGeneration(`http://127.0.0.1:${application.port}`, config.model, collector.value).pipe(Effect.provide(FetchHttpClient.layer))
          yield* verifyOfflineRecovery(observe, generation => attestRuntimeModules(generation.native, expected).pipe(
            Effect.zipRight(attestGeneration(target, host, config.model, generation.native))), value => evidence(`R3-${value._tag}.json`, OfflineRecoveryEvidence, value).pipe(
              Effect.tap(item => Effect.sync(() => { offlineEvidence.push(item) })), Effect.asVoid), detail => { cleanupErrors.push(`Offline network restoration: ${detail}`) }).pipe(
            Effect.provideService(DesktopDriver, yield* desktop), Effect.provideService(CliTests, yield* cli), Effect.provide(nativeNetworkFault))
          return CaseObservation.make({ detail: "Reloaded the cached model and attested generation with external traffic blocked, then restored connectivity", evidence: [yield* inputEvidence, hostEvidence] })
        }
        case "R4": {
          if (Option.isNone(collector)) return yield* unavailable("Worker recovery requires native execution evidence")
          const expected = yield* admittedRuntimeModules((yield* manifest).release, target.artifactHost).pipe(Effect.provide(NodeArchiveExtractor))
          recoveryEvidence.push(yield* evidence("R4-admitted-modules.json", Schema.Array(LoadedBackendModule), expected))
          const observe = observeGeneration(`http://127.0.0.1:${application.port}`, config.model, collector.value).pipe(Effect.provide(FetchHttpClient.layer))
          yield* verifyWorkerRecovery(application.profile, observe, generation => attestRuntimeModules(generation.native, expected).pipe(
            Effect.zipRight(attestGeneration(target, host, config.model, generation.native))), value => evidence(`R4-${value._tag}.json`, WorkerRecoveryEvidence, value).pipe(
              Effect.tap(item => Effect.sync(() => { recoveryEvidence.push(item) })), Effect.asVoid)).pipe(
            Effect.provideService(DesktopDriver, yield* desktop), Effect.provideService(CliTests, yield* cli), Effect.provide(nativeWorkerFault))
          return CaseObservation.make({ detail: "Reloaded after an owned worker crash and verified a new native generation without replacing application owners", evidence: [yield* inputEvidence, hostEvidence] })
        }
        case "R5": {
          yield* (yield* cli).reloadModel
          const generation = yield* (yield* endpoint).generate
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("R5-generation.json", Generation, generation)] })
        }
        case "R6": {
          const observations = yield* verifyServiceOwnership(yield* session).pipe(Effect.provideService(CliTests, yield* cli))
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("R6-service-ownership.json", Schema.Array(ApplicationIdentity), observations)] })
        }
        case "E5": yield* (yield* endpoint).invalid; break
        case "H1": case "H2": case "H3": case "H4": case "H5": case "H6": {
          if (Option.isNone(tools) || Option.isNone(test.harness)) return yield* unavailable("Worker has no qualified harness tool configuration")
          const suite = yield* harnessSuites.get(test.harness.value)!.pipe(Effect.provideService(HarnessTools, tools.value),
            Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(ProcessExecutor, processes))
          const turn = yield* (test.id === "H5" ? suite.tools : test.id === "H4" || test.id === "H6" ? suite.recall : suite.initial).pipe(
            Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(ProcessExecutor, processes))
          if (test.id === "H3" && !turn.streamed) return yield* unavailable("This harness reports completed text parts; token streaming qualification requires additional evidence")
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence(`${test.id}-${test.harness.value}-turn.json`, HarnessTurn, turn)] })
        }
        case "H7": {
          if (Option.isNone(test.harness)) return yield* unavailable("Interactive terminal qualification requires a harness selection")
          const harness = test.harness.value
          const terminal = yield* Effect.serviceOption(TerminalDriver)
          const runtime = config.environment.LAB_TERMINAL_NODE_EXECUTABLE
          if (Option.isNone(tools) || Option.isNone(terminal) || !runtime || !isAbsolute(runtime)) {
            return yield* unavailable("Terminal tests require qualified harness tools, a TerminalDriver and an absolute LAB_TERMINAL_NODE_EXECUTABLE")
          }
          const home = application.harnessHome, prepared = yield* candidateEnvironment
          const terminalConfig = { executable: yield* tools.value.executable(harness), runtime,
            cwd: config.root, evidence: join(evidenceDirectory, "harness", harness, "terminal"), model: config.model, initialModel: config.model,
            environment: { ...prepared, HOME: home, USERPROFILE: home, PI_CODING_AGENT_DIR: join(home, ".pi", "agent"), HERMES_HOME: join(home, ".hermes"),
              XDG_CONFIG_HOME: join(home, ".config"), XDG_DATA_HOME: join(home, ".local", "share"),
              XDG_CACHE_HOME: join(home, ".cache"), XDG_STATE_HOME: join(home, ".local", "state"),
              PATH: `${dirname(runtime)}${isWindows(target.os) ? ";" : ":"}${prepared.PATH ?? ""}` },
            // The completed marker must be absent from echoed input. Familiar words avoid
            // turning terminal qualification into an arbitrary numeric-copying test.
            interrupt: { prompt: "First concatenate SUN and FLOWER without a space and print that uppercase word. Then count from 1 to 10000, one number per line. Do not use tools.", expected: "SUNFLOWER" },
            recovery: { prompt: "Concatenate RAIN and BOW without a space. Reply only with the resulting uppercase word. Do not use tools.", expected: "RAINBOW" },
          }
          const cleanup = (message: string) => { cleanupErrors.push(`${harness} terminal: ${message}`) }
          const journey = harness === "hermes" ? hermesTerminal({ ...terminalConfig, endpoint: `http://127.0.0.1:${application.port}/inference/v1` }, cleanup)
            : harness === "pi" ? piTerminal(terminalConfig, cleanup) : openCodeModelName(home, config.model).pipe(
            Effect.flatMap(name => openCodeTerminal({ ...terminalConfig, modelName: name, initialModelName: name }, cleanup)))
          const receipt = yield* journey.pipe(Effect.provideService(TerminalDriver, terminal.value),
            Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(ProcessExecutor, processes))
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence(`H7-${harness}-terminal.json`, HarnessTerminalReceipt, receipt)] })
        }
        case "C1": yield* (yield* cli).version; break
        case "C2": yield* (yield* desktop).ready(); yield* (yield* cli).inspect; break
        case "C3": yield* (yield* cli).modelLifecycle; break
        case "C4": {
          const tests = yield* cli
          for (const fixture of connections) yield* tests.connections(fixture.harness, fixture.inspect)
          break
        }
        case "C5": {
          const driver = yield* desktop
          yield* driver.ready()
          const before = yield* driver.identity()
          const tests = yield* cli
          yield* tests.invalid
          const interruption = yield* verifyCliInterruption({ executable: (yield* installed).cli, port: application.port, environment: yield* candidateEnvironment }, before).pipe(
            Effect.provideService(ProcessExecutor, processes))
          yield* tests.inspect
          if (!Schema.equivalence(ApplicationIdentity)(before, yield* driver.identity())) return yield* new AssertionFailure({ message: "CLI interruption changed the owning application or service" })
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("C5-interruption.json", CliInterruption, interruption)] })
        }
        case "C6": yield* (yield* cli).nativeRuntime; break
        case "X1": {
          const app = yield* installed
          if (loginRemovalSelected) {
            if (application.mode !== "installed-user") return yield* unavailable("Login/process removal requires a qualified installed desktop user")
            const driver = yield* desktop
            yield* driver.ready()
            yield* driver.loginStartup(true)
            removalLogin = Option.some(yield* captureRemovalLogin(environment, target.os === "macos" ? Option.some(yield* driver.macLoginRegistration()) : Option.none()))
            removalProcesses = Option.some(yield* captureRemovalProcesses(yield* driver.identity()))
          }
          if (retentionSelected) yield* retainedProfile
          yield* (yield* session).stop
          yield* (yield* installation).remove
          const receipt = yield* verifyNativeRemoval(app, config.environment).pipe(
            Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(ProcessExecutor, processes))
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("X1-native-removal.json", RemovalReceipt, receipt)] })
        }
        case "X2": {
          if (Option.isNone(removalProcesses) || Option.isNone(removalLogin)) return yield* unavailable("Removal has no captured live process tree and enabled login entry")
          return CaseObservation.make({ detail: "Verified every captured application/service descendant exited and the enabled login entry cannot run after uninstall",
            evidence: [yield* inputEvidence, yield* evidence("X2-processes.json", RemovalProcesses, yield* verifyRemovalProcesses(removalProcesses.value)),
              yield* evidence("X2-enabled-login.json", RemovalLoginEntry, removalLogin.value),
              yield* evidence("X2-removed-login.json", RemovedLoginEntry, yield* verifyRemovedLogin(removalLogin.value))] })
        }
        case "X3": {
          const receipt = yield* verifyRetainedProfile(application.profile, yield* retainedProfile).pipe(Effect.provideService(FileSystem.FileSystem, fs))
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("X3-retained-profile.json", RetainedProfile, receipt)] })
        }
        case "X4": {
          yield* installed
          const driver = yield* desktop
          if ((yield* driver.host()) !== (yield* manifest).release.version) return yield* new AssertionFailure({ message: "Reinstalled application version differs from admitted candidate" })
          yield* driver.verifyTheme("dark")
          yield* driver.ready()
          if (Option.isSome(removalLogin)) yield* driver.verifyLoginStartup(true)
          break
        }
        default: return yield* unavailable(`Case ${test.id} is not yet connected to the candidate worker; no acceptance claimed`)
      }
      return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence, hostEvidence] })
    }).pipe(Effect.provide(Layer.mergeAll(Layer.succeed(FileSystem.FileSystem, fs), Layer.succeed(ArtifactStore, objects), Layer.succeed(ProcessExecutor, processes))), Effect.onExit(exit => Exit.isFailure(exit) ? Effect.gen(function* () {
      if (test.suite === "update") {
        const preparedPath = join(config.root, "update-journey", "profile", "updates", "update.json")
        if (yield* fs.exists(preparedPath)) {
          if (Number((yield* fs.stat(preparedPath)).size) < 1024 * 1024) {
            const prepared = yield* fs.readFileString(preparedPath).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Unknown))))
            updaterEvidence.set(test.id, [...(updaterEvidence.get(test.id) ?? []), yield* evidence(`${test.id}-prepared-update.json`, Schema.Unknown, prepared)])
          }
        }
      }
      const identity = `${test.id}-${Option.getOrElse(test.harness, () => "shared")}`
      const detail = Cause.pretty(exit.cause).replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]").slice(0, 32 * 1024)
      diagnostics.set(identity, yield* evidence(`${identity}-failure.json`, Schema.Struct({ detail: Schema.String }), { detail }))
    }).pipe(Effect.orDie) : Effect.void), Effect.mapError(error => error._tag === "AssertionFailure" || error._tag === "InfrastructureFailure" ? error : unavailable(error.message)))
    const results = yield* runCases(target, assignment.target.cases).pipe(Effect.provideService(CaseExecutor, { execute }))
    return results.map(result => {
      const diagnostic = diagnostics.get(`${result.caseId}-${Option.getOrElse(result.harness, () => "shared")}`)

      const refs = [...result.evidence, applicationEvidence, ...(updaterEvidence.get(result.caseId) ?? []), ...(result.harness._tag === "Some" && result.harness.value === "hermes" ? harnessSetupEvidence : []), ...(result.caseId === "E6" ? backendEvidence : []), ...(result.caseId === "R4" ? recoveryEvidence : []), ...(result.caseId === "R3" ? offlineEvidence : []), ...(result.caseId === "R2" ? downloadEvidence : []), ...(diagnostic ? [diagnostic] : [])]
      return { ...result, evidence: [...new Map(refs.map(item => [item.sha256, item])).values()] }
    })
  })
  // Closing the shared scope closes the UI before uninstalling its package. Cleanup cannot mask test results.
  const cases = yield* program.pipe(Effect.timeoutFail({ duration: Math.max(1, DateTime.toEpochMillis(assignment.deadline) - Date.now()), onTimeout: () => unavailable("Worker assignment deadline expired") }), Effect.ensuring(Scope.close(scope, Exit.void).pipe(
    Effect.catchAllCause(() => Effect.sync(() => { cleanupErrors.push("Application cleanup failed; inspect local worker diagnostics") })),
  )), Effect.mapError(error => new CandidateWorkerFailure({ message: error.message, cleanupErrors: [...cleanupErrors] })))
  // Export bounded, redacted process output after the desktop scope has flushed it.
  const desktopLog = join(evidenceDirectory, "desktop", "desktop.log")
  const finalCases = [...cases]
  // Custom selections need not contain I3. Preserve shared diagnostics on a selected
  // result even when the desktop was acquired by a different scenario.
  const desktopCase = finalCases.find(result => result.caseId === "I3") ?? finalCases[0]
  yield* Effect.gen(function* () {
    if (!desktopCase || !(yield* fs.exists(desktopLog))) return
    const detail = (yield* fs.readFileString(desktopLog)).replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]").slice(-2 * 1024 * 1024)
    const log = yield* evidence("desktop-process.json", Schema.Struct({ detail: Schema.String }), { detail })
    const index = finalCases.indexOf(desktopCase)
    finalCases[index] = { ...finalCases[index]!, evidence: [...finalCases[index]!.evidence, log] }
  }).pipe(Effect.catchAll(error => Effect.sync(() => { cleanupErrors.push(`Desktop process evidence: ${error.message}`) })))
  // Traces are finalized when the desktop scope closes. Publish before the allocator removes
  // the worker; a local path alone is not durable evidence. Export errors preserve case results.
  const exportFile = (relative: string, caseId: string, maxBytes: number, harness?: Harness) => Effect.gen(function* () {
    const index = finalCases.findIndex(result => result.caseId === caseId && (harness === undefined || Option.contains(result.harness, harness)))
    if (index < 0) return
    const item = yield* publishEvidenceFile(evidenceDirectory, relative, maxBytes)
    finalCases[index] = { ...finalCases[index]!, evidence: [...finalCases[index]!.evidence, item] }
  }).pipe(Effect.catchAll(error => Effect.sync(() => { cleanupErrors.push(`Evidence ${relative}: ${error.message}`) })))
  const updateEvidenceDirectory = join(evidenceDirectory, "update-journey")
  if (yield* fs.exists(updateEvidenceDirectory)) {
    const launches = ["", ...(yield* fs.readDirectory(updateEvidenceDirectory)).filter(name => /^relaunch-\d+$/.test(name))]
    const selected = finalCases.filter(result => result.caseId.startsWith("U"))
    for (const launch of launches) for (const file of ["ui-trace.zip", "desktop.log"]) {
      const relative = join("update-journey", launch, file)
      if (yield* fs.exists(join(evidenceDirectory, relative))) for (const result of selected) yield* exportFile(relative, result.caseId, file.endsWith("zip") ? 128 * 1024 * 1024 : 2 * 1024 * 1024)
    }
  }
  const baselineEvidence = join(evidenceDirectory, "update-baseline")
  if (yield* fs.exists(baselineEvidence)) {
    const launches = ["", ...(yield* fs.readDirectory(baselineEvidence)).filter(name => /^relaunch-\d+$/.test(name))]
    for (const launch of launches) for (const file of ["ui-trace.zip", "desktop.log"]) {
      const relative = join("update-baseline", launch, file)
      if (yield* fs.exists(join(evidenceDirectory, relative))) yield* exportFile(relative, "U1", file.endsWith("zip") ? 128 * 1024 * 1024 : 2 * 1024 * 1024)
    }
  }
  if (desktopCase && (yield* fs.exists(join(evidenceDirectory, "desktop", "ui-trace.zip")))) yield* exportFile("desktop/ui-trace.zip", desktopCase.caseId, 128 * 1024 * 1024, Option.getOrUndefined(desktopCase.harness))
  const desktopEvidence = join(evidenceDirectory, "desktop")
  if (yield* fs.exists(desktopEvidence)) {
    const relaunches = (yield* fs.readDirectory(desktopEvidence)).filter(name => /^relaunch-\d+$/.test(name))
    for (const name of relaunches) {
      for (const file of ["ui-trace.zip", "desktop.log"]) {
        if (yield* fs.exists(join(desktopEvidence, name, file))) {
          for (const caseId of ["A4", "A6", "A7", "R6", "X1", "X4"]) yield* exportFile(`desktop/${name}/${file}`, caseId, file.endsWith("zip") ? 128 * 1024 * 1024 : 2 * 1024 * 1024)
        }
      }
    }
  }
  const cliEvidence = join(evidenceDirectory, "cli")
  if (yield* fs.exists(cliEvidence)) {
    const cliCase = finalCases.find(result => result.caseId === "C1")
      ?? finalCases.find(result => ["I4", "R2", "R3", "R4", "R5", "R6", "C2", "C3", "C4", "C5", "C6"].includes(result.caseId))
    const names = (yield* fs.readDirectory(cliEvidence)).filter(name => /^\d+-[a-z0-9-]+\.json$/.test(name)).sort()
    if (names.length > 100) cleanupErrors.push("CLI evidence exceeded its file-count limit")
    else if (cliCase) for (const name of names) yield* exportFile(`cli/${name}`, cliCase.caseId, 32 * 1024 * 1024, Option.getOrUndefined(cliCase.harness))
  }
  for (const harness of Harness.literals) {
    for (const file of ["session.jsonl", "lifecycle.jsonl", "selected-model.json", "aborted.session.json", "recovered.session.json", "streaming.json", "recovered.json", "terminal-output.txt", "terminal-screen.json", "failure-screen.json", "terminal-events.jsonl", "terminal-bridge.stderr.log"]) {
      const relative = `harness/${harness}/terminal/${file}`
      if (yield* fs.exists(join(evidenceDirectory, relative))) yield* exportFile(relative, "H7", 32 * 1024 * 1024, harness)
    }
    const directory = join(evidenceDirectory, "harness", harness, "events")
    if (!(yield* fs.exists(directory))) continue
    const names = (yield* fs.readDirectory(directory)).filter(name => /^(version\.txt|models\.txt|turn-\d+\.(jsonl|native-events\.jsonl|stream\.json|stderr\.log|prompt\.txt|session\.json))$/.test(name)).sort()
    if (names.length > 100) cleanupErrors.push(`${harness} evidence exceeded its file-count limit`)
    else for (const name of names) yield* exportFile(`harness/${harness}/events/${name}`, "H1", 32 * 1024 * 1024, harness)
  }
  return TargetResult.make({ cases: finalCases, cleanupErrors })
}).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" || error._tag === "CandidateWorkerFailure" ? error : unavailable(error.message)))
