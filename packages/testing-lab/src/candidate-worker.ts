import { FetchHttpClient, FileSystem } from "@effect/platform"
import { Cause, Context, DateTime, Effect, Exit, Layer, Option, Schema, Scope, Stream } from "effect"
import { join } from "node:path"
import { ArtifactStore } from "./artifact-store"
import { ArtifactInput, InputManifest } from "./inputs"
import { SourceBuilder } from "./source-builder"
import { runtimeEnvironment } from "./runtime-release"
import { CaseExecutor, CaseObservation, runCases } from "./case-runner"
import { prepareCandidate, selectInstaller } from "./candidate"
import { RemovalReceipt, verifyNativeRemoval } from "./suites/uninstall"
import { installationSession } from "./installation-session"
import { selectedHarnesses } from "./catalog"
import { captureRetainedProfile, RetainedProfile, verifyRetainedProfile } from "./retained-profile"
import { ApplicationIdentity, assertServiceExited } from "./application-identity"
import { verifyServiceOwnership } from "./suites/service"
import { occupyServicePort } from "./port-fault"
import { exerciseConnectionError } from "./harnesses/connection-error"
import { desktopSession } from "./desktop-session"
import { DesktopDriver } from "./desktop-driver"
import { AssertionFailure, Evidence, InfrastructureFailure } from "./domain"
import { HostInspector } from "./host-inspector"
import { HostObservation } from "./hardware"
import { Installer } from "./installer"
import { ProcessExecutor } from "./process"
import { sha256 } from "./snapshot"
import { publishEvidenceFile } from "./evidence"
import { inspectPackageIdentity, PackageIdentity } from "./suites/package"
import { rejectCorruptInstaller } from "./suites/install"
import { connectionFixture, ConnectionReceipt } from "./harnesses/connection-fixture"
import { Harness } from "./domain"
import { harnessSuite, HarnessTools, HarnessTurn } from "./harnesses/suite"
import { EndpointTests, endpointTests, Generation } from "./suites/endpoint"
import { bundledCliTests, CliTests } from "./suites/cli"
import { CliInterruption, verifyCliInterruption } from "./suites/cli-interruption"
import { WorkAssignment, TargetResult } from "./work-store"
import { prepareUpdatePair, UpdatePair } from "./update-pair"
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
    const input = assignment.plan.request.input
    let size = 0
    const chunks = yield* objects.get(input.digest).pipe(Stream.tap(chunk => Effect.gen(function* () {
      size += chunk.byteLength
      if (size > 16 * 1024 * 1024) return yield* unavailable("Artifact manifest exceeds 16 MiB")
    })), Stream.runCollect)
    const wire = Buffer.concat(Array.from(chunks))
    if (sha256(wire) !== input.digest) return yield* unavailable("Worker input manifest does not match admitted digest")
    const admitted = yield* Schema.decodeUnknown(Schema.parseJson(InputManifest))(wire.toString("utf8")).pipe(Effect.mapError(() => unavailable("Invalid admitted input manifest")))
    if (admitted.kind !== input.kind) return yield* unavailable("Input kind does not match assignment")
    const builder = yield* Effect.serviceOption(SourceBuilder)
    if (admitted.kind === "source" && Option.isNone(builder)) return yield* unavailable("Worker has no source builder")
    const source = admitted.kind === "source" ? yield* Option.getOrThrow(builder).prepare(admitted, input.digest, target) : undefined
    const manifest = yield* Effect.cached(admitted.kind === "artifacts" ? Effect.succeed(admitted)
      : source!.package.pipe(Effect.map(result => result.input)))
    const sourceEvidence = yield* evidence("admitted-input.json", InputManifest, admitted)
    const inputEvidence = yield* Effect.cached(Effect.gen(function* () {
      const value = yield* manifest
      yield* selectInstaller(value.release, target)
      return yield* evidence("artifact-input.json", ArtifactInput, value)
    }))
    // Native Unix control sockets have a small byte limit independent of the artifact path.
    // Keep one private, scope-owned state directory shared by the app and bundled CLI.
    const stateDirectory = process.platform === "win32" ? join(config.root, "profile", "state")
      : yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-state-" }).pipe(Effect.provideService(Scope.Scope, scope))
    const environment = { ...config.environment, MAGNITUDE_DESKTOP_STATE_DIR: stateDirectory, HOME: join(config.root, "home"), USERPROFILE: join(config.root, "home"),
      APPDATA: join(config.root, "home", "AppData", "Roaming"), LOCALAPPDATA: join(config.root, "home", "AppData", "Local"),
      XDG_CONFIG_HOME: join(config.root, "home", ".config"), XDG_DATA_HOME: join(config.root, "home", ".local", "share"),
      MAGNITUDE_DEV_DATA_DIR: join(config.root, "profile"), MAGNITUDE_DEV_PORT: String(config.port), MAGNITUDE_SHELL_ENV_INHERITED: "1" }
    yield* fs.makeDirectory(environment.HOME, { recursive: true, mode: 0o700 })
    const candidateEnvironment = yield* Effect.cached(manifest.pipe(Effect.flatMap(value => runtimeEnvironment(value.release, target.artifactHost, environment)),
      Effect.provideService(ArtifactStore, objects), Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(Scope.Scope, scope)))
    const selection = assignment.plan.request.selection
    const harnesses = selectedHarnesses(selection)
    const connections = assignment.target.cases.some(test => test.id === "A5" || test.id === "C4" || test.id === "A7")
      ? yield* Effect.forEach(harnesses, harness => connectionFixture(join(environment.MAGNITUDE_DEV_DATA_DIR, "harness-home"), harness, `http://127.0.0.1:${config.port}/inference/v1`)) : []
    const candidate = yield* Effect.cached(manifest.pipe(Effect.flatMap(value => prepareCandidate(value.release, target, join(config.root, "candidate"))),
      Effect.provideService(ArtifactStore, objects), Effect.provideService(FileSystem.FileSystem, fs)))
    const installation = yield* Effect.cached(candidate.pipe(Effect.flatMap(value => installationSession(value,
      detail => { cleanupErrors.push(`Uninstall: ${detail}`) }).pipe(Effect.provideService(Installer, installer), Effect.provideService(Scope.Scope, scope)))))
    const installed = installation.pipe(Effect.flatMap(value => value.get))
    let activeSession: Option.Option<Effect.Effect.Success<ReturnType<typeof desktopSession>>> = Option.none()
    let fixtureCleanupFailed = false
    const session = yield* Effect.cached(Effect.gen(function* () {
      const app = yield* installed
      const value = yield* desktopSession({ executable: app.executable, profile: environment.MAGNITUDE_DEV_DATA_DIR,
        evidence: join(evidenceDirectory, "desktop"), port: config.port, environment: yield* candidateEnvironment }, detail => { cleanupErrors.push(detail) }).pipe(
        Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(Scope.Scope, scope))
      activeSession = Option.some(value)
      return value
    }))
    const desktop = session.pipe(Effect.flatMap(value => value.driver))
    const retentionSelected = assignment.target.cases.some(test => test.id === "X3" || test.id === "X4")
    const retainedProfile = yield* Effect.cached(Effect.gen(function* () {
      const driver = yield* desktop
      yield* driver.theme("dark")
      yield* driver.quit()
      yield* (yield* session).stop
      return yield* captureRetainedProfile(environment.MAGNITUDE_DEV_DATA_DIR).pipe(Effect.provideService(FileSystem.FileSystem, fs))
    }))
    const cli = yield* Effect.cached(Effect.gen(function* () {
      const app = yield* installed
      const context = yield* Layer.buildWithScope(bundledCliTests({ executable: app.cli, version: (yield* manifest).release.version, model: config.model,
        evidence: join(evidenceDirectory, "cli"), environment: yield* candidateEnvironment }).pipe(Layer.provide(Layer.merge(Layer.succeed(ProcessExecutor, processes), Layer.succeed(FileSystem.FileSystem, fs)))), scope)
      return Context.get(context, CliTests)
    }))
    const endpoint = yield* Effect.cached(Effect.gen(function* () {
      const context = yield* Layer.buildWithScope(endpointTests(`http://127.0.0.1:${config.port}`, config.model).pipe(Layer.provide(FetchHttpClient.layer)), scope)
      return Context.get(context, EndpointTests)
    }))
    const tools = yield* Effect.serviceOption(HarnessTools)
    const harnessSuites = new Map<Harness, ReturnType<typeof harnessSuite>>()
    for (const harness of harnesses) harnessSuites.set(harness, yield* Effect.cached(candidateEnvironment.pipe(Effect.flatMap(prepared => harnessSuite(harness, config.model,
      join(evidenceDirectory, "harness", harness), join(environment.MAGNITUDE_DEV_DATA_DIR, "harness-home"), prepared)))))
    const execute: CaseExecutor["execute"] = test => Effect.gen(function* () {
      if (fixtureCleanupFailed) return yield* unavailable("Native fixture restoration failed; refusing further operations on an uncertain installation")
      switch (test.id as string) {
        case "U1": {
          const baseline = assignment.plan.request.updateFrom
          if (Option.isNone(baseline)) return yield* unavailable("Update tests require an admitted previous artifact manifest through --update-from")
          const pair = yield* prepareUpdatePair(baseline.value.digest, (yield* manifest).release, target, join(config.root, "update-pair"))
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
              const updateState = target.os === "windows" ? join(config.root, "update-profile", "state")
                : yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-up-state-" })
              const baselineEnvironment = yield* runtimeEnvironment(pair.previousRelease, target.artifactHost, environment)
              const updateEnvironment = { ...baselineEnvironment, MAGNITUDE_DEV_DATA_DIR: join(config.root, "update-profile"), MAGNITUDE_DESKTOP_STATE_DIR: updateState }
              const updateSession = yield* desktopSession({ executable: app.executable, profile: updateEnvironment.MAGNITUDE_DEV_DATA_DIR,
                evidence: join(evidenceDirectory, "update-baseline"), port: config.port, environment: updateEnvironment }, detail => { cleanupErrors.push(detail) })
              const observation = yield* verifyUpdateBaseline(updateSession, pair.previous.version)
              const driver = yield* updateSession.driver
              const identity = yield* inspectPackageIdentity(app, yield* driver.host(), updateEnvironment)
              yield* driver.quit()
              yield* updateSession.stop
              yield* assertServiceExited(observation.reopenedOwner.servicePid)
              const profile = yield* captureRetainedProfile(updateEnvironment.MAGNITUDE_DEV_DATA_DIR)
              return CaseObservation.make({ detail: "Installed the admitted previous version, verified desktop/service/CLI package identity and persisted theme across an application/service restart",
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
          if (source) {
            const result = yield* source.compile
            return CaseObservation.make({ ...result, evidence: [...result.evidence, sourceEvidence, hostEvidence] })
          }
          return CaseObservation.make({ detail: `Recorded supplied artifact provenance ${(yield* manifest).release.sourceCommit}; compilation was not executed`, evidence: [yield* inputEvidence, hostEvidence] })
        }
        case "P2":
        case "I1": {
          yield* candidate
          const build = source ? yield* source.package : undefined
          return CaseObservation.make({ detail: source ? "Built final native packages from the admitted source and verified exact installer bytes"
            : "Downloaded the selected native installer and verified its admitted hash and length; no package build was executed",
            evidence: [yield* inputEvidence, sourceEvidence, ...(build?.evidence ?? [])] })
        }
        case "P3": {
          const identity = yield* inspectPackageIdentity(yield* installed, yield* (yield* desktop).host(), environment).pipe(
            Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(ProcessExecutor, processes))
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence, yield* evidence("package-identity.json", PackageIdentity, identity)] })
        }
        case "I2": yield* installed; break
        case "I3": {
          const version = yield* (yield* desktop).host()
          if (version !== (yield* manifest).release.version) return yield* new AssertionFailure({ message: "Installed desktop reports a different version from the admitted candidate" })
          break
        }
        case "I5": yield* rejectCorruptInstaller(yield* candidate).pipe(Effect.provideService(Installer, installer), Effect.provideService(FileSystem.FileSystem, fs)); break
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
            yield* occupyServicePort(config.port)
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
            yield* exerciseConnectionError(join(environment.MAGNITUDE_DEV_DATA_DIR, "harness-home"), fixture.harness).pipe(
              Effect.provideService(DesktopDriver, driver), Effect.provideService(FileSystem.FileSystem, fs))
            yield* fixture.inspect(true)
          }
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("service-error.json", Schema.Struct({ message: Schema.String }), { message: serviceError })] })
        }
        case "E1": yield* (yield* desktop).search(config.model); yield* (yield* desktop).load(config.model); yield* (yield* endpoint).discover; break
        case "E2":
        case "E3":
        case "E4":
        case "R1": {
          const tests = yield* endpoint
          const generation = yield* test.id === "E2" ? tests.generate : test.id === "E3" ? tests.stream : test.id === "E4" ? tests.tools : tests.cancelAndRetry
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence(`${test.id}-generation.json`, Generation, generation)] })
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
          const interruption = yield* verifyCliInterruption({ executable: (yield* installed).cli, port: config.port, environment: yield* candidateEnvironment }, before).pipe(
            Effect.provideService(ProcessExecutor, processes))
          yield* tests.inspect
          if (!Schema.equivalence(ApplicationIdentity)(before, yield* driver.identity())) return yield* new AssertionFailure({ message: "CLI interruption changed the owning application or service" })
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("C5-interruption.json", CliInterruption, interruption)] })
        }
        case "C6": yield* (yield* cli).nativeRuntime; break
        case "X1": {
          const app = yield* installed
          if (retentionSelected) yield* retainedProfile
          yield* (yield* session).stop
          yield* (yield* installation).remove
          const receipt = yield* verifyNativeRemoval(app, config.environment).pipe(
            Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(ProcessExecutor, processes))
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("X1-native-removal.json", RemovalReceipt, receipt)] })
        }
        case "X3": {
          const receipt = yield* verifyRetainedProfile(environment.MAGNITUDE_DEV_DATA_DIR, yield* retainedProfile).pipe(Effect.provideService(FileSystem.FileSystem, fs))
          return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence,
            yield* evidence("X3-retained-profile.json", RetainedProfile, receipt)] })
        }
        case "X4": {
          yield* installed
          const driver = yield* desktop
          if ((yield* driver.host()) !== (yield* manifest).release.version) return yield* new AssertionFailure({ message: "Reinstalled application version differs from admitted candidate" })
          yield* driver.verifyTheme("dark")
          yield* driver.ready()
          break
        }
        default: return yield* unavailable(`Case ${test.id} is not yet connected to the candidate worker; no acceptance claimed`)
      }
      return CaseObservation.make({ detail: test.title, evidence: [yield* inputEvidence, hostEvidence] })
    }).pipe(Effect.provide(Layer.mergeAll(Layer.succeed(FileSystem.FileSystem, fs), Layer.succeed(ArtifactStore, objects), Layer.succeed(ProcessExecutor, processes))), Effect.onExit(exit => Exit.isFailure(exit) ? Effect.gen(function* () {
      const identity = `${test.id}-${Option.getOrElse(test.harness, () => "shared")}`
      const detail = Cause.pretty(exit.cause).replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]").slice(0, 32 * 1024)
      diagnostics.set(identity, yield* evidence(`${identity}-failure.json`, Schema.Struct({ detail: Schema.String }), { detail }))
    }).pipe(Effect.orDie) : Effect.void), Effect.mapError(error => error._tag === "AssertionFailure" || error._tag === "InfrastructureFailure" ? error : unavailable(error.message)))
    const results = yield* runCases(target, assignment.target.cases).pipe(Effect.provideService(CaseExecutor, { execute }))
    return results.map(result => {
      const diagnostic = diagnostics.get(`${result.caseId}-${Option.getOrElse(result.harness, () => "shared")}`)
      const buildEvidence = source && (result.caseId === "P1" || result.caseId === "P2") ? source.evidence().filter(item => result.caseId !== "P1" || item.path !== "evidence/build-package.json") : []
      const refs = [...result.evidence, ...buildEvidence, ...(diagnostic ? [diagnostic] : [])]
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
  if (yield* fs.exists(desktopLog)) {
    const detail = (yield* fs.readFileString(desktopLog)).replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]").slice(-2 * 1024 * 1024)
    const log = yield* evidence("desktop-process.json", Schema.Struct({ detail: Schema.String }), { detail })
    const index = finalCases.findIndex(result => result.caseId === "I3")
    if (index >= 0) finalCases[index] = { ...finalCases[index]!, evidence: [...finalCases[index]!.evidence, log] }
  }
  // Traces are finalized when the desktop scope closes. Publish before the allocator removes
  // the worker; a local path alone is not durable evidence. Export errors preserve case results.
  const exportFile = (relative: string, caseId: string, maxBytes: number, harness?: Harness) => Effect.gen(function* () {
    const index = finalCases.findIndex(result => result.caseId === caseId && (harness === undefined || Option.contains(result.harness, harness)))
    if (index < 0) return
    const item = yield* publishEvidenceFile(evidenceDirectory, relative, maxBytes)
    finalCases[index] = { ...finalCases[index]!, evidence: [...finalCases[index]!.evidence, item] }
  }).pipe(Effect.catchAll(error => Effect.sync(() => { cleanupErrors.push(`Evidence ${relative}: ${error.message}`) })))
  const baselineEvidence = join(evidenceDirectory, "update-baseline")
  if (yield* fs.exists(baselineEvidence)) {
    const launches = ["", ...(yield* fs.readDirectory(baselineEvidence)).filter(name => /^relaunch-\d+$/.test(name))]
    for (const launch of launches) for (const file of ["ui-trace.zip", "desktop.log"]) {
      const relative = join("update-baseline", launch, file)
      if (yield* fs.exists(join(evidenceDirectory, relative))) yield* exportFile(relative, "U1", file.endsWith("zip") ? 128 * 1024 * 1024 : 2 * 1024 * 1024)
    }
  }
  if (yield* fs.exists(join(evidenceDirectory, "desktop", "ui-trace.zip"))) yield* exportFile("desktop/ui-trace.zip", "I3", 128 * 1024 * 1024)
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
    const names = (yield* fs.readDirectory(cliEvidence)).filter(name => /^\d+-[a-z0-9-]+\.json$/.test(name)).sort()
    if (names.length > 100) cleanupErrors.push("CLI evidence exceeded its file-count limit")
    else for (const name of names) yield* exportFile(`cli/${name}`, "C1", 32 * 1024 * 1024)
  }
  for (const harness of Harness.literals) {
    const directory = join(evidenceDirectory, "harness", harness, "events")
    if (!(yield* fs.exists(directory))) continue
    const names = (yield* fs.readDirectory(directory)).filter(name => /^(version\.txt|models\.txt|turn-\d+\.(jsonl|stderr\.log|prompt\.txt|session\.json))$/.test(name)).sort()
    if (names.length > 100) cleanupErrors.push(`${harness} evidence exceeded its file-count limit`)
    else for (const name of names) yield* exportFile(`harness/${harness}/events/${name}`, "H1", 32 * 1024 * 1024, harness)
  }
  return TargetResult.make({ cases: finalCases, cleanupErrors })
}).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" || error._tag === "CandidateWorkerFailure" ? error : unavailable(error.message)))
