import { FileSystem } from "@effect/platform"
import { Context, DateTime, Effect, Exit, Layer, Schema, Scope, Stream } from "effect"
import { join } from "node:path"
import { ArtifactStore } from "./artifact-store"
import { ArtifactInput } from "./inputs"
import { CaseExecutor, CaseObservation, runCases } from "./case-runner"
import { prepareCandidate, selectInstaller } from "./candidate"
import { DesktopDriver, playwrightDesktop } from "./desktop-driver"
import { AssertionFailure, Evidence, InfrastructureFailure } from "./domain"
import { HostInspector } from "./host-inspector"
import { HostObservation } from "./hardware"
import { Installer } from "./installer"
import { ProcessExecutor } from "./process"
import { sha256 } from "./snapshot"
import { bundledCliTests, CliTests } from "./suites/cli"
import { WorkAssignment, TargetResult } from "./work-store"

export const ArtifactWorkerConfig = Schema.Struct({ root: Schema.NonEmptyString, port: Schema.Int.pipe(Schema.between(1024, 65535)),
  model: Schema.NonEmptyString, environment: Schema.Record({ key: Schema.String, value: Schema.String }) })
const unavailable = (message: string) => new InfrastructureFailure({ operation: "artifact-worker", message })

/** Guest-side execution. The caller supplies scoped objects and a native installer, never cloud credentials. */
export const runArtifactWorker = (assignment: WorkAssignment, config: typeof ArtifactWorkerConfig.Type) => Effect.gen(function* () {
  if (assignment.plan.request.input.kind !== "artifacts") return yield* unavailable("Artifact worker requires an existing-artifact input")
  if (DateTime.toEpochMillis(assignment.deadline) <= Date.now()) return yield* unavailable("Worker assignment has expired")
  const fs = yield* FileSystem.FileSystem
  const objects = yield* ArtifactStore
  const inspector = yield* HostInspector
  const installer = yield* Installer
  const processes = yield* ProcessExecutor
  const target = assignment.target.target
  const cleanupErrors: string[] = []
  const evidenceDirectory = join(config.root, "evidence")
  if (yield* fs.exists(config.root)) return yield* unavailable("Artifact worker requires a fresh owned workspace")
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
    const manifest = yield* Schema.decodeUnknown(Schema.parseJson(ArtifactInput))(wire.toString("utf8")).pipe(Effect.mapError(() => unavailable("Invalid admitted artifact manifest")))
    yield* selectInstaller(manifest.release, target)
    const inputEvidence = yield* evidence("artifact-input.json", ArtifactInput, manifest)
    const environment = { ...config.environment, HOME: join(config.root, "home"), USERPROFILE: join(config.root, "home"),
      APPDATA: join(config.root, "home", "AppData", "Roaming"), LOCALAPPDATA: join(config.root, "home", "AppData", "Local"),
      XDG_CONFIG_HOME: join(config.root, "home", ".config"), XDG_DATA_HOME: join(config.root, "home", ".local", "share"),
      MAGNITUDE_DEV_DATA_DIR: join(config.root, "profile"), MAGNITUDE_DEV_PORT: String(config.port), MAGNITUDE_SHELL_ENV_INHERITED: "1" }
    yield* fs.makeDirectory(environment.HOME, { recursive: true, mode: 0o700 })
    const candidate = yield* Effect.cached(prepareCandidate(manifest.release, target, join(config.root, "candidate")).pipe(
      Effect.provideService(ArtifactStore, objects), Effect.provideService(FileSystem.FileSystem, fs)))
    const installed = yield* Effect.cached(Effect.gen(function* () {
      const packageFile = yield* candidate
      return yield* Effect.acquireRelease(installer.install(packageFile), app => installer.uninstall(app).pipe(
        Effect.catchAll(error => Effect.sync(() => { cleanupErrors.push(`Uninstall: ${error.message}`) })),
      )).pipe(Effect.provideService(Scope.Scope, scope))
    }))
    const desktop = yield* Effect.cached(Effect.gen(function* () {
      const app = yield* installed
      const context = yield* Layer.buildWithScope(playwrightDesktop({ executable: app.executable, profile: environment.MAGNITUDE_DEV_DATA_DIR,
        evidence: join(evidenceDirectory, "desktop"), port: config.port, environment }).pipe(Layer.provide(Layer.succeed(FileSystem.FileSystem, fs))), scope)
      return Context.get(context, DesktopDriver)
    }))
    const cli = yield* Effect.cached(Effect.gen(function* () {
      const app = yield* installed
      const context = yield* Layer.buildWithScope(bundledCliTests({ executable: app.cli, version: manifest.release.version, model: config.model,
        evidence: join(evidenceDirectory, "cli"), environment }).pipe(Layer.provide(Layer.merge(Layer.succeed(ProcessExecutor, processes), Layer.succeed(FileSystem.FileSystem, fs)))), scope)
      return Context.get(context, CliTests)
    }))
    const execute: CaseExecutor["execute"] = test => Effect.gen(function* () {
      switch (test.id as string) {
        case "P1": return CaseObservation.make({ detail: `Recorded supplied artifact provenance ${manifest.release.sourceCommit}; compilation was not executed`, evidence: [inputEvidence, hostEvidence] })
        case "P2":
        case "I1": {
          yield* candidate
          return CaseObservation.make({ detail: "Downloaded the selected native installer and verified its admitted hash and length; no package build was executed", evidence: [inputEvidence] })
        }
        case "I2": yield* installed; break
        case "I3": {
          const version = yield* (yield* desktop).host()
          if (version !== manifest.release.version) return yield* new AssertionFailure({ message: "Installed desktop reports a different version from the admitted candidate" })
          break
        }
        case "A1": yield* (yield* desktop).ready(); break
        case "C1": yield* (yield* cli).version; break
        case "C2": yield* (yield* desktop).ready(); yield* (yield* cli).inspect; break
        case "C6": yield* (yield* cli).nativeRuntime; break
        default: return yield* unavailable(`Case ${test.id} is not yet connected to the artifact worker; no acceptance claimed`)
      }
      return CaseObservation.make({ detail: test.title, evidence: [inputEvidence, hostEvidence] })
    }).pipe(Effect.mapError(error => error._tag === "AssertionFailure" || error._tag === "InfrastructureFailure" ? error : unavailable(error.message)))
    return yield* runCases(target, assignment.target.cases).pipe(Effect.provideService(CaseExecutor, { execute }))
  })
  // Closing the shared scope closes the UI before uninstalling its package. Cleanup cannot mask test results.
  const cases = yield* program.pipe(Effect.timeoutFail({ duration: Math.max(1, DateTime.toEpochMillis(assignment.deadline) - Date.now()), onTimeout: () => unavailable("Worker assignment deadline expired") }), Effect.ensuring(Scope.close(scope, Exit.void).pipe(
    Effect.catchAllCause(() => Effect.sync(() => { cleanupErrors.push("Application cleanup failed; inspect local worker diagnostics") })),
  )))
  return TargetResult.make({ cases, cleanupErrors })
}).pipe(Effect.mapError(error => error._tag === "AssertionFailure" || error._tag === "InfrastructureFailure" ? error : unavailable(error.message)))
