import { assertRuntime } from "../src/runtime"
import { ReleaseArtifactSchema, ReleaseManifestSchema, validateReleaseManifest } from "@magnitudedev/release/contracts"
import { Digest } from "../src/domain"
import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Option, Schema } from "effect"
import { join, resolve } from "node:path"
import { buildAcnBinary } from "../../release/scripts/build/acn"
import { buildCliBinary } from "../../release/scripts/build/cli"
import { buildDesktopApplication, DesktopTarget } from "../../release/scripts/build/desktop"
import { buildDesktopDmg } from "../../release/scripts/apple/desktop"
import { buildLinuxDesktopInstaller } from "../../release/scripts/build/desktop-linux"
import { buildWindowsDesktopInstaller } from "../../release/scripts/build/desktop-windows"
import { ProcessExecutorLive, checkedCommand } from "../src/process"
import { InfrastructureFailure } from "../src/domain"
import { ACN_EXECUTABLE_NAME } from "../../release/src/executables"
import { compileIcnBase, CompiledIcnBase, packageIcnBase } from "../../release/scripts/build/icn-base"
import { buildBackendArtifact } from "../../release/scripts/build/backend"
import { currentHost } from "../../release/src/targets"
import { Backend } from "../src/domain"
import { selectBuildBackendPacks } from "../src/native-build-selection"
import { buildArchive } from "../../release/scripts/build/common"
import { buildMacApp } from "../../release/scripts/apple/build-app"
import { regularAppleFiles } from "../../release/scripts/apple/distribution"
import { acnArchive } from "../../release/src/targets"
import { releaseBundleSizes } from "../../release/src/acquisition"

// Invoked inside a disposable source workspace by the build worker. Release owns all package logic.
const root = resolve(import.meta.dir, "../../..")
export const CompiledCandidate = Schema.Struct({ schemaVersion: Schema.Literal(1), sourceDigest: Digest,
  sourceCommit: ReleaseManifestSchema.fields.sourceCommit, target: DesktopTarget, app: Schema.NonEmptyString,
  version: Schema.NonEmptyString, revision: Schema.Int, rpc: ReleaseManifestSchema.fields.rpc,
  backend: Backend, nativeBase: CompiledIcnBase, nativePacks: Schema.Array(ReleaseArtifactSchema) })
const run = Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const output = resolve(yield* Config.string("LAB_BUILD_OUTPUT"))
  const phase = yield* Config.literal("compile", "package", "all")("LAB_BUILD_PHASE").pipe(Config.withDefault("all"))
  const sourceDigest = yield* Config.string("LAB_BUILD_SOURCE_DIGEST").pipe(Effect.flatMap(Schema.decodeUnknown(Digest)))
  const sourceCommit = yield* Config.string("LAB_BUILD_SOURCE_COMMIT").pipe(Effect.flatMap(Schema.decodeUnknown(ReleaseManifestSchema.fields.sourceCommit)))
  const target = yield* Schema.decodeUnknown(DesktopTarget)({ platform: process.platform, arch: process.arch })
  const host = currentHost()
  const backend = yield* Config.literal("cpu", "metal", "cuda")("LAB_BUILD_BACKEND").pipe(Config.withDefault(host === "darwin-arm64" ? "metal" : "cpu"))
  const packs = yield* selectBuildBackendPacks(host, backend)
  yield* fs.makeDirectory(output, { recursive: true })
  const execute = (args: readonly string[], cwd = root) => checkedCommand(process.execPath, args, {
    cwd: Option.some(cwd), timeoutMs: 30 * 60_000, maxOutputBytes: 32 * 1024 * 1024,
  })
  const receiptPath = join(output, "compiled-candidate.json")
  if (phase !== "package") {
    if (yield* fs.exists(receiptPath)) return yield* new InfrastructureFailure({ operation: "build-source", message: "Compilation requires a fresh output directory" })
    yield* execute(["packages/version/scripts/generate-version.ts"])
    const identity = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ cliVersion: Schema.String, revision: Schema.Int, rpc: ReleaseManifestSchema.fields.rpc })))(
      yield* fs.readFileString(join(root, "packages/release/release-plan.json")))
    yield* execute(["run", "icn:catalog:build-bundle"])
    const nativeBase = yield* compileIcnBase(host)
    const nativePacks: (typeof ReleaseArtifactSchema.Type)[] = []
    for (const pack of packs) {
      const directory = join(output, "native-backends", pack.id)
      yield* Effect.tryPromise({ try: () => buildBackendArtifact(pack.id, directory), catch: error => new InfrastructureFailure({ operation: "build-backend", message: String(error) }) })
      const artifact = yield* fs.readFileString(join(directory, `icn-backend-${pack.id}.artifact.json`)).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(ReleaseArtifactSchema))))
      if (Option.getOrUndefined(artifact.nativeBuild) !== nativeBase.identity.native_build || Option.getOrUndefined(artifact.backendModuleAbi) !== nativeBase.identity.backend_module_abi) {
        return yield* new InfrastructureFailure({ operation: "build-backend", message: "Compiled backend pack differs from native base identity" })
      }
      nativePacks.push(artifact)
    }
    yield* execute(["run", "build"], join(root, "desktop"))
    const bunTarget = `bun-${target.platform === "win32" ? "windows" : target.platform}-${target.arch}`
    const service = yield* Effect.tryPromise({ try: () => buildAcnBinary(bunTarget), catch: () => new InfrastructureFailure({ operation: "build-acn", message: "Release-owned service compilation failed" }) })
    const cli = yield* Effect.tryPromise({ try: () => buildCliBinary(bunTarget), catch: () => new InfrastructureFailure({ operation: "build-cli", message: "Release-owned CLI compilation failed" }) })
    const apps = yield* buildDesktopApplication({ service, cli, version: identity.cliVersion, revision: identity.revision, outputDirectory: join(output, "application") })
    const app = target.platform === "darwin" ? join(apps[0]!, "Magnitude.app") : apps[0]!
    const resources = join(app, target.platform === "darwin" ? "Contents/Resources" : "resources")
    for (const [name, argument] of [[ACN_EXECUTABLE_NAME, "version"], ["magnitude", "--version"]] as const) {
      const observed = yield* checkedCommand(join(resources, `${name}${target.platform === "win32" ? ".exe" : ""}`), [argument])
      if (observed.stdout.trim() !== identity.cliVersion) return yield* new InfrastructureFailure({ operation: "package-identity", message: `${name} version does not match the package` })
    }
    yield* fs.writeFileString(receiptPath, yield* Schema.encode(Schema.parseJson(CompiledCandidate))({ schemaVersion: 1, sourceDigest, sourceCommit, target, app,
      version: identity.cliVersion, revision: identity.revision, rpc: identity.rpc, backend, nativeBase, nativePacks }), { flag: "wx" })
  }
  if (phase === "compile") return
  const compiled = yield* fs.readFileString(receiptPath).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(CompiledCandidate))))
  const app = target.platform === "darwin" ? join(output, "application", `Magnitude-${target.platform}-${target.arch}`, "Magnitude.app")
    : join(output, "application", `Magnitude-${target.platform}-${target.arch}`)
  if (compiled.sourceDigest !== sourceDigest || compiled.sourceCommit !== sourceCommit || compiled.target.platform !== target.platform || compiled.target.arch !== target.arch || compiled.app !== app || compiled.backend !== backend) {
    return yield* new InfrastructureFailure({ operation: "package-source", message: "Compiled candidate does not match this source and native host" })
  }
  const artifacts: (typeof ReleaseArtifactSchema.Type)[] = []
  // Runtime startup reads the service archive's size from the ordinary release manifest,
  // even when this desktop owns its already-bundled service executable.
  const resources = join(app, target.platform === "darwin" ? "Contents/Resources" : "resources")
  const service = join(resources, `${ACN_EXECUTABLE_NAME}${target.platform === "win32" ? ".exe" : ""}`)
  const serviceSources = target.platform === "darwin"
    ? (yield* regularAppleFiles(yield* buildMacApp(join(output, "service-application"), service, compiled.version, compiled.revision)))
      .map(file => ({ ...file, path: `Magnitude.app/${file.path}` }))
    : [{ path: `bin/${ACN_EXECUTABLE_NAME}${target.platform === "win32" ? ".exe" : ""}`, source: service, mode: 0o755 }]
  artifacts.push(yield* Effect.tryPromise({ try: () => buildArchive(join(output, "artifacts", acnArchive(host)), join(output, "artifacts", `acn-${host}.artifact.json`), {
    id: `acn-${host}`, kind: "acn", host: Option.some(host), backend: Option.none(), requiredBaseId: Option.none(),
    nativeBuild: Option.none(), backendModuleAbi: Option.none(), compatibility: Option.none(),
  }, serviceSources), catch: error => new InfrastructureFailure({ operation: "package-service", message: String(error) }) }))
  artifacts.push(yield* packageIcnBase(host, compiled.nativeBase, join(root, "inference/target/catalog-inputs"), join(output, "artifacts")))
  if (compiled.nativePacks.length !== packs.length) return yield* new InfrastructureFailure({ operation: "package-backend", message: "Compiled pack count differs from selected backend" })
  for (const pack of packs) {
    const artifact = compiled.nativePacks.find(artifact => artifact.id === `icn-backend-${pack.id}`)
    if (!artifact || !/^[A-Za-z0-9][A-Za-z0-9._+-]*$/.test(artifact.filename)) return yield* new InfrastructureFailure({ operation: "package-backend", message: "Compiled native pack is missing or has an unsafe filename" })
    yield* fs.copyFile(join(output, "native-backends", pack.id, artifact.filename), join(output, "artifacts", artifact.filename))
    artifacts.push(artifact)
  }
  if (target.platform === "darwin") {
    const result = yield* buildDesktopDmg({ app, output: join(output, "artifacts"), host: target.arch === "arm64" ? "darwin-arm64" : "darwin-x64" })
    artifacts.push(result.artifact, result.updateArtifact)
  }
  else if (target.platform === "linux") {
    for (const format of ["deb", "rpm"] as const) artifacts.push((yield* buildLinuxDesktopInstaller({ app, version: compiled.version, revision: compiled.revision,
      arch: target.arch, format, output: join(output, "artifacts") })).artifact)
  } else {
    const guard = yield* Config.string("LAB_WINDOWS_INSTALL_GUARD")
    const makensis = yield* Config.string("LAB_WINDOWS_NSIS")
    artifacts.push((yield* buildWindowsDesktopInstaller({ app, guard, makensis, version: compiled.version, revision: compiled.revision, output: join(output, "artifacts") })).artifact)
  }
  const manifest = yield* Schema.decodeUnknown(ReleaseManifestSchema)({ schemaVersion: 2, version: compiled.version,
    acnRevision: compiled.revision, rpc: compiled.rpc, plugins: [], tag: `@magnitudedev/cli@${compiled.version}`, sourceCommit,
    artifacts: yield* Effect.forEach(artifacts, artifact => Schema.encode(ReleaseArtifactSchema)(artifact)) }).pipe(Effect.flatMap(validateReleaseManifest))
  yield* releaseBundleSizes(manifest, host)
  yield* fs.writeFileString(join(output, "artifacts", "release-manifest.json"), yield* Schema.encode(Schema.parseJson(ReleaseManifestSchema))(manifest), { flag: "wx" })
  yield* Effect.logInfo(`Packaged candidate: ${output}`)
})
if (import.meta.main) BunRuntime.runMain(run.pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
