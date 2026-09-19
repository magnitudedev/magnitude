import { FileSystem } from "@effect/platform"
import { BackendEligibilityReport, IcnBinaryIdentity } from "@magnitudedev/icn-protocol"
import { Data, Effect, Option, Schema } from "effect"
import { basename, delimiter, dirname, resolve } from "node:path"
import { buildIcnBinary } from "../../../../inference/scripts/compile"
import { ICN_EXECUTABLE_NAME } from "../../src/executables"
import { hostById, icnBaseArchive, releaseBuildEnvironment, type HostId } from "../../src/targets"
import { signAppleCode } from "../apple/signing"
import { buildArchive, run, verifyAppleDeploymentTarget, verifyOwnedLoaderPaths } from "./common"
import { isWindowsEngineLibrary, signWindowsCode } from "./windows-signing"

export const CompiledIcnBase = Schema.Struct({ binary: Schema.NonEmptyString, identity: IcnBinaryIdentity,
  backendModules: Schema.Array(Schema.NonEmptyString), runtimeLibraries: Schema.Array(Schema.NonEmptyString) })
export class IcnBaseBuildFailed extends Data.TaggedError("IcnBaseBuildFailed")<{ readonly message: string }> {}
const native = <A>(operation: string, action: () => Promise<A>) => Effect.tryPromise({ try: action,
  catch: error => new IcnBaseBuildFailed({ message: `${operation}: ${error instanceof Error ? error.message : String(error)}` }) })

/** Shared native base compilation, platform closure and signing for release and local candidates. */
export const compileIcnBase = (hostId: HostId) => Effect.gen(function* () {
  const host = hostById(hostId)
  const fs = yield* FileSystem.FileSystem
  const result = yield* native("Compile native CPU base", () => buildIcnBinary({ target: host.bunTarget,
    profile: `base-${host.id}`, features: host.cargoFeatures, buildEnvironment: releaseBuildEnvironment(host) }))
  const modules = result.backendModules.filter(file => basename(file).toLowerCase().includes("cpu"))
  if (!modules.length) return yield* new IcnBaseBuildFailed({ message: `${host.id} ICN base emitted no CPU module` })
  yield* native("Verify native loader closure", () => verifyOwnedLoaderPaths({ host: host.id, executable: result.binary, modules, runtime: result.runtimeLibraries }))
  yield* native("Verify native platform floor", () => verifyAppleDeploymentTarget(host.id, [result.binary, ...result.runtimeLibraries, ...modules]))
  if (host.id.startsWith("darwin-")) {
    for (const file of [result.binary, ...result.runtimeLibraries, ...modules]) {
      yield* signAppleCode(file, `dev.magnitude.inference.${basename(file)}`, file === result.binary ? "native" : "library")
    }
  }
  yield* fs.chmod(result.binary, 0o755)
  const loader = host.id.startsWith("windows-") ? "PATH" : host.id.startsWith("darwin-") ? "DYLD_LIBRARY_PATH" : "LD_LIBRARY_PATH"
  const eligibility = yield* native("Probe compiled native base", () => run([result.binary, "backend-eligibility", "--json"], {
    env: { ...process.env, [loader]: [...result.runtimeLibraries.map(dirname), process.env[loader]].filter(Boolean).join(delimiter) },
  }))
  yield* Schema.decodeUnknown(Schema.parseJson(BackendEligibilityReport))(eligibility)
  if (host.id === "windows-x64-msvc") {
    yield* Effect.forEach([result.binary, ...modules, ...result.runtimeLibraries.filter(file => isWindowsEngineLibrary(basename(file)))], signWindowsCode, { discard: true })
  }
  return CompiledIcnBase.make({ ...result, backendModules: modules })
})

/** Archive the exact compiled and signed inputs; no second native build occurs during packaging. */
export const packageIcnBase = (hostId: HostId, compiled: typeof CompiledIcnBase.Type, catalogRoot: string, output: string) => Effect.gen(function* () {
  const host = hostById(hostId)
  return yield* native("Package native CPU base", () => buildArchive(resolve(output, icnBaseArchive(host.id)), resolve(output, `icn-base-${host.id}.artifact.json`), {
    id: `icn-base-${host.id}`, kind: "icn-base", host: Option.some(host.id), backend: Option.some("cpu"), requiredBaseId: Option.none(),
    nativeBuild: Option.some(compiled.identity.native_build), backendModuleAbi: Option.some(compiled.identity.backend_module_abi), compatibility: Option.none(),
  }, [
    { path: `bin/${ICN_EXECUTABLE_NAME}${host.executableExtension}`, source: compiled.binary, mode: 0o755 },
    { path: "catalog/model-planner-inputs.bundle", source: resolve(catalogRoot, "model-planner-inputs.bundle"), mode: 0o644 },
    ...compiled.runtimeLibraries.map(source => ({ path: `runtime/${basename(source)}`, source, mode: 0o755 })),
    ...compiled.backendModules.map(source => ({ path: `backends/${basename(source)}`, source, mode: 0o755 })),
  ]))
})
