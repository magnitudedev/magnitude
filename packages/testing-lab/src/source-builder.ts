import { isWindows } from "./domain"
import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { ArtifactStore } from "./artifact-store"
import { snapshotArtifacts } from "./artifact-input"
import { CaseObservation } from "./case-runner"
import { AssertionFailure, Backend, Digest, Evidence, InfrastructureFailure, Target } from "./domain"
import { buildUpdateAcceptance } from "./update-build"
import { ArtifactInput } from "./inputs"
import { command, CommandOutput, ProcessExecutor } from "./process"
import { extractSource, sha256, SourceManifest } from "./snapshot"

export interface BuildStages {
  readonly evidence: () => readonly (typeof Evidence.Type)[]
  readonly compile: Effect.Effect<CaseObservation, AssertionFailure | InfrastructureFailure>
  readonly package: Effect.Effect<{ readonly input: typeof ArtifactInput.Type; readonly digest: Digest; readonly evidence: readonly (typeof Evidence.Type)[] }, AssertionFailure | InfrastructureFailure>
}
export interface SourceBuilder {
  readonly prepare: (source: SourceManifest, digest: Digest, target: Target, backend: typeof Backend.Type, updates?: boolean) => Effect.Effect<BuildStages, InfrastructureFailure>
}
export const SourceBuilder = Context.GenericTag<SourceBuilder>("@magnitudedev/testing-lab/SourceBuilder")
export const SourceBuildConfig = Schema.Struct({ root: Schema.NonEmptyString, objects: Schema.NonEmptyString,
  environment: Schema.Record({ key: Schema.String, value: Schema.String }) })
const failure = (message: string) => new InfrastructureFailure({ operation: "source-build", message })

/** The source snapshot owns build logic; the guest owns isolation, phases and byte admission. */
export const nativeSourceBuilder = (config: typeof SourceBuildConfig.Type) => Layer.effect(SourceBuilder, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const objects = yield* ArtifactStore
  const executor = yield* ProcessExecutor
  return {
    prepare: (source, digest, target, backend, updates = false) => Effect.gen(function* () {
      const platform = target.os === "macos" ? "darwin" : isWindows(target.os) ? "win32" : "linux"
      if (platform !== process.platform || target.arch !== process.arch) return yield* failure("Build must execute on the selected native OS and architecture")
      if (yield* fs.exists(config.root)) return yield* failure("Source build requires a fresh workspace")
      yield* fs.makeDirectory(config.root, { recursive: true, mode: 0o700 })
      const workspace = join(config.root, "source"), output = join(config.root, "output"), home = join(config.root, "home")
      const cudaRoot = join(config.root, "cuda-toolkit")
      yield* fs.makeDirectory(home, { mode: 0o700 })
      // Match the release ARM64 compiler: GCC on Ubuntu 24.04 cannot compile the SME variants.
      const env = { ...config.environment, ...(backend === "cuda" ? { CUDA_PATH: cudaRoot, CUDACXX: join(cudaRoot, "bin", platform === "win32" ? "nvcc.exe" : "nvcc"),
        PATH: `${join(cudaRoot, "bin")}${platform === "win32" ? ";" : ":"}${config.environment.PATH ?? ""}` } : {}),
        ...(platform === "linux" && target.arch === "arm64" ? { CC: "clang", CXX: "clang++" } : {}), HOME: home, USERPROFILE: home, APPDATA: join(home, "AppData", "Roaming"),
        LOCALAPPDATA: join(home, "AppData", "Local"), XDG_CONFIG_HOME: join(home, ".config"), XDG_CACHE_HOME: join(home, ".cache"),
        LAB_BUILD_OUTPUT: output, LAB_BUILD_SOURCE_DIGEST: digest, LAB_BUILD_SOURCE_COMMIT: source.commit, LAB_BUILD_BACKEND: backend }
      const record = (name: string, result: CommandOutput) => Effect.gen(function* () {
        const wire = yield* Schema.encode(Schema.parseJson(CommandOutput))({ ...result,
          stdout: result.stdout.replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]"), stderr: result.stderr.replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]") })
        const bytes = new TextEncoder().encode(wire), hash = sha256(bytes)
        yield* objects.put(hash, Stream.make(bytes))
        return Evidence.make({ path: `evidence/build-${name}.json`, sha256: hash, bytes: bytes.byteLength })
      })
      const receipts: (typeof Evidence.Type)[] = []
      const invoke = (phase: string, args: readonly string[], extra: Readonly<Record<string, string>> = {}) => Effect.gen(function* () {
        const nativeArgs = process.platform === "win32"
          ? ["-NoProfile", "-NonInteractive", "-File", join(workspace, "packages/testing-lab/scripts/windows-build.ps1"),
            "-BunExecutable", process.execPath, "-ArgumentsBase64", Buffer.from(yield* Schema.encode(Schema.parseJson(Schema.Array(Schema.String)))(args)).toString("base64")]
          : args
        const result = yield* command(process.platform === "win32" ? "pwsh.exe" : process.execPath, nativeArgs, { cwd: Option.some(workspace), env: { ...env, ...extra, LAB_BUILD_PHASE: phase },
          inheritEnv: false, timeoutMs: 60 * 60_000, maxOutputBytes: 32 * 1024 * 1024 }).pipe(Effect.provideService(ProcessExecutor, executor))
        receipts.push(yield* record(phase, result))
        if (result.exitCode !== 0) return yield* new AssertionFailure({ message: `Source ${phase} exited ${result.exitCode}: ${(result.stderr || result.stdout).slice(-1800)}` })
      }).pipe(Effect.mapError(error => error._tag === "AssertionFailure" || error._tag === "InfrastructureFailure" ? error : failure(error.message)))
      const compile = yield* Effect.cached(Effect.gen(function* () {
        yield* extractSource(source, config.objects, workspace).pipe(Effect.provideService(FileSystem.FileSystem, fs))
        if (backend === "cuda") yield* invoke("toolchain", [fileURLToPath(new URL("../scripts/prepare-cuda-toolkit.ts", import.meta.url)), cudaRoot])
        yield* invoke("dependencies", ["install", "--frozen-lockfile"])
        yield* invoke("compile", ["packages/testing-lab/scripts/build-desktop-candidate.ts"])
        return CaseObservation.make({ detail: `Compiled admitted source ${digest} on ${target.artifactHost}`, evidence: [...receipts] })
      }).pipe(Effect.mapError(error => error._tag === "AssertionFailure" || error._tag === "InfrastructureFailure" ? error : failure(error.message))))
      const packaged = yield* Effect.cached(Effect.gen(function* () {
        yield* compile
        yield* invoke("package", ["packages/testing-lab/scripts/build-desktop-candidate.ts"])
        const frozen = yield* snapshotArtifacts(join(output, "artifacts", "release-manifest.json"), config.objects).pipe(Effect.provideService(FileSystem.FileSystem, fs))
        const input = yield* Schema.decodeUnknown(Schema.parseJson(ArtifactInput))(frozen.json)
        if (input.release.sourceCommit !== source.commit) return yield* new AssertionFailure({ message: "Built package changed the source commit identity" })
        const complete = updates ? ArtifactInput.make({ ...input, updateAcceptance: Option.some(yield* buildUpdateAcceptance(digest, input, config.root, config.objects, invoke).pipe(
          Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(ArtifactStore, objects), Effect.provideService(ProcessExecutor, executor))) }) : input
        const wire = yield* Schema.encode(Schema.parseJson(ArtifactInput))(complete)
        const packagedDigest = sha256(wire)
        yield* objects.put(packagedDigest, Stream.make(new TextEncoder().encode(wire)))
        return { input: complete, digest: packagedDigest, evidence: [...receipts] }
      }).pipe(Effect.mapError(error => error._tag === "AssertionFailure" || error._tag === "InfrastructureFailure" ? error : failure(error.message))))
      return { compile, package: packaged, evidence: () => [...receipts] }
    }).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : failure(error.message))),
  } satisfies SourceBuilder
}))
