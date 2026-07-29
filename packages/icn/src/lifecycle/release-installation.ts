import { createHash } from "node:crypto"
import { delimiter } from "node:path"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import {
  acquireRelease,
  currentHost,
  IcnInstallationSchema,
  installArtifact,
  NodeArchiveExtractor,
  selectArtifact,
  type ReleaseArtifact,
  type ReleaseManifest,
} from "@magnitudedev/release"
import { Data, Effect, Option, Schema } from "effect"

const CudaEligibility = Schema.Union(
  Schema.TaggedStruct("usable", {
    driverApi: Schema.Int,
    architectures: Schema.Array(Schema.String),
  }),
  Schema.TaggedStruct("absent", { diagnostic: Schema.String }),
  Schema.TaggedStruct("failed", { diagnostic: Schema.String }),
)

const VulkanEligibility = Schema.Union(
  Schema.TaggedStruct("usable", { loaderApi: Schema.Int }),
  Schema.TaggedStruct("absent", { diagnostic: Schema.String }),
  Schema.TaggedStruct("failed", { diagnostic: Schema.String }),
)

const MetalEligibility = Schema.Union(
  Schema.TaggedStruct("usable", {}),
  Schema.TaggedStruct("absent", { diagnostic: Schema.String }),
)

const EligibilityReport = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  cuda: CudaEligibility,
  vulkan: VulkanEligibility,
  metal: MetalEligibility,
})

const BinaryIdentity = Schema.Struct({
  native_build: Schema.String,
  backend_module_abi: Schema.String,
})

type BinaryIdentity = typeof BinaryIdentity.Type
type SelectedBackend = {
  readonly backend: "cpu" | "metal" | "cuda" | "vulkan"
  readonly pack: Option.Option<ReleaseArtifact>
}

export class ReleaseIcnInstallationError extends Data.TaggedError(
  "ReleaseIcnInstallationError",
)<{
  readonly stage: "acquire" | "probe" | "compose" | "verify"
  readonly message: string
}> {}

export interface ReleaseIcnInstallation {
  readonly binaryPath: string
  readonly declarationPath: string
  readonly environment: Readonly<Record<string, string>>
}

const installationError = (
  stage: ReleaseIcnInstallationError["stage"],
  message: string,
) => new ReleaseIcnInstallationError({ stage, message })

const executableName = () => process.platform === "win32"
  ? "magnitude-icn.exe"
  : "magnitude-icn"

const loaderEnvironment = (runtime: string): Readonly<Record<string, string>> => {
  const key = process.platform === "win32"
    ? "PATH"
    : process.platform === "darwin"
      ? "DYLD_LIBRARY_PATH"
      : "LD_LIBRARY_PATH"
  const inherited = process.env[key]
  return { [key]: inherited ? `${runtime}${delimiter}${inherited}` : runtime }
}

const run = (
  command: readonly [string, ...string[]],
  environment: Readonly<Record<string, string>>,
): Effect.Effect<
  string,
  ReleaseIcnInstallationError,
  CommandExecutor.CommandExecutor
> =>
  Command.make(...command).pipe(
    Command.env(environment),
    Command.string,
    Effect.timeout("15 seconds"),
    Effect.mapError(() => installationError("probe", `ICN command failed: ${command[1] ?? ""}`)),
  )

const isNonEmptyFile = (
  path: string,
): Effect.Effect<boolean, never, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const info = yield* fs.stat(path).pipe(Effect.option)
    return Option.isSome(info) && info.value.type === "File" && Number(info.value.size) > 0
  })

const validateArtifactDirectory = (
  artifact: ReleaseArtifact,
  directory: string,
): Effect.Effect<boolean, never, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const directoryInfo = yield* fs.stat(directory).pipe(Effect.option)
    if (Option.isNone(directoryInfo) || directoryInfo.value.type !== "Directory") return false
    if (artifact.kind === "icn-base") {
      const files = yield* Effect.all([
        isNonEmptyFile(path.join(directory, "bin", executableName())),
        isNonEmptyFile(path.join(directory, "catalog", "release-catalog.lock.json")),
        isNonEmptyFile(path.join(directory, "catalog", "model-planner-inputs.bundle")),
      ])
      if (files.some((present) => !present)) return false
    }
    const backends = yield* fs.readDirectory(path.join(directory, "backends")).pipe(
      Effect.option,
    )
    return Option.isSome(backends) && backends.value.length > 0
  })

const ensureArtifact = (
  baseUrl: string,
  version: string,
  artifact: ReleaseArtifact,
  root: string,
): Effect.Effect<
  string,
  ReleaseIcnInstallationError,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const destination = path.join(root, artifact.id, artifact.sha256)
    if (!(yield* validateArtifactDirectory(artifact, destination))) {
      yield* fs.remove(destination, { recursive: true, force: true }).pipe(
        Effect.mapError(() => installationError("acquire", `unable to replace ${artifact.id}`)),
      )
      yield* installArtifact(baseUrl, version, artifact, destination).pipe(
        Effect.provide(NodeArchiveExtractor),
        Effect.mapError((cause) => installationError("acquire", cause.message)),
      )
    }
    if (!(yield* validateArtifactDirectory(artifact, destination))) {
      return yield* installationError(
        "verify",
        `${artifact.id} did not produce a complete installation`,
      )
    }
    return destination
  })

const readIdentity = (
  base: string,
  artifact: ReleaseArtifact,
): Effect.Effect<
  BinaryIdentity,
  ReleaseIcnInstallationError,
  CommandExecutor.CommandExecutor | Path.Path
> =>
  Effect.gen(function* () {
    const path = yield* Path.Path
    const value = yield* run(
      [path.join(base, "bin", executableName()), "version", "--json"],
      loaderEnvironment(path.join(base, "runtime")),
    ).pipe(
      Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(BinaryIdentity))),
      Effect.mapError(() => installationError("verify", "ICN base identity is malformed")),
    )
    if (
      Option.getOrUndefined(artifact.nativeBuild) !== value.native_build ||
      Option.getOrUndefined(artifact.backendModuleAbi) !== value.backend_module_abi
    ) {
      return yield* installationError(
        "verify",
        "ICN base identity differs from the release manifest",
      )
    }
    return value
  })

const compatibleCuda = (
  manifest: ReleaseManifest,
  driverApi: number,
  architectures: readonly string[],
): Option.Option<ReleaseArtifact> =>
  Option.fromNullable(manifest.artifacts.find((artifact) => {
    if (
      artifact.kind !== "icn-backend" ||
      Option.getOrUndefined(artifact.host) !== currentHost() ||
      Option.getOrUndefined(artifact.backend) !== "cuda" ||
      Option.isNone(artifact.compatibility) ||
      artifact.compatibility.value.kind !== "cuda"
    ) return false
    const compatibility = artifact.compatibility.value
    return driverApi >= compatibility.minimumDriverApi &&
      architectures.some((architecture) =>
        Number.parseInt(architecture, 10) >= compatibility.minimumArchitecture
      )
  }))

const vulkanVersionAtLeast = (encoded: number, required: string): boolean => {
  const [requiredMajor = 0, requiredMinor = 0] = required.split(".").map(Number)
  const major = (encoded >>> 22) & 0x7f
  const minor = (encoded >>> 12) & 0x3ff
  return major > requiredMajor || (major === requiredMajor && minor >= requiredMinor)
}

const compatibleVulkan = (
  manifest: ReleaseManifest,
  loaderApi: number,
): Option.Option<ReleaseArtifact> =>
  Option.fromNullable(manifest.artifacts.find((artifact) =>
    artifact.kind === "icn-backend" &&
    Option.getOrUndefined(artifact.host) === currentHost() &&
    Option.getOrUndefined(artifact.backend) === "vulkan" &&
    Option.isSome(artifact.compatibility) &&
    artifact.compatibility.value.kind === "vulkan" &&
    vulkanVersionAtLeast(loaderApi, artifact.compatibility.value.minimumApi)
  ))

const selectBackend = (
  manifest: ReleaseManifest,
  base: string,
): Effect.Effect<
  SelectedBackend,
  ReleaseIcnInstallationError,
  CommandExecutor.CommandExecutor | Path.Path
> =>
  Effect.gen(function* () {
    const path = yield* Path.Path
    const report = yield* run(
      [path.join(base, "bin", executableName()), "backend-eligibility", "--json"],
      loaderEnvironment(path.join(base, "runtime")),
    ).pipe(
      Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(EligibilityReport))),
      Effect.mapError(() => installationError("probe", "ICN backend eligibility report is malformed")),
    )

    if (currentHost() === "darwin-arm64") {
      if (report.metal._tag !== "usable") {
        return yield* installationError("probe", report.metal.diagnostic)
      }
      const pack = yield* selectArtifact(
        manifest,
        "icn-backend",
        currentHost(),
        "metal",
      ).pipe(Effect.mapError((cause) => installationError("acquire", cause.message)))
      return { backend: "metal", pack: Option.some(pack) }
    }

    if (report.cuda._tag === "failed") {
      return yield* installationError("probe", report.cuda.diagnostic)
    }
    if (report.cuda._tag === "usable") {
      const pack = compatibleCuda(
        manifest,
        report.cuda.driverApi,
        report.cuda.architectures,
      )
      if (Option.isSome(pack)) return { backend: "cuda", pack }
    }

    if (report.vulkan._tag === "failed") {
      return yield* installationError("probe", report.vulkan.diagnostic)
    }
    if (report.vulkan._tag === "usable") {
      const pack = compatibleVulkan(manifest, report.vulkan.loaderApi)
      if (Option.isSome(pack)) return { backend: "vulkan", pack }
    }
    return { backend: "cpu", pack: Option.none() }
  })

const compositionId = (
  manifestSha256: string,
  base: ReleaseArtifact,
  selected: SelectedBackend,
  native: BinaryIdentity,
): string =>
  createHash("sha256").update(JSON.stringify({
    manifestSha256,
    base: base.sha256,
    pack: Option.match(selected.pack, {
      onNone: () => null,
      onSome: (artifact) => artifact.sha256,
    }),
    backend: selected.backend,
    nativeBuild: native.native_build,
    backendModuleAbi: native.backend_module_abi,
  })).digest("hex")

const validateComposition = (
  root: string,
): Effect.Effect<
  ReleaseIcnInstallation,
  ReleaseIcnInstallationError,
  CommandExecutor.CommandExecutor | Path.Path
> =>
  Effect.gen(function* () {
    const path = yield* Path.Path
    const binaryPath = path.join(root, "bin", executableName())
    const declarationPath = path.join(root, "installation.json")
    const environment = loaderEnvironment(path.join(root, "runtime"))
    yield* run(
      [binaryPath, "installation-check", "--installation", declarationPath],
      environment,
    ).pipe(
      Effect.mapError(() => installationError("verify", "ICN installation validation failed")),
    )
    return { binaryPath, declarationPath, environment }
  })

const copyDirectoryContents = (
  source: string,
  destination: string,
): Effect.Effect<void, ReleaseIcnInstallationError, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    yield* fs.makeDirectory(destination, { recursive: true, mode: 0o700 })
    const entries = yield* fs.readDirectory(source)
    yield* Effect.forEach(entries, (entry) =>
      fs.copy(path.join(source, entry), path.join(destination, entry), {
        overwrite: false,
        preserveTimestamps: true,
      }), { concurrency: 4 })
  }).pipe(
    Effect.mapError(() => installationError("compose", "unable to compose ICN release files")),
  )

const publishComposition = (
  root: string,
  base: string,
  pack: Option.Option<string>,
  selected: SelectedBackend,
  native: BinaryIdentity,
): Effect.Effect<
  ReleaseIcnInstallation,
  ReleaseIcnInstallationError,
  FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
> =>
  Effect.scoped(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const parent = path.dirname(root)
      yield* fs.makeDirectory(parent, { recursive: true, mode: 0o700 }).pipe(
        Effect.mapError(() => installationError("compose", "unable to create ICN release directory")),
      )
      const staging = yield* fs.makeTempDirectoryScoped({
        directory: parent,
        prefix: ".composition-",
      }).pipe(
        Effect.mapError(() => installationError("compose", "unable to stage ICN composition")),
      )
      for (const directory of ["bin", "catalog", "runtime", "backends"]) {
        yield* copyDirectoryContents(
          path.join(base, directory),
          path.join(staging, directory),
        )
      }
      if (Option.isSome(pack)) {
        const runtime = path.join(pack.value, "runtime")
        if (yield* fs.exists(runtime).pipe(Effect.orElseSucceed(() => false))) {
          yield* copyDirectoryContents(
            runtime,
            path.join(staging, "runtime"),
          )
        }
        yield* copyDirectoryContents(
          path.join(pack.value, "backends"),
          path.join(staging, "backends"),
        )
      }
      const declaration = {
        schemaVersion: 1 as const,
        backend: selected.backend,
        nativeBuild: native.native_build,
        backendModuleAbi: native.backend_module_abi,
      }
      const serialized = yield* Schema.encode(
        Schema.parseJson(IcnInstallationSchema),
      )(declaration).pipe(
        Effect.mapError(() => installationError("compose", "unable to encode ICN installation")),
      )
      yield* fs.writeFileString(path.join(staging, "installation.json"), `${serialized}\n`, {
        flag: "wx",
        mode: 0o600,
      }).pipe(
        Effect.mapError(() => installationError("compose", "unable to write ICN installation")),
      )
      yield* validateComposition(staging)
      yield* fs.rename(staging, root).pipe(
        Effect.catchAll((renameError) =>
          validateComposition(root).pipe(
            Effect.asVoid,
            Effect.mapError(() => renameError),
          )
        ),
        Effect.mapError(() => installationError("compose", "unable to publish ICN composition")),
      )
      return yield* validateComposition(root)
    }),
  )

export const resolveReleaseIcnInstallation = (
  version: string,
  dataDir: string,
  baseUrl: string,
): Effect.Effect<
  ReleaseIcnInstallation,
  ReleaseIcnInstallationError,
  | FileSystem.FileSystem
  | Path.Path
  | HttpClient.HttpClient
  | CommandExecutor.CommandExecutor
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const host = currentHost()
    const releaseRoot = path.join(dataDir, "releases")
    const release = yield* acquireRelease(
      baseUrl,
      version,
      path.join(releaseRoot, "manifests", version),
    ).pipe(Effect.mapError((cause) => installationError("acquire", cause.message)))
    const baseArtifact = yield* selectArtifact(
      release.manifest,
      "icn-base",
      host,
      "cpu",
    ).pipe(Effect.mapError((cause) => installationError("acquire", cause.message)))
    const artifactRoot = path.join(releaseRoot, "artifacts", version, host)
    const prepare = Effect.suspend(() =>
      Effect.gen(function* () {
        const base = yield* ensureArtifact(baseUrl, version, baseArtifact, artifactRoot)
        const native = yield* readIdentity(base, baseArtifact)
        const selected = yield* selectBackend(release.manifest, base)
        const pack = yield* Option.match(selected.pack, {
          onNone: () => Effect.succeed(Option.none<string>()),
          onSome: (artifact) =>
            ensureArtifact(baseUrl, version, artifact, artifactRoot).pipe(
              Effect.map(Option.some),
            ),
        })
        const id = compositionId(
          release.manifestSha256,
          baseArtifact,
          selected,
          native,
        )
        const root = path.join(releaseRoot, "icn", version, host, id)
        const existing = yield* validateComposition(root).pipe(Effect.option)
        return Option.isSome(existing)
          ? existing.value
          : yield* fs.remove(root, { recursive: true, force: true }).pipe(
            Effect.mapError(() =>
              installationError("compose", "unable to replace invalid ICN composition")
            ),
            Effect.zipRight(publishComposition(
              root,
              base,
              pack,
              selected,
              native,
            )),
          )
      })
    )
    return yield* prepare.pipe(
      Effect.catchAll((cause) =>
        cause.stage === "verify" || cause.stage === "compose"
          ? fs.remove(artifactRoot, { recursive: true, force: true }).pipe(
            Effect.mapError(() =>
              installationError("acquire", "unable to invalidate corrupt ICN artifacts")
            ),
            Effect.zipRight(prepare),
          )
          : Effect.fail(cause)
      ),
    )
  })
