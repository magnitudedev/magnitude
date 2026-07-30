import { createHash } from "node:crypto"
import { delimiter } from "node:path"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import {
  BackendEligibilityReport,
  IcnBinaryIdentity,
  IcnInstallationDeclaration,
} from "@magnitudedev/icn-protocol"
import {
  acquireRelease,
  currentHost,
  installArtifact,
  NodeArchiveExtractor,
  releaseBundleSizes,
  selectArtifact,
  type ReleaseArtifact,
  type ReleaseManifest,
} from "@magnitudedev/release"
import { IcnPreparationReporter } from "./preparation.js"
import { Data, Effect, Option, Schema } from "effect"
type SelectedBackend = {
  readonly backend: "cpu" | "metal" | "cuda" | "vulkan"
  readonly pack: Option.Option<ReleaseArtifact>
}

export class ReleaseIcnInstallationError extends Data.TaggedError(
  "ReleaseIcnInstallationError"
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
  message: string
) => new ReleaseIcnInstallationError({ stage, message })

const executableName = () =>
  process.platform === "win32" ? "magnitude-icn.exe" : "magnitude-icn"

const loaderEnvironment = (
  runtime: string
): Readonly<Record<string, string>> => {
  const key =
    process.platform === "win32"
      ? "PATH"
      : process.platform === "darwin"
      ? "DYLD_LIBRARY_PATH"
      : "LD_LIBRARY_PATH"
  const inherited = process.env[key]
  return { [key]: inherited ? `${runtime}${delimiter}${inherited}` : runtime }
}

const run = (
  command: readonly [string, ...string[]],
  environment: Readonly<Record<string, string>>
): Effect.Effect<
  string,
  ReleaseIcnInstallationError,
  CommandExecutor.CommandExecutor
> =>
  Command.make(...command).pipe(
    Command.env(environment),
    Command.string,
    Effect.timeout("15 seconds"),
    Effect.mapError(() =>
      installationError("probe", `ICN command failed: ${command[1] ?? ""}`)
    )
  )

const isNonEmptyFile = (
  path: string
): Effect.Effect<boolean, never, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const info = yield* fs.stat(path).pipe(Effect.option)
    return (
      Option.isSome(info) &&
      info.value.type === "File" &&
      Number(info.value.size) > 0
    )
  })

const validateArtifactDirectory = (
  artifact: ReleaseArtifact,
  directory: string
): Effect.Effect<boolean, never, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const directoryInfo = yield* fs.stat(directory).pipe(Effect.option)
    if (
      Option.isNone(directoryInfo) ||
      directoryInfo.value.type !== "Directory"
    )
      return false
    if (artifact.kind === "icn-base") {
      const files = yield* Effect.all([
        isNonEmptyFile(path.join(directory, "bin", executableName())),
        isNonEmptyFile(
          path.join(directory, "catalog", "release-catalog.lock.json")
        ),
        isNonEmptyFile(
          path.join(directory, "catalog", "model-planner-inputs.bundle")
        ),
      ])
      if (files.some((present) => !present)) return false
    }
    const backends = yield* fs
      .readDirectory(path.join(directory, "backends"))
      .pipe(Effect.option)
    return Option.isSome(backends) && backends.value.length > 0
  })

const ensureArtifact = (
  baseUrl: string,
  version: string,
  artifact: ReleaseArtifact,
  root: string
): Effect.Effect<
  string,
  ReleaseIcnInstallationError,
  | FileSystem.FileSystem
  | Path.Path
  | HttpClient.HttpClient
  | IcnPreparationReporter
> =>
  Effect.gen(function* () {
    const reporter = yield* IcnPreparationReporter
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const destination = path.join(root, artifact.id, artifact.sha256)
    if (!(yield* validateArtifactDirectory(artifact, destination))) {
      yield* reporter.report({ _tag: "InstallationRequired" })
      yield* fs
        .remove(destination, { recursive: true, force: true })
        .pipe(
          Effect.mapError(() =>
            installationError("acquire", `unable to replace ${artifact.id}`)
          )
        )
      yield* installArtifact(baseUrl, version, artifact, destination, {
        observer: Option.some({
          report: (event) =>
            reporter.report({
              _tag: "Artifact",
              artifact:
                artifact.kind === "icn-base" ? "Base" : "Accelerator",
              event,
            }),
        }),
      }).pipe(
        Effect.provide(NodeArchiveExtractor),
        Effect.mapError((cause) => installationError("acquire", cause.message))
      )
    }
    if (!(yield* validateArtifactDirectory(artifact, destination))) {
      return yield* installationError(
        "verify",
        `${artifact.id} did not produce a complete installation`
      )
    }
    return destination
  })

const readIdentity = (
  base: string,
  artifact: ReleaseArtifact
): Effect.Effect<
  IcnBinaryIdentity,
  ReleaseIcnInstallationError,
  CommandExecutor.CommandExecutor | Path.Path
> =>
  Effect.gen(function* () {
    const path = yield* Path.Path
    const value = yield* run(
      [path.join(base, "bin", executableName()), "version", "--json"],
      loaderEnvironment(path.join(base, "runtime"))
    ).pipe(
      Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(IcnBinaryIdentity))),
      Effect.mapError(() =>
        installationError("verify", "ICN base identity is malformed")
      )
    )
    if (
      Option.getOrUndefined(artifact.nativeBuild) !== value.native_build ||
      Option.getOrUndefined(artifact.backendModuleAbi) !==
        value.backend_module_abi
    ) {
      return yield* installationError(
        "verify",
        "ICN base identity differs from the release manifest"
      )
    }
    return value
  })

const compatibleCuda = (
  manifest: ReleaseManifest,
  driverApi: number,
  architectures: readonly string[]
): Option.Option<ReleaseArtifact> =>
  Option.fromNullable(
    manifest.artifacts.find((artifact) => {
      if (
        artifact.kind !== "icn-backend" ||
        Option.getOrUndefined(artifact.host) !== currentHost() ||
        Option.getOrUndefined(artifact.backend) !== "cuda" ||
        Option.isNone(artifact.compatibility) ||
        artifact.compatibility.value.kind !== "cuda"
      )
        return false
      const compatibility = artifact.compatibility.value
      return (
        driverApi >= compatibility.minimumDriverApi &&
        architectures.some(
          (architecture) =>
            Number.parseInt(architecture, 10) >=
            compatibility.minimumArchitecture
        )
      )
    })
  )

const vulkanVersionAtLeast = (encoded: number, required: string): boolean => {
  const [requiredMajor = 0, requiredMinor = 0] = required.split(".").map(Number)
  const major = (encoded >>> 22) & 0x7f
  const minor = (encoded >>> 12) & 0x3ff
  return (
    major > requiredMajor || (major === requiredMajor && minor >= requiredMinor)
  )
}

const compatibleVulkan = (
  manifest: ReleaseManifest,
  loaderApi: number
): Option.Option<ReleaseArtifact> =>
  Option.fromNullable(
    manifest.artifacts.find(
      (artifact) =>
        artifact.kind === "icn-backend" &&
        Option.getOrUndefined(artifact.host) === currentHost() &&
        Option.getOrUndefined(artifact.backend) === "vulkan" &&
        Option.isSome(artifact.compatibility) &&
        artifact.compatibility.value.kind === "vulkan" &&
        vulkanVersionAtLeast(loaderApi, artifact.compatibility.value.minimumApi)
    )
  )

const selectBackend = (
  manifest: ReleaseManifest,
  base: string
): Effect.Effect<
  SelectedBackend,
  ReleaseIcnInstallationError,
  CommandExecutor.CommandExecutor | Path.Path
> =>
  Effect.gen(function* () {
    const path = yield* Path.Path
    const report = yield* run(
      [
        path.join(base, "bin", executableName()),
        "backend-eligibility",
        "--json",
      ],
      loaderEnvironment(path.join(base, "runtime"))
    ).pipe(
      Effect.flatMap(
        Schema.decodeUnknown(Schema.parseJson(BackendEligibilityReport))
      ),
      Effect.mapError(() =>
        installationError(
          "probe",
          "ICN backend eligibility report is malformed"
        )
      )
    )

    if (currentHost() === "darwin-arm64") {
      if (report.metal.state !== "usable") {
        return yield* installationError("probe", report.metal.diagnostic)
      }
      const pack = yield* selectArtifact(
        manifest,
        "icn-backend",
        currentHost(),
        "metal"
      ).pipe(
        Effect.mapError((cause) => installationError("acquire", cause.message))
      )
      return { backend: "metal", pack: Option.some(pack) }
    }

    if (report.cuda.state === "failed") {
      return yield* installationError("probe", report.cuda.diagnostic)
    }
    if (report.cuda.state === "usable") {
      const pack = compatibleCuda(
        manifest,
        report.cuda.driverApi,
        report.cuda.architectures
      )
      if (Option.isSome(pack)) return { backend: "cuda", pack }
    }

    if (report.vulkan.state === "failed") {
      return yield* installationError("probe", report.vulkan.diagnostic)
    }
    if (report.vulkan.state === "usable") {
      const pack = compatibleVulkan(manifest, report.vulkan.loaderApi)
      if (Option.isSome(pack)) return { backend: "vulkan", pack }
    }
    return { backend: "cpu", pack: Option.none() }
  })

const compositionId = (
  manifestSha256: string,
  base: ReleaseArtifact,
  selected: SelectedBackend,
  native: IcnBinaryIdentity
): string =>
  createHash("sha256")
    .update(
      JSON.stringify({
        manifestSha256,
        base: base.sha256,
        pack: Option.match(selected.pack, {
          onNone: () => null,
          onSome: (artifact) => artifact.sha256,
        }),
        backend: selected.backend,
        nativeBuild: native.native_build,
        backendModuleAbi: native.backend_module_abi,
      })
    )
    .digest("hex")

const inspectComposition = (
  root: string,
  selected: SelectedBackend,
  native: IcnBinaryIdentity,
): Effect.Effect<
  ReleaseIcnInstallation,
  ReleaseIcnInstallationError,
  FileSystem.FileSystem | Path.Path
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const binaryPath = path.join(root, "bin", executableName())
    const declarationPath = path.join(root, "installation.json")
    const environment = loaderEnvironment(path.join(root, "runtime"))
    const declaration = yield* fs.readFileString(declarationPath).pipe(
      Effect.flatMap(
        Schema.decodeUnknown(Schema.parseJson(IcnInstallationDeclaration))
      ),
      Effect.mapError(() =>
        installationError(
          "verify",
          "ICN installation declaration is invalid"
        )
      )
    )
    if (
      declaration.backend !== selected.backend ||
      declaration.nativeBuild !== native.native_build ||
      declaration.backendModuleAbi !== native.backend_module_abi
    ) {
      return yield* installationError(
        "verify",
        "ICN installation declaration does not match its release"
      )
    }
    const requiredFiles = yield* Effect.all([
      isNonEmptyFile(binaryPath),
      isNonEmptyFile(
        path.join(root, "catalog", "release-catalog.lock.json")
      ),
      isNonEmptyFile(
        path.join(root, "catalog", "model-planner-inputs.bundle")
      ),
    ])
    const backends = yield* fs
      .readDirectory(path.join(root, "backends"))
      .pipe(Effect.option)
    if (
      requiredFiles.some((present) => !present) ||
      Option.isNone(backends) ||
      backends.value.length === 0
    ) {
      return yield* installationError(
        "verify",
        "ICN installation is structurally incomplete"
      )
    }
    return { binaryPath, declarationPath, environment }
  })

const copyDirectoryContents = (
  source: string,
  destination: string
): Effect.Effect<
  void,
  ReleaseIcnInstallationError,
  FileSystem.FileSystem | Path.Path
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    yield* fs.makeDirectory(destination, { recursive: true, mode: 0o700 })
    const entries = yield* fs.readDirectory(source)
    yield* Effect.forEach(
      entries,
      (entry) =>
        fs.copy(path.join(source, entry), path.join(destination, entry), {
          overwrite: false,
          preserveTimestamps: true,
        }),
      { concurrency: 4 }
    )
  }).pipe(
    Effect.mapError(() =>
      installationError("compose", "unable to compose ICN release files")
    )
  )

const publishComposition = (
  root: string,
  base: string,
  pack: Option.Option<string>,
  selected: SelectedBackend,
  native: IcnBinaryIdentity
): Effect.Effect<
  ReleaseIcnInstallation,
  ReleaseIcnInstallationError,
  | FileSystem.FileSystem
  | Path.Path
  | CommandExecutor.CommandExecutor
  | IcnPreparationReporter
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const reporter = yield* IcnPreparationReporter
      const parent = path.dirname(root)
    yield* fs
      .makeDirectory(parent, { recursive: true, mode: 0o700 })
      .pipe(
        Effect.mapError(() =>
          installationError(
            "compose",
            "unable to create ICN release directory"
          )
        )
      )
    return yield* Effect.acquireUseRelease(
      fs
        .makeTempDirectory({
          directory: parent,
          prefix: ".composition-",
        })
        .pipe(
          Effect.mapError(() =>
            installationError("compose", "unable to stage ICN composition")
          )
        ),
      (staging) =>
        Effect.gen(function* () {
          for (const directory of ["bin", "catalog", "runtime", "backends"]) {
            yield* copyDirectoryContents(
              path.join(base, directory),
              path.join(staging, directory)
            )
          }
          if (Option.isSome(pack)) {
            const runtime = path.join(pack.value, "runtime")
            if (
              yield* fs.exists(runtime).pipe(Effect.orElseSucceed(() => false))
            ) {
              yield* copyDirectoryContents(
                runtime,
                path.join(staging, "runtime")
              )
            }
            yield* copyDirectoryContents(
              path.join(pack.value, "backends"),
              path.join(staging, "backends")
            )
          }
          const declaration = {
            schemaVersion: 1 as const,
            backend: selected.backend,
            nativeBuild: native.native_build,
            backendModuleAbi: native.backend_module_abi,
          }
          const serialized = yield* Schema.encode(
            Schema.parseJson(IcnInstallationDeclaration)
          )(declaration).pipe(
            Effect.mapError(() =>
              installationError("compose", "unable to encode ICN installation")
            )
          )
          yield* fs
            .writeFileString(
              path.join(staging, "installation.json"),
              `${serialized}\n`,
              {
                flag: "wx",
                mode: 0o600,
              }
            )
            .pipe(
              Effect.mapError(() =>
                installationError(
                  "compose",
                  "unable to write ICN installation"
                )
              )
            )
          yield* inspectComposition(staging, selected, native)
          yield* fs.rename(staging, root).pipe(
            Effect.catchAll((renameError) =>
              inspectComposition(root, selected, native).pipe(
                Effect.asVoid,
                Effect.mapError(() => renameError)
              )
            ),
            Effect.mapError(() =>
              installationError(
                "compose",
                "unable to publish ICN composition"
              )
            )
          )
          return {
            binaryPath: path.join(root, "bin", executableName()),
            declarationPath: path.join(root, "installation.json"),
            environment: loaderEnvironment(path.join(root, "runtime")),
          }
        }),
      (staging) =>
        fs
          .remove(staging, { recursive: true, force: true })
          .pipe(Effect.ignore)
    )
  })

export const resolveReleaseIcnInstallation = (
  version: string,
  dataDir: string,
  baseUrl: string
): Effect.Effect<
  ReleaseIcnInstallation,
  ReleaseIcnInstallationError,
  | FileSystem.FileSystem
  | Path.Path
  | HttpClient.HttpClient
  | CommandExecutor.CommandExecutor
  | IcnPreparationReporter
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const reporter = yield* IcnPreparationReporter
    const host = currentHost()
    const releaseRoot = path.join(dataDir, "releases")
    const release = yield* acquireRelease(
      baseUrl,
      version,
      path.join(releaseRoot, "manifests", version)
    ).pipe(
      Effect.mapError((cause) => installationError("acquire", cause.message))
    )
    const bundleSizes = yield* releaseBundleSizes(
      release.manifest,
      host,
    ).pipe(
      Effect.mapError((cause) => installationError("acquire", cause.message)),
    )
    yield* reporter.report({ _tag: "Planned", plan: bundleSizes })
    const baseArtifact = yield* selectArtifact(
      release.manifest,
      "icn-base",
      host,
      "cpu"
    ).pipe(
      Effect.mapError((cause) => installationError("acquire", cause.message))
    )
    const artifactRoot = path.join(releaseRoot, "artifacts", version, host)
    const prepare = Effect.suspend(() =>
      Effect.gen(function* () {
        const base = yield* ensureArtifact(
          baseUrl,
          version,
          baseArtifact,
          artifactRoot
        )
        const native = yield* readIdentity(base, baseArtifact)
        const selected = yield* selectBackend(release.manifest, base)
        yield* reporter.report({
          _tag: "Planned",
          plan: {
            daemonBytes: bundleSizes.daemonBytes,
            inferenceEngineBytes:
              baseArtifact.bytes +
              Option.match(selected.pack, {
                onNone: () => 0,
                onSome: (artifact) => artifact.bytes,
              }),
            inferenceEngineBytesExact: true,
          },
        })
        const pack = yield* Option.match(selected.pack, {
          onNone: () => Effect.succeed(Option.none<string>()),
          onSome: (artifact) =>
            ensureArtifact(baseUrl, version, artifact, artifactRoot).pipe(
              Effect.map(Option.some)
            ),
        })
        const id = compositionId(
          release.manifestSha256,
          baseArtifact,
          selected,
          native
        )
        const root = path.join(releaseRoot, "icn", version, host, id)
        const existing = yield* inspectComposition(
          root,
          selected,
          native,
        ).pipe(Effect.option)
        return Option.isSome(existing)
          ? existing.value
          : yield* reporter.report({ _tag: "InstallationRequired" }).pipe(
              Effect.zipRight(
                fs.remove(root, { recursive: true, force: true }).pipe(
                  Effect.mapError(() =>
                    installationError(
                      "compose",
                      "unable to replace invalid ICN composition"
                    )
                  ),
                  Effect.zipRight(
                    publishComposition(root, base, pack, selected, native)
                  )
                )
              )
            )
      })
    )
    return yield* prepare.pipe(
      Effect.catchAll((cause) =>
        cause.stage === "verify" || cause.stage === "compose"
          ? fs.remove(artifactRoot, { recursive: true, force: true }).pipe(
              Effect.mapError(() =>
                installationError(
                  "acquire",
                  "unable to invalidate corrupt ICN artifacts"
                )
              ),
              Effect.zipRight(prepare)
            )
          : Effect.fail(cause)
      )
    )
  })
