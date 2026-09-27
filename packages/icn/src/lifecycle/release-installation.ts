import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import {
  IcnBinaryIdentity,
  IcnInstallationDeclaration,
} from "@magnitudedev/icn-protocol"
import {
  acquireRelease,
  currentHost,
  installArtifact,
  ICN_EXECUTABLE_NAME,
  hostById,
  inferenceRequiredPaths,
  type HostId,
  NodeArchiveExtractor,
  releaseBundleSizes,
  selectArtifact,
  type ReleaseArtifact,
} from "@magnitudedev/release"
import { IcnPreparationReporter } from "./preparation.js"
import { installationLoaderEnvironment, installationNativePath } from "./installation-environment.js"
import { Data, Effect, Fiber, Option, Schema, Stream } from "effect"

export class ReleaseIcnInstallationError extends Data.TaggedError(
  "ReleaseIcnInstallationError"
)<{
  readonly stage: "acquire" | "probe" | "declare" | "verify"
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
  `${ICN_EXECUTABLE_NAME}${process.platform === "win32" ? ".exe" : ""}`

const MAXIMUM_COMMAND_OUTPUT = 64 * 1024

const outputTail = (stream: Stream.Stream<Uint8Array, unknown>): Effect.Effect<string, unknown> =>
  stream.pipe(
    Stream.decodeText(),
    Stream.runFold("", (tail, chunk) => {
      const combined = tail + chunk
      return combined.length <= MAXIMUM_COMMAND_OUTPUT
        ? combined
        : combined.slice(combined.length - MAXIMUM_COMMAND_OUTPUT)
    }),
  )

const run = (
  command: readonly [string, ...string[]],
  environment: Readonly<Record<string, string>>
): Effect.Effect<
  string,
  ReleaseIcnInstallationError,
  CommandExecutor.CommandExecutor
> =>
  Effect.scoped(Effect.gen(function* () {
    const process = yield* Command.start(Command.make(installationNativePath(command[0]), ...command.slice(1)).pipe(
      Command.env(environment),
    ))
    const stdout = yield* outputTail(process.stdout).pipe(Effect.forkScoped)
    const stderr = yield* outputTail(process.stderr).pipe(Effect.forkScoped)
    const exitCode = Number(yield* process.exitCode)
    const [stdoutText, stderrText] = yield* Effect.all([
      Fiber.join(stdout),
      Fiber.join(stderr),
    ])
    if (exitCode !== 0) {
      return yield* installationError(
        "probe",
        `ICN command ${command[1] ?? ""} exited with code ${exitCode}${stderrText.trim() ? `:\n${stderrText.trim()}` : ""}`,
      )
    }
    return stdoutText
  })).pipe(
    Effect.mapError((cause) => cause instanceof ReleaseIcnInstallationError
      ? cause
      : installationError(
        "probe",
        `ICN command ${command[1] ?? ""} failed: ${cause instanceof Error ? cause.message : String(cause)}`,
      )),
    Effect.timeoutFail({
      duration: "15 seconds",
      onTimeout: () => installationError(
        "probe",
        `ICN command ${command[1] ?? ""} timed out`,
      ),
    }),
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

/** The inference artifact is one complete installation layout with every backend of its host. */
export const isCompleteArtifact = (
  directory: string,
  host: HostId,
): Effect.Effect<boolean, never, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const path = yield* Path.Path
    const files = yield* Effect.all(
      inferenceRequiredPaths(hostById(host)).map((required) =>
        isNonEmptyFile(path.join(directory, required))
      )
    )
    return files.every((present) => present)
  })

const ensureArtifact = (
  baseUrl: string,
  version: string,
  artifact: ReleaseArtifact,
  root: string,
  host: HostId,
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
    if (!(yield* isCompleteArtifact(destination, host))) {
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
          report: (event) => reporter.report({ _tag: "Artifact", event }),
        }),
      }).pipe(
        Effect.provide(NodeArchiveExtractor),
        Effect.mapError((cause) => installationError("acquire", cause.message))
      )
    }
    if (!(yield* isCompleteArtifact(destination, host))) {
      return yield* installationError(
        "verify",
        `${artifact.id} did not produce a complete installation`
      )
    }
    return destination
  })

const readIdentity = (
  root: string,
  artifact: ReleaseArtifact
): Effect.Effect<
  IcnBinaryIdentity,
  ReleaseIcnInstallationError,
  CommandExecutor.CommandExecutor | Path.Path
> =>
  Effect.gen(function* () {
    const path = yield* Path.Path
    const output = yield* run(
      [path.join(root, "bin", executableName()), "version", "--json"],
      installationLoaderEnvironment(path.join(root, "runtime"))
    )
    const value = yield* Schema.decodeUnknown(Schema.parseJson(IcnBinaryIdentity))(
      output,
    ).pipe(
      Effect.mapError(() =>
        installationError("verify", "ICN identity is malformed")
      )
    )
    if (Option.getOrUndefined(artifact.nativeBuild) !== value.native_build) {
      return yield* installationError(
        "verify",
        "ICN identity differs from the release manifest"
      )
    }
    return value
  })

const readDeclaration = (
  declarationPath: string
): Effect.Effect<Option.Option<IcnInstallationDeclaration>, never, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    return yield* fs.readFileString(declarationPath).pipe(
      Effect.flatMap(
        Schema.decodeUnknown(Schema.parseJson(IcnInstallationDeclaration))
      ),
      Effect.option,
    )
  })

/** Writes the installation declaration that binds the installed layout to its engine build. */
const declareInstallation = (
  root: string,
  native: IcnBinaryIdentity
): Effect.Effect<
  ReleaseIcnInstallation,
  ReleaseIcnInstallationError,
  FileSystem.FileSystem | Path.Path
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const declarationPath = path.join(root, "installation.json")
    const installation = {
      binaryPath: path.join(root, "bin", executableName()),
      declarationPath,
      environment: installationLoaderEnvironment(path.join(root, "runtime")),
    }
    const existing = yield* readDeclaration(declarationPath)
    if (Option.isSome(existing) && existing.value.nativeBuild === native.native_build) {
      return installation
    }
    const serialized = yield* Schema.encode(
      Schema.parseJson(IcnInstallationDeclaration)
    )({ schemaVersion: 1, nativeBuild: native.native_build }).pipe(
      Effect.mapError(() =>
        installationError("declare", "unable to encode ICN installation")
      )
    )
    const staged = path.join(root, `.installation-${process.pid}.json`)
    yield* fs.writeFileString(staged, `${serialized}\n`, { mode: 0o600 }).pipe(
      Effect.zipRight(fs.rename(staged, declarationPath)),
      Effect.mapError(() =>
        installationError("declare", "unable to write ICN installation")
      ),
    )
    return installation
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
    const artifact = yield* selectArtifact(
      release.manifest,
      "icn-base",
      host,
    ).pipe(
      Effect.mapError((cause) => installationError("acquire", cause.message))
    )
    const artifactRoot = path.join(releaseRoot, "artifacts", version, host)
    const prepare = Effect.suspend(() =>
      Effect.gen(function* () {
        const root = yield* ensureArtifact(baseUrl, version, artifact, artifactRoot, host)
        const native = yield* readIdentity(root, artifact)
        return yield* declareInstallation(root, native)
      })
    )
    return yield* prepare.pipe(
      Effect.catchAll((cause) =>
        cause.stage === "verify"
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
