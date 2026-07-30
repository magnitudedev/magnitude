import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import * as NodeCommandExecutor from "@effect/platform-node/NodeCommandExecutor"
import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import * as NodePath from "@effect/platform-node/NodePath"
import { homedir } from "node:os"
import { Effect, Exit, Layer, Option } from "effect"
import {
  acquireRelease,
  installArtifact,
  selectArtifact,
} from "./acquisition"
import { ArchiveExtractor, NodeArchiveExtractor } from "./archive"
import { ReleaseAcquisitionError } from "./errors"
import { makeLauncherInstallationProgress } from "./launcher-progress"
import { currentHost } from "./targets"

const releaseBaseUrl = () => (
  process.env.MAGNITUDE_RELEASE_BASE_URL ??
  "https://github.com/magnitudedev/magnitude/releases/download"
).replace(/\/+$/, "")

const releaseRoot = () => `${homedir()}/.magnitude/releases`

const smokeCli = (
  executable: string,
  version: string,
): Effect.Effect<void, ReleaseAcquisitionError, CommandExecutor.CommandExecutor> =>
  Command.make(executable, "--version").pipe(
    Command.string,
    Effect.timeout("10 seconds"),
    Effect.mapError(() => new ReleaseAcquisitionError({
      stage: "verify",
      message: "Magnitude CLI identity probe failed",
      transient: false,
    })),
    Effect.flatMap((actual) =>
      actual.trim() === version
        ? Effect.void
        : Effect.fail(new ReleaseAcquisitionError({
          stage: "verify",
          message: `Magnitude CLI version mismatch: ${actual.trim()}`,
          transient: false,
        }))
    ),
  )

const cachedCli = (
  version: string,
): Effect.Effect<
  Option.Option<string>,
  never,
  FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const host = currentHost()
    const pointer = path.join(releaseRoot(), "cli", version, host, "current.txt")
    const digest = yield* fs.readFileString(pointer).pipe(
      Effect.map((value) => value.trim()),
      Effect.orElseSucceed(() => ""),
    )
    if (!/^[a-f0-9]{64}$/.test(digest)) return Option.none()
    const extension = process.platform === "win32" ? ".exe" : ""
    const executable = path.join(
      releaseRoot(),
      "cli",
      version,
      host,
      digest,
      "bin",
      `magnitude-cli${extension}`,
    )
    const valid = yield* fs.exists(executable).pipe(
      Effect.flatMap((exists) =>
        exists
          ? smokeCli(executable, version)
          : Effect.fail(new ReleaseAcquisitionError({
            stage: "verify",
            message: "cached CLI is missing",
            transient: false,
          }))
      ),
      Effect.as(true),
      Effect.orElseSucceed(() => false),
    )
    return valid ? Option.some(executable) : Option.none()
  })

const publishPointer = (
  version: string,
  digest: string,
): Effect.Effect<void, ReleaseAcquisitionError, FileSystem.FileSystem | Path.Path> =>
  Effect.scoped(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const pointer = path.join(
        releaseRoot(),
        "cli",
        version,
        currentHost(),
        "current.txt",
      )
      const parent = path.dirname(pointer)
      yield* fs.makeDirectory(parent, { recursive: true, mode: 0o700 }).pipe(
        Effect.mapError(() => new ReleaseAcquisitionError({
          stage: "install",
          message: "unable to create CLI release pointer directory",
          transient: false,
        })),
      )
      const temporary = yield* fs.makeTempFileScoped({
        directory: parent,
        prefix: ".current-",
      }).pipe(
        Effect.mapError(() => new ReleaseAcquisitionError({
          stage: "install",
          message: "unable to create CLI release pointer",
          transient: false,
        })),
      )
      yield* fs.writeFileString(temporary, `${digest}\n`, { mode: 0o600 }).pipe(
        Effect.mapError(() => new ReleaseAcquisitionError({
          stage: "install",
          message: "unable to write CLI release pointer",
          transient: false,
        })),
      )
      yield* fs.rename(temporary, pointer).pipe(
        Effect.catchAll(() =>
          fs.remove(pointer, { force: true }).pipe(
            Effect.zipRight(fs.rename(temporary, pointer)),
          )
        ),
        Effect.mapError(() => new ReleaseAcquisitionError({
          stage: "install",
          message: "unable to publish CLI release pointer",
          transient: false,
        })),
      )
    }),
  )

const ensureBinaryEffect = (
  version: string,
): Effect.Effect<
  string,
  unknown,
  | FileSystem.FileSystem
  | Path.Path
  | CommandExecutor.CommandExecutor
  | HttpClient.HttpClient
  | ArchiveExtractor
> =>
  Effect.gen(function* () {
    const cached = yield* cachedCli(version)
    if (Option.isSome(cached)) return cached.value
    const path = yield* Path.Path
    const fs = yield* FileSystem.FileSystem
    const host = currentHost()
    const release = yield* acquireRelease(
      releaseBaseUrl(),
      version,
      path.join(releaseRoot(), "manifests", version),
    )
    const artifact = yield* selectArtifact(release.manifest, "cli", host)
    const destination = path.join(
      releaseRoot(),
      "cli",
      version,
      host,
      artifact.sha256,
    )
    const extension = process.platform === "win32" ? ".exe" : ""
    const executable = path.join(destination, "bin", `magnitude-cli${extension}`)
    const existing = yield* fs.exists(destination)
    if (existing) {
      const valid = yield* smokeCli(executable, version).pipe(
        Effect.as(true),
        Effect.orElseSucceed(() => false),
      )
      if (!valid) yield* fs.remove(destination, { recursive: true, force: true })
    }
    if (!(yield* fs.exists(destination))) {
      const progress = makeLauncherInstallationProgress()
      yield* installArtifact(
        releaseBaseUrl(),
        version,
        artifact,
        destination,
        { observer: Option.some(progress.observer) },
      ).pipe(
        Effect.onExit((exit) =>
          Exit.isSuccess(exit) ? progress.succeeded : progress.failed
        ),
      )
    }
    yield* smokeCli(executable, version)
    yield* publishPointer(version, artifact.sha256)
    return executable
  })

const RuntimeLayer = Layer.mergeAll(
  NodeFileSystem.layer,
  NodePath.layer,
  FetchHttpClient.layer,
  NodeArchiveExtractor,
  NodeCommandExecutor.layer.pipe(Layer.provide(NodeFileSystem.layer)),
)

export const ensureBinary = (version: string): Promise<string> =>
  Effect.runPromise(ensureBinaryEffect(version).pipe(Effect.provide(RuntimeLayer)))
