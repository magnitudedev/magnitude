import { createHash, randomUUID } from "node:crypto"
import { Context, Effect, Layer, Schema, Stream } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import * as Path from "@effect/platform/Path"
import type {
  DistributionInstallEvent,
  DistributionState,
  LlamaCppDistributionConfig,
  LlamaCppReleaseAsset,
  ResolvedDistribution,
} from "./contracts"
import { DistributionInspectionError, DistributionInstallError } from "./errors"
import { DEFAULT_LLAMACPP_RELEASE } from "./release-manifest"
import { MINIMUM_LLMACPP_VERSION } from "./version"
import { validateBinary } from "./binary/validate"

type DistributionPlatform =
  | FileSystem.FileSystem
  | Path.Path
  | CommandExecutor.CommandExecutor
  | HttpClient.HttpClient

export interface LlamaCppDistributionApi {
  readonly inspect: Effect.Effect<DistributionState, DistributionInspectionError>
  readonly install: Stream.Stream<DistributionInstallEvent, DistributionInstallError>
}

export class LlamaCppDistribution extends Context.Tag("LlamaCppDistribution")<
  LlamaCppDistribution,
  LlamaCppDistributionApi
>() {}

interface ManagedMarker {
  readonly build: number
  readonly executable: string
}

const ManagedMarkerSchema = Schema.Struct({
  build: Schema.Number,
  executable: Schema.String,
})

const ManagedMarkerJson = Schema.parseJson(ManagedMarkerSchema)

const markerPath = (config: LlamaCppDistributionConfig, path: Path.Path): string =>
  path.join(config.managedRoot, "current.json")

const inspectExecutable = (
  executablePath: string,
  source: ResolvedDistribution["source"],
): Effect.Effect<DistributionState, DistributionInspectionError, FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const exists = yield* fs.exists(executablePath).pipe(
      Effect.mapError((cause) => new DistributionInspectionError({ operation: "inspect", reason: "Could not inspect executable", cause })),
    )
    if (!exists) {
      return source === "configured"
        ? { _tag: "Invalid", reason: `Configured llama-server does not exist: ${executablePath}` }
        : { _tag: "Missing" }
    }
    const build = yield* validateBinary(executablePath).pipe(
      Effect.mapError((cause) => new DistributionInspectionError({ operation: "inspect", reason: cause.reason, cause })),
    )
    if (build < MINIMUM_LLMACPP_VERSION) {
      return {
        _tag: "Invalid",
        reason: `llama-server build ${build} is older than required build ${MINIMUM_LLMACPP_VERSION}`,
      }
    }
    return {
      _tag: "Ready",
      distribution: {
        executablePath,
        directory: path.dirname(executablePath),
        build,
        source,
      },
    }
  })

const inspectManaged = (
  config: LlamaCppDistributionConfig,
): Effect.Effect<DistributionState, DistributionInspectionError, FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const file = markerPath(config, path)
    const exists = yield* fs.exists(file).pipe(
      Effect.mapError((cause) => new DistributionInspectionError({ operation: "inspect", reason: "Could not inspect managed distribution", cause })),
    )
    if (!exists) return { _tag: "Missing" }
    const marker = yield* fs.readFileString(file).pipe(
      Effect.flatMap(Schema.decodeUnknown(ManagedMarkerJson)),
      Effect.mapError((cause) => cause instanceof DistributionInspectionError
        ? cause
        : new DistributionInspectionError({ operation: "inspect", reason: "Managed distribution marker is invalid", cause })),
    )
    const state = yield* inspectExecutable(marker.executable, "managed")
    if (state._tag === "Ready" && state.distribution.build !== marker.build) {
      return { _tag: "Invalid", reason: "Managed distribution marker does not match executable build" }
    }
    return state
  })

const inspectDistribution = (
  config: LlamaCppDistributionConfig,
): Effect.Effect<DistributionState, DistributionInspectionError, FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor> =>
  config.configuredExecutable?.trim()
    ? inspectExecutable(config.configuredExecutable.trim(), "configured")
    : inspectManaged(config)

const releaseAsset = (config: LlamaCppDistributionConfig): LlamaCppReleaseAsset | null => {
  const platform = process.platform
  const architecture = process.arch
  if ((platform !== "darwin" && platform !== "linux") || (architecture !== "arm64" && architecture !== "x64")) {
    return null
  }
  const manifest = config.release ?? DEFAULT_LLAMACPP_RELEASE
  const requested = platform === "darwin"
    ? (architecture === "arm64" ? "metal" : "cpu")
    : config.accelerator === "vulkan" ? "vulkan" : "cpu"
  return manifest.assets.find((asset) =>
    asset.platform === platform
    && asset.architecture === architecture
    && asset.accelerator === requested
  ) ?? null
}

const locateExecutable = (
  root: string,
): Effect.Effect<string | null, never, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const entries = yield* fs.readDirectory(root).pipe(Effect.orElseSucceed(() => []))
    for (const entry of entries) {
      const candidate = path.join(root, entry)
      const info = yield* fs.stat(candidate).pipe(Effect.orElseSucceed(() => null))
      if (info?.type === "File" && entry === "llama-server") return candidate
      if (info?.type === "Directory") {
        const nested = yield* locateExecutable(candidate)
        if (nested) return nested
      }
    }
    return null
  })

const installDistribution = (
  config: LlamaCppDistributionConfig,
): Stream.Stream<DistributionInstallEvent, DistributionInstallError, DistributionPlatform> =>
  Stream.unwrapScoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const client = yield* HttpClient.HttpClient
    const manifest = config.release ?? DEFAULT_LLAMACPP_RELEASE
    const asset = releaseAsset(config)
    if (!asset) {
      return Stream.fail(new DistributionInstallError({
        operation: "install",
        code: "unsupported_platform",
        stage: "resolving",
        reason: `Unsupported platform ${process.platform}/${process.arch}`,
      }))
    }

    const temporaryRoot = path.join(config.managedRoot, `.install-${manifest.build}-${randomUUID()}`)
    const archivePath = path.join(temporaryRoot, asset.fileName)
    const extractRoot = path.join(temporaryRoot, "extracted")
    yield* fs.makeDirectory(extractRoot, { recursive: true }).pipe(
      Effect.mapError((cause) => new DistributionInstallError({ operation: "install", code: "storage_failed", stage: "resolving", reason: "Could not create installation directory", cause })),
    )
    yield* Effect.addFinalizer(() => fs.remove(temporaryRoot, { recursive: true, force: true }).pipe(Effect.ignore))

    const file = yield* fs.open(archivePath, { flag: "w" }).pipe(
      Effect.mapError((cause) => new DistributionInstallError({ operation: "install", code: "storage_failed", stage: "downloading", reason: "Could not open release archive", cause })),
    )
    const hash = createHash("sha256")
    let completedBytes = 0
    const download = Stream.unwrap(
      client.execute(HttpClientRequest.get(asset.url)).pipe(
        Effect.flatMap(HttpClientResponse.filterStatusOk),
        Effect.map((response) => response.stream),
        Effect.mapError((cause) => new DistributionInstallError({ operation: "install", code: "download_failed", stage: "downloading", reason: "Release download failed", cause })),
      ),
    ).pipe(
      Stream.mapError((cause) => cause instanceof DistributionInstallError
        ? cause
        : new DistributionInstallError({ operation: "install", code: "download_failed", stage: "downloading", reason: "Release body stream failed", cause })),
      Stream.mapEffect((chunk) => file.write(chunk).pipe(
        Effect.tap(() => Effect.sync(() => {
          hash.update(chunk)
          completedBytes += chunk.byteLength
        })),
        Effect.map((): DistributionInstallEvent => ({
          _tag: "Downloading",
          completedBytes,
          totalBytes: asset.sizeBytes,
        })),
        Effect.mapError((cause) => new DistributionInstallError({ operation: "install", code: "storage_failed", stage: "downloading", reason: "Could not write release archive", cause })),
      )),
    )

    const verifyArchive = Effect.gen(function* () {
      if (completedBytes !== asset.sizeBytes) {
        return yield* new DistributionInstallError({
          operation: "install",
          code: "integrity_failed",
          stage: "verifying",
          reason: `Release size mismatch: expected ${asset.sizeBytes}, received ${completedBytes}`,
        })
      }
      const digest = hash.digest("hex")
      if (digest !== asset.sha256) {
        return yield* new DistributionInstallError({
          operation: "install",
          code: "integrity_failed",
          stage: "verifying",
          reason: `Release SHA-256 mismatch: expected ${asset.sha256}, received ${digest}`,
        })
      }
    })

    const extract = Command.exitCode(Command.make("tar", "-xzf", archivePath, "-C", extractRoot)).pipe(
      Effect.flatMap((code) => code === 0
        ? Effect.void
        : Effect.fail(new DistributionInstallError({ operation: "install", code: "integrity_failed", stage: "extracting", reason: `tar exited with status ${code}` }))),
      Effect.mapError((cause) => cause instanceof DistributionInstallError
        ? cause
        : new DistributionInstallError({ operation: "install", code: "integrity_failed", stage: "extracting", reason: "Could not extract release archive", cause })),
    )

    const publish = Effect.gen(function* () {
      const executable = yield* locateExecutable(extractRoot)
      if (!executable) {
        return yield* new DistributionInstallError({ operation: "install", code: "integrity_failed", stage: "verifying", reason: "Release does not contain llama-server" })
      }
      yield* fs.chmod(executable, 0o755).pipe(Effect.ignore)
      const build = yield* validateBinary(executable).pipe(
        Effect.mapError((cause) => new DistributionInstallError({ operation: "install", code: "integrity_failed", stage: "verifying", reason: cause.reason, cause })),
      )
      if (build !== manifest.build) {
        return yield* new DistributionInstallError({ operation: "install", code: "integrity_failed", stage: "verifying", reason: `Expected build ${manifest.build}, found ${build}` })
      }

      const sourceDirectory = path.dirname(executable)
      const targetDirectory = path.join(config.managedRoot, `llama-${manifest.tag}`)
      const backupDirectory = path.join(config.managedRoot, `.previous-${randomUUID()}`)
      const targetExists = yield* fs.exists(targetDirectory).pipe(Effect.orElseSucceed(() => false))
      if (targetExists) {
        yield* fs.rename(targetDirectory, backupDirectory).pipe(
          Effect.mapError((cause) => new DistributionInstallError({ operation: "install", code: "storage_failed", stage: "publishing", reason: "Could not stage previous distribution", cause })),
        )
      }
      const published = yield* fs.rename(sourceDirectory, targetDirectory).pipe(Effect.exit)
      if (published._tag === "Failure") {
        if (targetExists) yield* fs.rename(backupDirectory, targetDirectory).pipe(Effect.ignore)
        return yield* new DistributionInstallError({ operation: "install", code: "storage_failed", stage: "publishing", reason: "Could not publish distribution", cause: published.cause })
      }
      const finalExecutable = path.join(targetDirectory, "llama-server")
      const distribution: ResolvedDistribution = {
        executablePath: finalExecutable,
        directory: targetDirectory,
        build,
        source: "managed",
      }
      const marker = yield* Schema.encode(ManagedMarkerJson)({
        build,
        executable: finalExecutable,
      } satisfies ManagedMarker).pipe(
        Effect.mapError((cause) => new DistributionInstallError({ operation: "install", code: "storage_failed", stage: "publishing", reason: "Could not encode distribution marker", cause })),
      )
      const temporaryMarker = path.join(config.managedRoot, `.current-${randomUUID()}.json`)
      const markerPublished = yield* fs.writeFileString(temporaryMarker, marker).pipe(
        Effect.zipRight(fs.rename(temporaryMarker, markerPath(config, path))),
        Effect.exit,
      )
      if (markerPublished._tag === "Failure") {
        yield* fs.remove(temporaryMarker, { force: true }).pipe(Effect.ignore)
        yield* fs.remove(targetDirectory, { recursive: true, force: true }).pipe(Effect.ignore)
        if (targetExists) yield* fs.rename(backupDirectory, targetDirectory).pipe(Effect.ignore)
        return yield* new DistributionInstallError({
          operation: "install",
          code: "storage_failed",
          stage: "publishing",
          reason: "Could not publish distribution marker",
          cause: markerPublished.cause,
        })
      }
      if (targetExists) yield* fs.remove(backupDirectory, { recursive: true, force: true }).pipe(Effect.ignore)
      return distribution
    })

    const installEvent = (event: DistributionInstallEvent): DistributionInstallEvent => event
    const stage = (event: DistributionInstallEvent, effect: Effect.Effect<void, DistributionInstallError, DistributionPlatform>) =>
      Stream.make(event).pipe(Stream.concat(Stream.fromEffect(effect).pipe(Stream.drain)))

    return Stream.make(installEvent({ _tag: "Resolving" })).pipe(
      Stream.concat(download),
      Stream.concat(stage({ _tag: "Verifying" }, verifyArchive)),
      Stream.concat(stage({ _tag: "Extracting" }, extract)),
      Stream.concat(Stream.make(installEvent({ _tag: "Verifying" }))),
      Stream.concat(Stream.make(installEvent({ _tag: "Publishing" }))),
      Stream.concat(Stream.fromEffect(Effect.uninterruptible(publish)).pipe(
        Stream.map((distribution): DistributionInstallEvent => ({ _tag: "Ready", distribution })),
      )),
    )
  }))

export const LlamaCppDistributionLive = (
  config: LlamaCppDistributionConfig,
): Layer.Layer<LlamaCppDistribution, never, DistributionPlatform> =>
  Layer.effect(
    LlamaCppDistribution,
    Effect.gen(function* () {
      const platform = yield* Effect.context<DistributionPlatform>()
      const installLock = yield* Effect.makeSemaphore(1)
      const install = Stream.acquireRelease(
        installLock.take(1),
        () => installLock.release(1),
      ).pipe(
        Stream.flatMap(() => installDistribution(config)),
        Stream.provideContext(platform),
      )
      return LlamaCppDistribution.of({
        inspect: inspectDistribution(config).pipe(Effect.provide(platform)),
        install,
      })
    }),
  )
