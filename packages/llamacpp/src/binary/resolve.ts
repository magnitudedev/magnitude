import { Context, Effect, Layer } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import * as HttpClient from "@effect/platform/HttpClient"
import { BunFileSystem, BunPath, BunCommandExecutor } from "@effect/platform-bun"
import { FetchHttpClient } from "@effect/platform"
import {
  LlamaCppBinaryNotFound,
  LlamaCppBinaryVersionTooOld,
  LlamaCppBinaryDownloadFailed,
  LlamaCppUnsupportedPlatform,
  LlamaCppBinaryValidationFailed,
} from "../errors"
import {
  MINIMUM_LLMACPP_VERSION,
  RECOMMENDED_LLMACPP_VERSION,
  meetsMinimum,
} from "../version"
import {
  cachedBinaryPath,
  versionMarkerPath,
} from "../paths"
import { detectPlatform, type GpuPreference } from "../platform"
import { downloadBinary } from "./download"
import { validateBinary } from "./validate"
import { BinarySource, ResolvedBinary, BinaryStatus } from "./types"

// ── Service Tag ──

export interface LlamaCppBinaryApi {
  readonly resolve: (options?: {
    readonly gpuPreference?: GpuPreference
  }) => Effect.Effect<
    ResolvedBinary,
    LlamaCppBinaryNotFound | LlamaCppBinaryVersionTooOld | LlamaCppBinaryDownloadFailed
      | LlamaCppUnsupportedPlatform | LlamaCppBinaryValidationFailed
  >

  readonly getStatus: () => Effect.Effect<BinaryStatus>

  readonly install: (options?: {
    readonly gpuPreference?: GpuPreference
  }) => Effect.Effect<
    BinaryStatus,
    LlamaCppBinaryDownloadFailed | LlamaCppUnsupportedPlatform | LlamaCppBinaryValidationFailed
  >
}

export class LlamaCppBinary extends Context.Tag("LlamaCppBinary")<
  LlamaCppBinary,
  LlamaCppBinaryApi
>() {}

// ── Platform layer (baked in) ──

const PlatformLayer = Layer.mergeAll(
  BunPath.layer,
  BunCommandExecutor.layer,
  FetchHttpClient.layer,
).pipe(Layer.provideMerge(BunFileSystem.layer))

// ── Factory ──

export interface LlamaCppBinaryDeps {
  readonly configuredPath?: string
}

export function makeLlamaCppBinary(deps: LlamaCppBinaryDeps = {}): LlamaCppBinaryApi {
  const resolve: LlamaCppBinaryApi["resolve"] = (options) =>
    resolveBinary(deps, options?.gpuPreference ?? "auto").pipe(
      Effect.provide(PlatformLayer),
      Effect.catchAll((err) =>
        isKnownBinaryError(err)
          ? Effect.fail(err)
          : Effect.fail(new LlamaCppBinaryNotFound({ searched: [] })),
      ),
    )

  const getStatus: LlamaCppBinaryApi["getStatus"] = () =>
    getBinaryStatus(deps).pipe(Effect.provide(PlatformLayer))

  const install: LlamaCppBinaryApi["install"] = (options) =>
    installRecommended(options?.gpuPreference ?? "auto").pipe(
      Effect.flatMap(() => getBinaryStatus(deps)),
      Effect.provide(PlatformLayer),
      Effect.catchAll((err) =>
        isKnownInstallError(err)
          ? Effect.fail(err)
          : Effect.fail(new LlamaCppBinaryDownloadFailed({ url: "", reason: String(err) })),
      ),
    )

  return { resolve, getStatus, install }
}

function isKnownBinaryError(err: unknown): boolean {
  return err instanceof LlamaCppBinaryNotFound
    || err instanceof LlamaCppBinaryVersionTooOld
    || err instanceof LlamaCppBinaryDownloadFailed
    || err instanceof LlamaCppUnsupportedPlatform
    || err instanceof LlamaCppBinaryValidationFailed
}

function isKnownInstallError(err: unknown): boolean {
  return err instanceof LlamaCppBinaryDownloadFailed
    || err instanceof LlamaCppUnsupportedPlatform
    || err instanceof LlamaCppBinaryValidationFailed
}

// ── Resolution ──

function resolveBinary(
  deps: LlamaCppBinaryDeps,
  gpuPreference: GpuPreference,
): Effect.Effect<
  ResolvedBinary,
  | LlamaCppBinaryNotFound
  | LlamaCppBinaryVersionTooOld
  | LlamaCppBinaryDownloadFailed
  | LlamaCppUnsupportedPlatform
  | LlamaCppBinaryValidationFailed,
  FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient
> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem

    // 1. Env override
    const envPath = process.env.LLAMA_SERVER_PATH?.trim()
    if (envPath) {
      const result = yield* tryValidate(envPath, "env")
      if (result) return result
    }

    // 2. Configured path
    if (deps.configuredPath?.trim()) {
      const result = yield* tryValidate(deps.configuredPath.trim(), "config")
      if (result) return result
    }

    // 3. Magnitude-managed cache (version marker)
    const marker = versionMarkerPath()
    const markerExists = yield* fs.exists(marker).pipe(Effect.catchAll(() => Effect.succeed(false)))
    if (markerExists) {
      const markerContent = yield* fs.readFileString(marker).pipe(Effect.catchAll(() => Effect.succeed("")))
      const markerBuild = Number(markerContent.trim())
      if (Number.isFinite(markerBuild) && markerBuild > 0) {
        const binPath = cachedBinaryPath(markerBuild)
        const binExists = yield* fs.exists(binPath).pipe(Effect.catchAll(() => Effect.succeed(false)))
        if (binExists) {
          const result = yield* tryValidate(binPath, "cache")
          if (result) return result
        }
      }
    }

    // 4. PATH detection
    const whichResult = yield* Command.string(Command.make("which", "llama-server")).pipe(
      Effect.catchAll(() => Effect.succeed("")),
    )
    const whichPath = whichResult.trim()
    if (whichPath) {
      const result = yield* tryValidate(whichPath, "path")
      if (result) return result
    }

    // 5. Common locations
    const commonLocations = ["/usr/local/bin/llama-server", "/opt/homebrew/bin/llama-server"]
    for (const loc of commonLocations) {
      const exists = yield* fs.exists(loc).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (exists) {
        const result = yield* tryValidate(loc, "common-location")
        if (result) return result
      }
    }

    // 6. Download recommended
    return yield* installRecommended(gpuPreference)
  })
}

function tryValidate(
  path: string,
  source: BinarySource,
): Effect.Effect<
  ResolvedBinary | null,
  LlamaCppBinaryVersionTooOld | LlamaCppBinaryValidationFailed,
  FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const pathSvc = yield* Path.Path
    const exists = yield* fs.exists(path).pipe(Effect.catchAll(() => Effect.succeed(false)))
    if (!exists) return null

    const buildNumber = yield* validateBinary(path)
    if (!meetsMinimum(buildNumber)) {
      return yield* new LlamaCppBinaryVersionTooOld({
        path,
        actual: buildNumber,
        minimum: MINIMUM_LLMACPP_VERSION,
      })
    }

    return { path, directory: pathSvc.dirname(path), buildNumber, source }
  })
}

// ── Install ──

function installRecommended(
  gpuPreference: GpuPreference,
): Effect.Effect<
  ResolvedBinary,
  LlamaCppBinaryDownloadFailed | LlamaCppUnsupportedPlatform | LlamaCppBinaryValidationFailed,
  FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient
> {
  return Effect.gen(function* () {
    const platformInfo = yield* detectPlatform(gpuPreference)
    return yield* downloadBinary(RECOMMENDED_LLMACPP_VERSION, platformInfo.asset).pipe(
      Effect.map((result) => ({
        path: result.path,
        directory: result.directory,
        buildNumber: result.buildNumber,
        source: "download" as BinarySource,
      })),
    )
  })
}

// ── Status ──

function getBinaryStatus(
  deps: LlamaCppBinaryDeps,
): Effect.Effect<
  BinaryStatus,
  never,
  FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient
> {
  return Effect.gen(function* () {
    const result = yield* resolveBinary(deps, "auto").pipe(
      Effect.map((binary): { _tag: "found"; binary: ResolvedBinary } => ({ _tag: "found", binary })),
      Effect.catchAll(() => Effect.succeed({ _tag: "not_found" as const })),
    )

    if (result._tag === "found") {
      return {
        installed: true,
        buildNumber: result.binary.buildNumber,
        path: result.binary.path,
        source: result.binary.source,
        meetsMinimum: meetsMinimum(result.binary.buildNumber),
        minimumRequired: MINIMUM_LLMACPP_VERSION,
        recommended: RECOMMENDED_LLMACPP_VERSION,
      }
    }

    return {
      installed: false,
      buildNumber: null,
      path: null,
      source: null,
      meetsMinimum: false,
      minimumRequired: MINIMUM_LLMACPP_VERSION,
      recommended: RECOMMENDED_LLMACPP_VERSION,
    }
  })
}
