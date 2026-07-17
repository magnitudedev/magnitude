import { randomUUID } from "node:crypto"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import { Data, Effect, Option, Schema, Stream } from "effect"
import { CommandCaptureOptions, runCommand } from "../command-output"
import { FileSystemFailureReason, Sha256Digest } from "../model-files"
import { normalizeFileSystemFailure, sha256File } from "../model-files/platform"
import type { LlamaDistributionVariantId } from "./identity"
import {
  makeLlamaCppInstallation,
  validateLlamaCppExecutable,
  type LlamaCppExecutableError,
  type LlamaCppInstallation,
  type LlamaBuildNumber,
} from "./installation"

export const LlamaArchiveFormat = Schema.Literal("tar.gz", "tar.xz", "zip")
export type LlamaArchiveFormat = Schema.Schema.Type<typeof LlamaArchiveFormat>
export const LlamaManagedInstallFailureOperation = Schema.Literal("select", "download", "verify", "list-archive", "extract", "publish")
export type LlamaManagedInstallFailureOperation = Schema.Schema.Type<typeof LlamaManagedInstallFailureOperation>
export const LlamaManagedInstallFailureReason = Schema.Union(Schema.Literal("variant-required", "incompatible-variant", "http-rejected", "transport", "digest-mismatch", "unexpected-build", "unsafe-archive", "unsafe-path", "command-failed"), FileSystemFailureReason)
export type LlamaManagedInstallFailureReason = Schema.Schema.Type<typeof LlamaManagedInstallFailureReason>
export const LlamaManagedInstallInternalStage = Schema.Literal("Resolving", "Downloading", "VerifyingArchive", "Extracting", "VerifyingInstallation", "Publishing")
export type LlamaManagedInstallInternalStage = Schema.Schema.Type<typeof LlamaManagedInstallInternalStage>

export interface LlamaDistributionVariant {
  readonly id: LlamaDistributionVariantId
  readonly platform: NodeJS.Platform
  readonly architecture: string
  readonly archiveUrl: URL
  readonly sha256: Schema.Schema.Type<typeof Sha256Digest>
  readonly executables: {
    readonly server: string
    readonly fitParams: string
  }
  readonly archive: LlamaArchiveFormat
}

export interface LlamaDistributionManifest {
  readonly version: 1
  readonly release: string
  readonly variants: readonly LlamaDistributionVariant[]
}

export class LlamaManagedInstallError extends Data.TaggedError("LlamaManagedInstallError")<{
  readonly operation: LlamaManagedInstallFailureOperation
  readonly reason: LlamaManagedInstallFailureReason
  readonly variant: Option.Option<LlamaDistributionVariantId>
  readonly status: Option.Option<number>
  readonly path: Option.Option<string>
}> {}

const installError = (
  operation: LlamaManagedInstallFailureOperation,
  reason: LlamaManagedInstallFailureReason,
  variant: Option.Option<LlamaDistributionVariantId> = Option.none(),
  status: Option.Option<number> = Option.none(),
  failurePath: Option.Option<string> = Option.none(),
): LlamaManagedInstallError => new LlamaManagedInstallError({
  operation,
  reason,
  variant,
  status,
  path: failurePath,
})

export interface InstallManagedLlamaCppOptions {
  readonly managedRoot: string
  readonly manifest: LlamaDistributionManifest
  readonly platform: NodeJS.Platform
  readonly nativeArchitecture: string
  readonly expectedBuild: LlamaBuildNumber
  readonly variant: Option.Option<LlamaDistributionVariantId>
  readonly onStage: (stage: LlamaManagedInstallInternalStage) => Effect.Effect<void>
}

const ManagedMarker = Schema.Struct({
  version: Schema.Literal(1),
  release: Schema.String,
  variant: Schema.String,
  executables: Schema.Struct({ server: Schema.String, fitParams: Schema.String }),
})
const ManagedMarkerJson = Schema.parseJson(ManagedMarker, { space: 2 })

interface ArchiveCommands {
  readonly executable: "tar" | "unzip"
  readonly listArguments: (archive: string) => readonly string[]
  readonly extractArguments: (archive: string, destination: string) => readonly string[]
}

const archiveCommands = (format: LlamaArchiveFormat): ArchiveCommands => {
  if (format === "zip") {
    return {
      executable: "unzip",
      listArguments: (archive) => ["-Z1", archive],
      extractArguments: (archive, destination) => ["-q", archive, "-d", destination],
    }
  }

  if (format === "tar.gz") {
    return {
      executable: "tar",
      listArguments: (archive) => ["-tzf", archive],
      extractArguments: (archive, destination) => ["-xzf", archive, "-C", destination],
    }
  }

  return {
    executable: "tar",
    listArguments: (archive) => ["-tJf", archive],
    extractArguments: (archive, destination) => ["-xJf", archive, "-C", destination],
  }
}

const containedPath = (path: Path.Path, root: string, candidate: string): boolean => {
  const relative = path.relative(root, candidate)
  return relative === "" || (relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
}

export const installManagedLlamaCpp = (
  options: InstallManagedLlamaCppOptions,
): Effect.Effect<
  LlamaCppInstallation,
  LlamaManagedInstallError | LlamaCppExecutableError,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | CommandExecutor.CommandExecutor
> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const http = yield* HttpClient.HttpClient
  const commands = yield* CommandExecutor.CommandExecutor
  const root = path.resolve(options.managedRoot)
  const markerPath = path.join(root, "current.json")
  const provideInstallation = (executables: { readonly server: string; readonly fitParams: string }) => Effect.all({
    server: validateLlamaCppExecutable(executables.server),
    fitParams: validateLlamaCppExecutable(executables.fitParams),
  }, { concurrency: 2 }).pipe(
    Effect.flatMap((validated) => makeLlamaCppInstallation({
      ...validated,
      ownership: "magnitude",
      discoveries: [{ _tag: "Managed", markerPath, release: options.manifest.release }],
    })),
    Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(Path.Path, path), Effect.provideService(CommandExecutor.CommandExecutor, commands),
  )
    yield* options.onStage("Resolving")
    const compatible = options.manifest.variants.filter((variant) => variant.platform === options.platform && variant.architecture === options.nativeArchitecture)
    let chosen: Option.Option<LlamaDistributionVariant>
    let selectionFailure: LlamaManagedInstallFailureReason

    const requested = options.variant
    if (Option.isSome(requested)) {
      chosen = Option.fromNullable(compatible.find(({ id }) => id === requested.value))
      selectionFailure = "incompatible-variant"
    } else {
      chosen = compatible.length === 1 ? Option.fromIterable(compatible) : Option.none()
      selectionFailure = "variant-required"
    }

    if (Option.isNone(chosen)) {
      return yield* installError("select", selectionFailure, options.variant)
    }

    const selected = chosen.value
    yield* fs.makeDirectory(root, { recursive: true }).pipe(Effect.mapError((error) => installError("publish", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(root))))
    return yield* Effect.scoped(Effect.gen(function* () {
      const temporary = yield* fs.makeTempDirectoryScoped({ directory: root, prefix: ".install-" }).pipe(Effect.mapError((error) => installError("publish", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(root))))
      const archive = path.join(temporary, `archive.${selected.archive.replace(".", "-")}`)
      const extracted = path.join(temporary, "extracted")
      yield* fs.makeDirectory(extracted, { recursive: true }).pipe(Effect.mapError((error) => installError("extract", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(extracted))))
      yield* options.onStage("Downloading")
      const response = yield* http.execute(HttpClientRequest.get(selected.archiveUrl.toString())).pipe(Effect.mapError(() => installError("download", "transport", Option.some(selected.id))))
      if (response.status < 200 || response.status >= 300) return yield* installError("download", "http-rejected", Option.some(selected.id), Option.some(response.status))
      yield* response.stream.pipe(Stream.mapError(() => installError("download", "transport", Option.some(selected.id))), Stream.run(fs.sink(archive, { flag: "wx" })), Effect.mapError((error) => error._tag === "LlamaManagedInstallError" ? error : installError("download", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(archive))))
      yield* options.onStage("VerifyingArchive")
      const digest = yield* sha256File(archive).pipe(Effect.provideService(FileSystem.FileSystem, fs), Effect.mapError((error) => installError("verify", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(archive))))
      if (digest !== selected.sha256) return yield* installError("verify", "digest-mismatch", Option.some(selected.id), Option.none(), Option.some(archive))
      const archiveCommand = archiveCommands(selected.archive)
      const listing = yield* runCommand(
        archiveCommand.executable,
        archiveCommand.listArguments(archive),
        CommandCaptureOptions.Default,
      ).pipe(Effect.provideService(CommandExecutor.CommandExecutor, commands), Effect.mapError(() => installError("list-archive", "command-failed", Option.some(selected.id))))
      if (listing.exitCode !== 0) return yield* installError("list-archive", "command-failed", Option.some(selected.id))
      for (const entry of listing.stdout.split("\n").filter((value) => value.length > 0)) {
        const normalized = entry.replaceAll("\\", "/")
        const windowsAbsolute = /^[A-Za-z]:\//.test(normalized)
        if (normalized.startsWith("/") || windowsAbsolute || normalized.split("/").includes("..")) return yield* installError("list-archive", "unsafe-archive", Option.some(selected.id), Option.none(), Option.some(entry))
      }
      yield* options.onStage("Extracting")
      const extraction = yield* runCommand(
        archiveCommand.executable,
        archiveCommand.extractArguments(archive, extracted),
        CommandCaptureOptions.Default,
      ).pipe(Effect.provideService(CommandExecutor.CommandExecutor, commands), Effect.mapError(() => installError("extract", "command-failed", Option.some(selected.id))))
      if (extraction.exitCode !== 0) return yield* installError("extract", "command-failed", Option.some(selected.id))
      const extractedExecutables = {
        server: path.resolve(extracted, selected.executables.server),
        fitParams: path.resolve(extracted, selected.executables.fitParams),
      }
      for (const executable of Object.values(extractedExecutables)) {
        const relativeExecutable = path.relative(extracted, executable)
        if (relativeExecutable === ".." || relativeExecutable.startsWith(`..${path.sep}`) || path.isAbsolute(relativeExecutable)) return yield* installError("extract", "unsafe-path", Option.some(selected.id), Option.none(), Option.some(executable))
        const realExecutable = yield* fs.realPath(executable).pipe(Effect.mapError((error) => installError("extract", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(executable))))
        if (!containedPath(path, extracted, realExecutable)) return yield* installError("extract", "unsafe-path", Option.some(selected.id), Option.none(), Option.some(executable))
        yield* fs.chmod(realExecutable, 0o755).pipe(Effect.mapError((error) => installError("extract", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(realExecutable))))
      }
      yield* options.onStage("VerifyingInstallation")
      const verified = yield* provideInstallation(extractedExecutables)
      if (verified.build !== options.expectedBuild) {
        return yield* installError("verify", "unexpected-build", Option.some(selected.id), Option.none(), Option.some(extractedExecutables.server))
      }
      yield* options.onStage("Publishing")
      const destination = path.join(root, `${options.manifest.release}-${selected.id}-${randomUUID()}`)
      yield* fs.rename(extracted, destination).pipe(Effect.mapError((error) => installError("publish", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(destination))))
      const relativeExecutables = {
        server: path.join(path.basename(destination), selected.executables.server),
        fitParams: path.join(path.basename(destination), selected.executables.fitParams),
      }
      const marker = yield* Schema.encode(ManagedMarkerJson)({ version: 1, release: options.manifest.release, variant: selected.id, executables: relativeExecutables }).pipe(Effect.mapError(() => installError("publish", "invalid-data", Option.some(selected.id), Option.none(), Option.some(markerPath))))
      const markerTemp = `${markerPath}.${randomUUID()}.tmp`
      yield* fs.writeFileString(markerTemp, marker, { mode: 0o600 }).pipe(Effect.mapError((error) => installError("publish", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(markerTemp))))
      yield* fs.rename(markerTemp, markerPath).pipe(Effect.mapError((error) => installError("publish", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(markerPath))))
      return yield* provideInstallation({
        server: path.join(destination, selected.executables.server),
        fitParams: path.join(destination, selected.executables.fitParams),
      })
    }))
})
