import { randomUUID } from "node:crypto"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import { Context, Data, Effect, Either, Option, Schema, Stream } from "effect"
import { CommandCaptureOptions, runCommand } from "../command-output"
import { FileSystemFailureReason, Sha256Digest, type FileSystemFailureReason as FileSystemFailureReasonType } from "../model-files"
import { normalizeFileSystemFailure, sha256File } from "../model-files/platform"
import type { LlamaDistributionVariantId } from "./identity"
import { validateLlamaBinary, type LlamaBinary, type LlamaBinaryError } from "./binary"

export const LlamaArchiveFormat = Schema.Literal("tar.gz", "tar.xz", "zip")
export type LlamaArchiveFormat = Schema.Schema.Type<typeof LlamaArchiveFormat>
export const LlamaDistributionOperation = Schema.Literal("resolve", "select", "download", "verify", "list-archive", "extract", "publish")
export type LlamaDistributionOperation = Schema.Schema.Type<typeof LlamaDistributionOperation>
export const LlamaDistributionFailureReason = Schema.Union(Schema.Literal("not-found", "variant-required", "incompatible-variant", "http-rejected", "transport", "digest-mismatch", "unsafe-archive", "unsafe-path", "command-failed"), FileSystemFailureReason)
export type LlamaDistributionFailureReason = Schema.Schema.Type<typeof LlamaDistributionFailureReason>

export interface LlamaDistributionVariant {
  readonly id: LlamaDistributionVariantId
  readonly platform: NodeJS.Platform
  readonly architecture: string
  readonly archiveUrl: URL
  readonly sha256: Schema.Schema.Type<typeof Sha256Digest>
  readonly executableRelativePath: string
  readonly archive: LlamaArchiveFormat
}

export interface LlamaDistributionManifest {
  readonly version: 1
  readonly release: string
  readonly variants: readonly LlamaDistributionVariant[]
}
export type LlamaDistributionDiagnostic = Data.TaggedEnum<{
  BinaryRejected: { readonly source: LlamaBinary["source"]; readonly path: string; readonly failure: LlamaBinaryError }
  ManagedMarkerInvalid: { readonly path: string; readonly reason: FileSystemFailureReasonType }
  ManagedBinaryEscapesRoot: { readonly markerPath: string; readonly executable: string }
}>
export const LlamaDistributionDiagnostic = Data.taggedEnum<LlamaDistributionDiagnostic>()
export interface LlamaDistributionStatus {
  readonly configured: Option.Option<LlamaBinary>
  readonly managed: Option.Option<LlamaBinary>
  readonly path: Option.Option<LlamaBinary>
  readonly diagnostics: readonly LlamaDistributionDiagnostic[]
}

export class LlamaDistributionError extends Data.TaggedError("LlamaDistributionError")<{
  readonly operation: LlamaDistributionOperation
  readonly reason: LlamaDistributionFailureReason
  readonly variant: Option.Option<LlamaDistributionVariantId>
  readonly status: Option.Option<number>
  readonly path: Option.Option<string>
}> {}

const distributionError = (
  operation: LlamaDistributionOperation,
  reason: LlamaDistributionFailureReason,
  variant: Option.Option<LlamaDistributionVariantId> = Option.none(),
  status: Option.Option<number> = Option.none(),
  failurePath: Option.Option<string> = Option.none(),
): LlamaDistributionError => new LlamaDistributionError({
  operation,
  reason,
  variant,
  status,
  path: failurePath,
})

export interface LlamaDistributionApi {
  readonly status: Effect.Effect<LlamaDistributionStatus>
  readonly resolve: Effect.Effect<LlamaBinary, LlamaDistributionError>
  readonly install: (
    variant: Option.Option<LlamaDistributionVariantId>,
  ) => Effect.Effect<LlamaBinary, LlamaDistributionError | LlamaBinaryError>
}
export class LlamaDistribution extends Context.Tag("@magnitudedev/local-inference/LlamaDistribution")<LlamaDistribution, LlamaDistributionApi>() {}
export interface LlamaDistributionOptions { readonly configuredExecutable: Option.Option<string>; readonly managedRoot: string; readonly manifest: LlamaDistributionManifest; readonly platform: NodeJS.Platform; readonly nativeArchitecture: string; readonly searchPath: readonly string[] }

const ManagedMarker = Schema.Struct({ version: Schema.Literal(1), release: Schema.String, variant: Schema.String, executable: Schema.String })
const ManagedMarkerJson = Schema.parseJson(ManagedMarker, { space: 2 })

type ManagedMarkerState = Data.TaggedEnum<{
  Missing: Record<never, never>
  Invalid: { readonly reason: FileSystemFailureReasonType }
  Present: { readonly executable: string }
}>
const ManagedMarkerState = Data.taggedEnum<ManagedMarkerState>()

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

export const makeLlamaDistribution = (options: LlamaDistributionOptions): Effect.Effect<LlamaDistributionApi, never, FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | CommandExecutor.CommandExecutor> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const http = yield* HttpClient.HttpClient
  const commands = yield* CommandExecutor.CommandExecutor
  const root = path.resolve(options.managedRoot)
  const markerPath = path.join(root, "current.json")
  const provideBinary = (executable: string, source: LlamaBinary["source"]) => validateLlamaBinary({ executable, source }).pipe(
    Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(Path.Path, path), Effect.provideService(CommandExecutor.CommandExecutor, commands),
  )
  const readMarker: Effect.Effect<ManagedMarkerState> = Effect.gen(function* () {
    const read = yield* Effect.either(fs.readFileString(markerPath))
    if (Either.isLeft(read)) {
      const reason = normalizeFileSystemFailure(read.left)
      if (reason === "not-found") return ManagedMarkerState.Missing()
      return ManagedMarkerState.Invalid({ reason })
    }
    const decoded = yield* Effect.either(Schema.decode(ManagedMarkerJson)(read.right))
    if (Either.isLeft(decoded)) return ManagedMarkerState.Invalid({ reason: "invalid-data" })
    const executable = path.resolve(root, decoded.right.executable)
    return containedPath(path, root, executable)
      ? ManagedMarkerState.Present({ executable })
      : ManagedMarkerState.Invalid({ reason: "invalid-data" })
  })
  const executableOnPath = Effect.gen(function* () {
    const name = options.platform === "win32" ? "llama-server.exe" : "llama-server"
    for (const directory of options.searchPath) {
      const candidate = path.join(directory, name)
      const info = yield* fs.stat(candidate).pipe(Effect.option)
      if (Option.isSome(info) && info.value.type === "File") return Option.some(candidate)
    }
    return Option.none<string>()
  })
  const status = Effect.gen(function* () {
    const diagnostics: LlamaDistributionDiagnostic[] = []
    const check = (candidate: Option.Option<string>, source: LlamaBinary["source"]): Effect.Effect<Option.Option<LlamaBinary>> => Option.match(candidate, {
      onNone: () => Effect.succeed(Option.none()),
      onSome: (executable) => provideBinary(executable, source).pipe(Effect.match({
        onFailure: (failure) => {
          diagnostics.push(LlamaDistributionDiagnostic.BinaryRejected({ source, path: executable, failure }))
          return Option.none()
        },
        onSuccess: Option.some,
      })),
    })
    const marker = yield* readMarker
    if (marker._tag === "Invalid") diagnostics.push(LlamaDistributionDiagnostic.ManagedMarkerInvalid({ path: markerPath, reason: marker.reason }))
    const configured = yield* check(options.configuredExecutable, "configured")
    const managedCandidate = Option.liftPredicate(marker, (state): state is Extract<ManagedMarkerState, { readonly _tag: "Present" }> => state._tag === "Present").pipe(Option.map(({ executable }) => executable))
    const managed = yield* check(managedCandidate, "managed").pipe(Effect.map((candidate) => Option.filter(candidate, (binary) => {
      const contained = containedPath(path, root, binary.executable)
      if (!contained) diagnostics.push(LlamaDistributionDiagnostic.ManagedBinaryEscapesRoot({ markerPath, executable: binary.executable }))
      return contained
    })))
    const discovered = yield* executableOnPath
    const pathBinary = yield* check(discovered, "path")
    return { configured, managed, path: pathBinary, diagnostics }
  })
  const install = (requested: Option.Option<LlamaDistributionVariantId>) => Effect.gen(function* () {
    const compatible = options.manifest.variants.filter((variant) => variant.platform === options.platform && variant.architecture === options.nativeArchitecture)
    let chosen: Option.Option<LlamaDistributionVariant>
    let selectionFailure: LlamaDistributionFailureReason

    if (Option.isSome(requested)) {
      chosen = Option.fromNullable(compatible.find(({ id }) => id === requested.value))
      selectionFailure = "incompatible-variant"
    } else {
      chosen = compatible.length === 1 ? Option.fromIterable(compatible) : Option.none()
      selectionFailure = "variant-required"
    }

    if (Option.isNone(chosen)) {
      return yield* distributionError("select", selectionFailure, requested)
    }

    const selected = chosen.value
    yield* fs.makeDirectory(root, { recursive: true }).pipe(Effect.mapError((error) => distributionError("publish", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(root))))
    return yield* Effect.scoped(Effect.gen(function* () {
      const temporary = yield* fs.makeTempDirectoryScoped({ directory: root, prefix: ".install-" }).pipe(Effect.mapError((error) => distributionError("publish", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(root))))
      const archive = path.join(temporary, `archive.${selected.archive.replace(".", "-")}`)
      const extracted = path.join(temporary, "extracted")
      yield* fs.makeDirectory(extracted, { recursive: true }).pipe(Effect.mapError((error) => distributionError("extract", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(extracted))))
      const response = yield* http.execute(HttpClientRequest.get(selected.archiveUrl.toString())).pipe(Effect.mapError(() => distributionError("download", "transport", Option.some(selected.id))))
      if (response.status < 200 || response.status >= 300) return yield* distributionError("download", "http-rejected", Option.some(selected.id), Option.some(response.status))
      yield* response.stream.pipe(Stream.mapError(() => distributionError("download", "transport", Option.some(selected.id))), Stream.run(fs.sink(archive, { flag: "wx" })), Effect.mapError((error) => error._tag === "LlamaDistributionError" ? error : distributionError("download", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(archive))))
      const digest = yield* sha256File(archive).pipe(Effect.provideService(FileSystem.FileSystem, fs), Effect.mapError((error) => distributionError("verify", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(archive))))
      if (digest !== selected.sha256) return yield* distributionError("verify", "digest-mismatch", Option.some(selected.id), Option.none(), Option.some(archive))
      const archiveCommand = archiveCommands(selected.archive)
      const listing = yield* runCommand(
        archiveCommand.executable,
        archiveCommand.listArguments(archive),
        CommandCaptureOptions.Default,
      ).pipe(Effect.provideService(CommandExecutor.CommandExecutor, commands), Effect.mapError(() => distributionError("list-archive", "command-failed", Option.some(selected.id))))
      if (listing.exitCode !== 0) return yield* distributionError("list-archive", "command-failed", Option.some(selected.id))
      for (const entry of listing.stdout.split("\n").filter((value) => value.length > 0)) {
        const normalized = entry.replaceAll("\\", "/")
        const windowsAbsolute = /^[A-Za-z]:\//.test(normalized)
        if (normalized.startsWith("/") || windowsAbsolute || normalized.split("/").includes("..")) return yield* distributionError("list-archive", "unsafe-archive", Option.some(selected.id), Option.none(), Option.some(entry))
      }
      const extraction = yield* runCommand(
        archiveCommand.executable,
        archiveCommand.extractArguments(archive, extracted),
        CommandCaptureOptions.Default,
      ).pipe(Effect.provideService(CommandExecutor.CommandExecutor, commands), Effect.mapError(() => distributionError("extract", "command-failed", Option.some(selected.id))))
      if (extraction.exitCode !== 0) return yield* distributionError("extract", "command-failed", Option.some(selected.id))
      const executable = path.resolve(extracted, selected.executableRelativePath)
      const relativeExecutable = path.relative(extracted, executable)
      if (relativeExecutable === ".." || relativeExecutable.startsWith(`..${path.sep}`) || path.isAbsolute(relativeExecutable)) return yield* distributionError("extract", "unsafe-path", Option.some(selected.id), Option.none(), Option.some(selected.executableRelativePath))
      const realExecutable = yield* fs.realPath(executable).pipe(Effect.mapError((error) => distributionError("extract", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(executable))))
      if (!containedPath(path, extracted, realExecutable)) return yield* distributionError("extract", "unsafe-path", Option.some(selected.id), Option.none(), Option.some(selected.executableRelativePath))
      yield* fs.chmod(realExecutable, 0o755).pipe(Effect.mapError((error) => distributionError("extract", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(realExecutable))))
      yield* provideBinary(realExecutable, "managed")
      const destination = path.join(root, `${options.manifest.release}-${selected.id}-${randomUUID()}`)
      yield* fs.rename(extracted, destination).pipe(Effect.mapError((error) => distributionError("publish", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(destination))))
      const relative = path.join(path.basename(destination), selected.executableRelativePath)
      const marker = yield* Schema.encode(ManagedMarkerJson)({ version: 1, release: options.manifest.release, variant: selected.id, executable: relative }).pipe(Effect.mapError(() => distributionError("publish", "invalid-data", Option.some(selected.id), Option.none(), Option.some(markerPath))))
      const markerTemp = `${markerPath}.${randomUUID()}.tmp`
      yield* fs.writeFileString(markerTemp, marker, { mode: 0o600 }).pipe(Effect.mapError((error) => distributionError("publish", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(markerTemp))))
      yield* fs.rename(markerTemp, markerPath).pipe(Effect.mapError((error) => distributionError("publish", normalizeFileSystemFailure(error), Option.some(selected.id), Option.none(), Option.some(markerPath))))
      return yield* provideBinary(path.join(destination, selected.executableRelativePath), "managed")
    }))
  })
  const resolvedBinary = (current: LlamaDistributionStatus): Option.Option<LlamaBinary> => Option.orElse(
    current.configured,
    () => Option.orElse(current.managed, () => current.path),
  )
  const resolve = status.pipe(
    Effect.flatMap((current) => Option.match(resolvedBinary(current), {
      onNone: () => Effect.fail(distributionError("resolve", "not-found")),
      onSome: Effect.succeed,
    })),
  )

  return {
    status,
    resolve,
    install,
  }
})
