import { randomUUID } from "node:crypto"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import { Context, Data, Effect, Either, Option, PubSub, Ref, Schema, Scope, Stream } from "effect"
import {
  LlamaCppInstallationDiscovery,
  LlamaBuildNumber,
  makeLlamaCppInstallation,
  validateLlamaCppExecutable,
  LlamaCppInstallationSchema,
  type LlamaCppInstallation,
} from "./installation"
import {
  installManagedLlamaCpp,
  type LlamaDistributionManifest,
  type LlamaManagedInstallInternalStage,
} from "./distribution"
import { LlamaCppInstallationId, LlamaDistributionVariantId, LlamaInstallOperationId } from "./identity"

export const MINIMUM_LLAMACPP_BUILD = LlamaBuildNumber.make(8868)
export const RECOMMENDED_LLAMACPP_BUILD = LlamaBuildNumber.make(10011)

export const LlamaCppInstallationDiscoveryDiagnostic = Schema.Union(
  Schema.TaggedStruct("ConfiguredPathUnavailable", { requestedPath: Schema.String, reason: Schema.String }),
  Schema.TaggedStruct("ManagedMarkerInvalid", { markerPath: Schema.String, reason: Schema.String }),
  Schema.TaggedStruct("InstallationRejected", { requestedPath: Schema.String, reason: Schema.String }),
)
export type LlamaCppInstallationDiscoveryDiagnostic = Schema.Schema.Type<typeof LlamaCppInstallationDiscoveryDiagnostic>

export const LlamaManagedInstallOperation = Schema.Union(
  Schema.TaggedStruct("Idle", {}),
  Schema.TaggedStruct("Running", {
    operationId: LlamaInstallOperationId,
    stage: Schema.Literal("Resolving", "Downloading", "VerifyingArchive", "Extracting", "VerifyingInstallation", "Publishing", "Applying"),
    bytesDownloaded: Schema.OptionFromSelf(Schema.NonNegativeInt),
    bytesTotal: Schema.OptionFromSelf(Schema.NonNegativeInt),
  }),
  Schema.TaggedStruct("Failed", { operationId: LlamaInstallOperationId, message: Schema.String }),
)
export type LlamaManagedInstallOperation = Schema.Schema.Type<typeof LlamaManagedInstallOperation>

export const LlamaManagedInstall = Schema.Struct({
  availability: Schema.Union(
    Schema.TaggedStruct("Available", { variantId: LlamaDistributionVariantId, build: LlamaBuildNumber }),
    Schema.TaggedStruct("UnsupportedPlatform", { reason: Schema.String }),
  ),
  operation: LlamaManagedInstallOperation,
})
export type LlamaManagedInstall = Schema.Schema.Type<typeof LlamaManagedInstall>

export const LlamaCppInstallationRegistrySnapshotSchema = Schema.Struct({
  minimumBuild: LlamaBuildNumber,
  recommendedBuild: LlamaBuildNumber,
  installations: Schema.Array(LlamaCppInstallationSchema),
  selectedInstallationId: Schema.OptionFromSelf(LlamaCppInstallationId),
  managedInstall: LlamaManagedInstall,
  diagnostics: Schema.Array(LlamaCppInstallationDiscoveryDiagnostic),
})
export type LlamaCppInstallationRegistrySnapshot = typeof LlamaCppInstallationRegistrySnapshotSchema.Type

export class LlamaCppInstallationUnavailable extends Data.TaggedError("LlamaCppInstallationUnavailable")<{
  readonly reason: "missing" | "outdated"
}> {}

export class LlamaCppInstallationRefreshError extends Data.TaggedError("LlamaCppInstallationRefreshError")<{
  readonly message: string
}> {}

export class LlamaInstallStartError extends Data.TaggedError("LlamaInstallStartError")<{
  readonly reason: "unsupported-platform"
}> {}

export interface LlamaCppInstallationRegistryApi {
  readonly snapshot: Effect.Effect<LlamaCppInstallationRegistrySnapshot>
  readonly changes: Stream.Stream<void>
  readonly refresh: Effect.Effect<void, LlamaCppInstallationRefreshError>
  readonly selected: Effect.Effect<LlamaCppInstallation, LlamaCppInstallationUnavailable>
  readonly installManaged: Effect.Effect<LlamaInstallOperationId, LlamaInstallStartError>
}

export class LlamaCppInstallationRegistry extends Context.Tag("@magnitudedev/local-inference/LlamaCppInstallationRegistry")<
  LlamaCppInstallationRegistry,
  LlamaCppInstallationRegistryApi
>() {}

export interface LlamaCppInstallationRegistryOptions {
  readonly configuredServerExecutable: Option.Option<string>
  readonly managedRoot: string
  readonly searchPath: readonly string[]
  readonly managedVariant: Option.Option<LlamaDistributionVariantId>
  readonly manifest: LlamaDistributionManifest
  readonly platform: NodeJS.Platform
  readonly nativeArchitecture: string
}

const ManagedMarker = Schema.Struct({
  version: Schema.Literal(1),
  release: Schema.String,
  variant: Schema.String,
  executables: Schema.Struct({
    server: Schema.String,
    fitParams: Schema.String,
  }),
})
const ManagedMarkerJson = Schema.parseJson(ManagedMarker)

interface DiscoveredInstallation {
  readonly server: string
  readonly fitParams: string
  readonly versionKey: string
  readonly discovery: LlamaCppInstallationDiscovery
  readonly ownership: LlamaCppInstallation["ownership"]
}

interface DiscoveryGroup {
  readonly server: string
  readonly fitParams: string
  readonly versionKey: string
  readonly discoveries: LlamaCppInstallationDiscovery[]
  readonly ownership: LlamaCppInstallation["ownership"]
}

const discoveryPriority = (discovery: LlamaCppInstallationDiscovery): number => {
  if (discovery._tag === "Configured") return 0
  if (discovery._tag === "Managed") return 1
  return 2 + discovery.priority
}

const selectedInstallation = (
  installations: readonly LlamaCppInstallation[],
  minimumBuild: LlamaBuildNumber,
): Option.Option<LlamaCppInstallation> => Option.fromNullable(
  installations
    .filter((installation) => installation.build >= minimumBuild)
    .toSorted((left, right) =>
      Math.min(...left.discoveries.map(discoveryPriority)) - Math.min(...right.discoveries.map(discoveryPriority))
      || left.executables.server.path.localeCompare(right.executables.server.path),
    )[0],
)

const equivalentDiscovery = (left: LlamaCppInstallationDiscovery, right: LlamaCppInstallationDiscovery): boolean => {
  if (left._tag !== right._tag) return false
  if (left._tag === "Configured" && right._tag === "Configured") return left.requestedPath === right.requestedPath
  if (left._tag === "Managed" && right._tag === "Managed") return left.markerPath === right.markerPath && left.release === right.release
  return left._tag === "Path" && right._tag === "Path" && left.requestedPath === right.requestedPath && left.priority === right.priority
}

const equivalentInstallation = (left: LlamaCppInstallation, right: LlamaCppInstallation): boolean =>
  left.id === right.id
  && left.executables.server.path === right.executables.server.path
  && left.executables.server.fingerprint === right.executables.server.fingerprint
  && left.executables.fitParams.path === right.executables.fitParams.path
  && left.executables.fitParams.fingerprint === right.executables.fitParams.fingerprint
  && left.build === right.build
  && Option.getOrNull(left.commit) === Option.getOrNull(right.commit)
  && left.ownership === right.ownership
  && left.discoveries.length === right.discoveries.length
  && left.discoveries.every((discovery, index) => {
    const candidate = right.discoveries[index]
    return candidate ? equivalentDiscovery(discovery, candidate) : false
  })

const equivalentInstall = Schema.equivalence(LlamaManagedInstall)
const equivalentSnapshot = (left: LlamaCppInstallationRegistrySnapshot, right: LlamaCppInstallationRegistrySnapshot): boolean =>
  left.minimumBuild === right.minimumBuild
  && left.recommendedBuild === right.recommendedBuild
  && Option.getOrNull(left.selectedInstallationId) === Option.getOrNull(right.selectedInstallationId)
  && left.installations.length === right.installations.length
  && left.installations.every((installation, index) => right.installations[index] ? equivalentInstallation(installation, right.installations[index]!) : false)
  && equivalentInstall(left.managedInstall, right.managedInstall)
  && JSON.stringify(left.diagnostics) === JSON.stringify(right.diagnostics)

const message = (cause: unknown): string => cause instanceof Error
  ? cause.message.slice(0, 512)
  : String(cause).slice(0, 512)

export const makeLlamaCppInstallationRegistry = (
  options: LlamaCppInstallationRegistryOptions,
): Effect.Effect<
  LlamaCppInstallationRegistryApi,
  never,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | CommandExecutor.CommandExecutor | Scope.Scope
> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const http = yield* HttpClient.HttpClient
  const executor = yield* CommandExecutor.CommandExecutor
  const scope = yield* Scope.Scope
  const managedRoot = path.resolve(options.managedRoot)
  const markerPath = path.join(managedRoot, "current.json")
  const availability: LlamaManagedInstall["availability"] = Option.match(options.managedVariant, {
    onNone: () => ({ _tag: "UnsupportedPlatform", reason: "No managed llama.cpp build is available for this platform and architecture." }),
    onSome: (variantId) => ({ _tag: "Available", variantId, build: RECOMMENDED_LLAMACPP_BUILD }),
  })
  const initial: LlamaCppInstallationRegistrySnapshot = {
    minimumBuild: MINIMUM_LLAMACPP_BUILD,
    recommendedBuild: RECOMMENDED_LLAMACPP_BUILD,
    installations: [],
    selectedInstallationId: Option.none(),
    managedInstall: { availability, operation: { _tag: "Idle" } },
    diagnostics: [],
  }
  const state = yield* Ref.make(initial)
  const changes = yield* PubSub.unbounded<void>()
  const lock = yield* Effect.makeSemaphore(1)
  const installLock = yield* Effect.makeSemaphore(1)
  const inspectionCache = new Map<string, { readonly versionKey: string; readonly installation: LlamaCppInstallation }>()
  const publish = Effect.asVoid(PubSub.publish(changes, undefined))

  const setState = (next: LlamaCppInstallationRegistrySnapshot) => Ref.modify(state, (previous) =>
    equivalentSnapshot(previous, next) ? [false, previous] as const : [true, next] as const,
  ).pipe(Effect.flatMap((changed) => changed ? publish : Effect.void))

  const updateInstall = (operation: LlamaManagedInstallOperation) => Ref.modify(state, (previous) => {
    const next = { ...previous, managedInstall: { ...previous.managedInstall, operation } }
    return equivalentSnapshot(previous, next) ? [false, previous] as const : [true, next] as const
  }).pipe(Effect.flatMap((changed) => changed ? publish : Effect.void))

  const inspect = Effect.gen(function* () {
    const diagnostics: LlamaCppInstallationDiscoveryDiagnostic[] = []
    const discovered: DiscoveredInstallation[] = []
    const serverName = process.platform === "win32" ? "llama-server.exe" : "llama-server"
    const fitParamsName = process.platform === "win32" ? "llama-fit-params.exe" : "llama-fit-params"
    const contained = (candidate: string) => {
      const relative = path.relative(managedRoot, candidate)
      return relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative)
    }
    const file = (requestedPath: string) => Effect.gen(function* () {
      const absolute = path.resolve(requestedPath)
      const stat = yield* Effect.either(fs.stat(absolute))
      if (Either.isLeft(stat) || stat.right.type !== "File") return Option.none<{
        readonly canonical: string
        readonly versionKey: string
      }>()
      const canonical = yield* Effect.either(fs.realPath(absolute))
      if (Either.isLeft(canonical)) return Option.none()
      const modifiedAt = Option.match(stat.right.mtime, {
        onNone: () => "unknown",
        onSome: (value) => String(value.getTime()),
      })
      return Option.some({ canonical: canonical.right, versionKey: `${stat.right.size}:${modifiedAt}` })
    })

    const add = (
      requestedServerPath: string,
      requestedFitParamsPath: Option.Option<string>,
      discovery: LlamaCppInstallationDiscovery,
      ownership: LlamaCppInstallation["ownership"],
      required: boolean,
    ) => Effect.gen(function* () {
      const server = yield* file(requestedServerPath)
      if (Option.isNone(server)) {
        if (required) diagnostics.push(discovery._tag === "Configured"
          ? { _tag: "ConfiguredPathUnavailable", requestedPath: requestedServerPath, reason: "The configured llama-server executable is unavailable." }
          : { _tag: "InstallationRejected", requestedPath: requestedServerPath, reason: "The managed llama-server executable is unavailable." })
        return
      }
      const fitCandidates = Option.match(requestedFitParamsPath, {
        onSome: (fitParamsPath) => [fitParamsPath],
        onNone: () => [...new Set([
          path.join(path.dirname(path.resolve(requestedServerPath)), fitParamsName),
          path.join(path.dirname(server.value.canonical), fitParamsName),
        ])],
      })
      const fitResults = yield* Effect.forEach(fitCandidates, file)
      const fitParams = fitResults.find(Option.isSome)
      if (!fitParams || Option.isNone(fitParams)) {
        if (required) diagnostics.push({
          _tag: "InstallationRejected",
          requestedPath: requestedServerPath,
          reason: `The matching ${fitParamsName} executable is unavailable.`,
        })
        return
      }
      if (discovery._tag === "Managed") {
        if (!contained(server.value.canonical) || !contained(fitParams.value.canonical)) {
          diagnostics.push({ _tag: "ManagedMarkerInvalid", markerPath, reason: "A managed executable resolves outside its installation root." })
          return
        }
      }
      discovered.push({
        server: server.value.canonical,
        fitParams: fitParams.value.canonical,
        versionKey: `${server.value.versionKey}\0${fitParams.value.versionKey}`,
        discovery,
        ownership,
      })
    })

    yield* Option.match(options.configuredServerExecutable, {
      onNone: () => Effect.void,
      onSome: (requestedPath) => add(requestedPath, Option.none(), { _tag: "Configured", requestedPath }, "user", true),
    })

    const markerRead = yield* Effect.either(fs.readFileString(markerPath))
    if (Either.isRight(markerRead)) {
      const marker = yield* Effect.either(Schema.decode(ManagedMarkerJson)(markerRead.right))
      if (Either.isLeft(marker)) {
        diagnostics.push({ _tag: "ManagedMarkerInvalid", markerPath, reason: "The managed installation marker is invalid." })
      } else {
        const server = path.resolve(managedRoot, marker.right.executables.server)
        const fitParams = path.resolve(managedRoot, marker.right.executables.fitParams)
        if (!contained(server) || !contained(fitParams)) {
          diagnostics.push({ _tag: "ManagedMarkerInvalid", markerPath, reason: "A managed executable escapes its installation root." })
        } else {
          yield* add(server, Option.some(fitParams), { _tag: "Managed", markerPath, release: marker.right.release }, "magnitude", true)
        }
      }
    }

    yield* Effect.forEach(
      options.searchPath.map((directory, priority) => ({ directory, priority })),
      ({ directory, priority }) => {
        const requestedPath = path.join(directory, serverName)
        return add(requestedPath, Option.none(), { _tag: "Path", requestedPath, priority }, "user", false)
      },
      { discard: true },
    )

    const groups = new Map<string, DiscoveryGroup>()
    for (const item of discovered) {
      const key = `${item.server}\0${item.fitParams}`
      const current = groups.get(key)
      if (current) {
        current.discoveries.push(item.discovery)
        if (item.ownership === "magnitude") groups.set(key, { ...current, ownership: "magnitude" })
      } else {
        groups.set(key, { server: item.server, fitParams: item.fitParams, versionKey: item.versionKey, discoveries: [item.discovery], ownership: item.ownership })
      }
    }

    const inspected = yield* Effect.forEach(groups.values(), (group) => {
      const discoveries = group.discoveries.toSorted((left, right) => discoveryPriority(left) - discoveryPriority(right))
      const key = `${group.server}\0${group.fitParams}`
      const cached = inspectionCache.get(key)
      if (cached?.versionKey === group.versionKey) {
        return Effect.succeed(Option.some({ ...cached.installation, ownership: group.ownership, discoveries }))
      }
      return Effect.all({
        server: validateLlamaCppExecutable(group.server),
        fitParams: validateLlamaCppExecutable(group.fitParams),
      }, { concurrency: 2 }).pipe(
        Effect.flatMap((executables) => makeLlamaCppInstallation({ ...executables, ownership: group.ownership, discoveries })),
        Effect.provideService(FileSystem.FileSystem, fs),
        Effect.provideService(Path.Path, path),
        Effect.provideService(CommandExecutor.CommandExecutor, executor),
        Effect.match({
          onFailure: (failure) => {
            inspectionCache.delete(key)
            diagnostics.push({ _tag: "InstallationRejected", requestedPath: group.server, reason: message(failure) })
            return Option.none<LlamaCppInstallation>()
          },
          onSuccess: (installation) => {
            inspectionCache.set(key, { versionKey: group.versionKey, installation })
            return Option.some(installation)
          },
        }),
      )
    }, { concurrency: 4 })
    const installations = inspected.flatMap(Option.toArray).toSorted((left, right) =>
      Math.min(...left.discoveries.map(discoveryPriority)) - Math.min(...right.discoveries.map(discoveryPriority))
      || left.executables.server.path.localeCompare(right.executables.server.path),
    )
    const selected = selectedInstallation(installations, MINIMUM_LLAMACPP_BUILD)
    const previous = yield* Ref.get(state)
    return {
      minimumBuild: MINIMUM_LLAMACPP_BUILD,
      recommendedBuild: RECOMMENDED_LLAMACPP_BUILD,
      installations,
      selectedInstallationId: Option.map(selected, (installation) => installation.id),
      managedInstall: previous.managedInstall,
      diagnostics,
    } satisfies LlamaCppInstallationRegistrySnapshot
  })

  const refresh = lock.withPermits(1)(inspect.pipe(
    Effect.flatMap(setState),
    Effect.mapError((cause) => new LlamaCppInstallationRefreshError({ message: message(cause) })),
    Effect.catchAllDefect((cause) => Effect.fail(new LlamaCppInstallationRefreshError({ message: message(cause) }))),
  ))
  yield* refresh.pipe(Effect.catchAll((cause) => Effect.logWarning("Unable to refresh llama.cpp installations").pipe(
    Effect.annotateLogs({ cause: cause.message }),
  )))

  const selected = Ref.get(state).pipe(Effect.flatMap((snapshot) => Option.match(snapshot.selectedInstallationId, {
    onNone: () => Effect.fail(new LlamaCppInstallationUnavailable({ reason: snapshot.installations.length === 0 ? "missing" : "outdated" })),
    onSome: (id) => Option.match(Option.fromNullable(snapshot.installations.find((installation) => installation.id === id)), {
      onNone: () => Effect.fail(new LlamaCppInstallationUnavailable({ reason: "missing" })),
      onSome: Effect.succeed,
    }),
  })))

  const installManaged = installLock.withPermits(1)(Effect.gen(function* () {
    const snapshot = yield* Ref.get(state)
    if (snapshot.managedInstall.operation._tag === "Running") return snapshot.managedInstall.operation.operationId
    if (snapshot.managedInstall.availability._tag === "UnsupportedPlatform") {
      return yield* new LlamaInstallStartError({ reason: "unsupported-platform" })
    }
    const operationId = LlamaInstallOperationId.make(randomUUID())
    const running = (stage: LlamaManagedInstallInternalStage | "Applying"): LlamaManagedInstallOperation => ({
      _tag: "Running",
      operationId,
      stage,
      bytesDownloaded: Option.none(),
      bytesTotal: Option.none(),
    })
    yield* updateInstall(running("Resolving"))
    const run = installManagedLlamaCpp({
      managedRoot: options.managedRoot,
      manifest: options.manifest,
      platform: options.platform,
      nativeArchitecture: options.nativeArchitecture,
      expectedBuild: RECOMMENDED_LLAMACPP_BUILD,
      variant: Option.some(snapshot.managedInstall.availability.variantId),
      onStage: (stage) => updateInstall(running(stage)),
    }).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.provideService(Path.Path, path),
      Effect.provideService(HttpClient.HttpClient, http),
      Effect.provideService(CommandExecutor.CommandExecutor, executor),
      Effect.asVoid,
      Effect.zipRight(updateInstall(running("Applying"))),
      Effect.zipRight(refresh),
      Effect.zipRight(updateInstall({ _tag: "Idle" })),
      Effect.catchAll((cause) => updateInstall({ _tag: "Failed", operationId, message: message(cause) })),
    )
    yield* Effect.forkIn(run, scope)
    return operationId
  }))

  return {
    snapshot: Ref.get(state),
    changes: Stream.fromPubSub(changes),
    refresh,
    selected,
    installManaged,
  }
})
