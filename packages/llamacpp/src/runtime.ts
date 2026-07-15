import { randomBytes, randomUUID } from "node:crypto"
import { Context, Effect, Exit, Layer, Option, Ref, Scope, Secret, Stream } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import type {
  EnsureServingRequest,
  EnsureManagedServingRequest,
  LlamaCppConnection,
  LlamaCppRuntimeConfig,
  LlamaCppRuntimeSnapshot,
  LlamaCppServerObservation,
  ResolvedDistribution,
  ResolvedModelArtifact,
  ServingTarget,
  VerifiedServedModelMetadata,
} from "./contracts"
import { makeLlamaCppEndpointClient, type LlamaCppEndpointProps, type LlamaCppServedModel } from "./client/index"
import { LlamaCppDistribution } from "./distribution"
import { LlamaCppModelStore } from "./model-store"
import { LlamaCppRuntimeError } from "./errors"
import { findFreePort } from "./inference/ports"

type RuntimeRequirements =
  | LlamaCppDistribution
  | LlamaCppModelStore
  | FileSystem.FileSystem
  | Path.Path
  | CommandExecutor.CommandExecutor
  | HttpClient.HttpClient

interface ManagedServerHandle {
  readonly serverId: string
  readonly process: CommandExecutor.Process
  readonly connection: LlamaCppConnection
  readonly presetPath: string
  readonly request: EnsureManagedServingRequest
}

interface ManagedServerController {
  readonly current: Effect.Effect<ManagedServerHandle | null>
  readonly ensure: (
    distribution: ResolvedDistribution,
    artifact: ResolvedModelArtifact,
    request: EnsureManagedServingRequest,
  ) => Effect.Effect<ManagedServerHandle, LlamaCppRuntimeError>
  readonly stop: Effect.Effect<void>
}

export interface LlamaCppRuntimeApi {
  readonly inspect: Effect.Effect<LlamaCppRuntimeSnapshot, LlamaCppRuntimeError>
  readonly ensureServing: (request: EnsureServingRequest) => Effect.Effect<ServingTarget, LlamaCppRuntimeError>
  readonly stopManaged: Effect.Effect<void>
}

export class LlamaCppRuntime extends Context.Tag("LlamaCppRuntime")<LlamaCppRuntime, LlamaCppRuntimeApi>() {}

const runtimeError = (
  operation: LlamaCppRuntimeError["operation"],
  code: LlamaCppRuntimeError["code"],
  reason: string,
  cause?: unknown,
): LlamaCppRuntimeError => new LlamaCppRuntimeError({
  operation,
  code,
  reason,
  ...(cause === undefined ? {} : { cause }),
})

type ServedModelMetadataKey = keyof NonNullable<LlamaCppServedModel["meta"]>

const metadataString = (model: LlamaCppServedModel, key: ServedModelMetadataKey): string | null => {
  const value = model.meta?.[key]
  return typeof value === "string" ? value : null
}

const metadataNumber = (model: LlamaCppServedModel, key: ServedModelMetadataKey): number | null => {
  const value = model.meta?.[key]
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

const verifiedMetadata = (model: LlamaCppServedModel, props: LlamaCppEndpointProps): VerifiedServedModelMetadata => ({
  architecture: metadataString(model, "general.architecture"),
  quantization: metadataString(model, "ftype") ?? props.model_ftype ?? null,
  sizeBytes: metadataNumber(model, "size"),
})

const reportedModelPath = (
  model: LlamaCppServedModel,
  props: LlamaCppEndpointProps,
): string | null => {
  const value = model.path ?? props.model_path
  if (value === undefined || value.trim().toLowerCase() === "none") return null
  return value
}

const verifyManagedIdentity = (
  model: LlamaCppServedModel,
  props: LlamaCppEndpointProps,
  artifact: ResolvedModelArtifact,
): Effect.Effect<void, LlamaCppRuntimeError> => Effect.gen(function* () {
  const servedPath = reportedModelPath(model, props)
  if (servedPath !== null && servedPath !== artifact.primaryPath) {
    return yield* runtimeError(
      "ensure_serving",
      "identity_mismatch",
      `Managed llama-server reported model path ${servedPath}; expected ${artifact.primaryPath}`,
    )
  }

  const metadata = verifiedMetadata(model, props)
  if (metadata.sizeBytes !== null && metadata.sizeBytes !== artifact.sizeBytes) {
    return yield* runtimeError(
      "ensure_serving",
      "identity_mismatch",
      `Managed llama-server reported ${metadata.sizeBytes} model bytes; expected ${artifact.sizeBytes}`,
    )
  }
})

const observeConnection = (
  connection: LlamaCppConnection,
  ownership: "managed" | "external",
  serverId: string,
  managedProviderModelId?: string,
): Effect.Effect<LlamaCppServerObservation, never, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const client = makeLlamaCppEndpointClient(connection)
    const health = yield* client.health
    if (health._tag !== "Ready") {
      return {
        serverId,
        ownership,
        health: health._tag === "Loading" ? "loading" : "unhealthy",
        models: [],
        build: null,
      }
    }
    const probed = yield* Effect.all([client.models, client.props], { concurrency: 2 }).pipe(Effect.option)
    if (Option.isNone(probed)) {
      return { serverId, ownership, health: "unhealthy", models: [], build: null }
    }
    const [reportedModels, props] = probed.value
    const models = managedProviderModelId === undefined
      ? reportedModels
      : reportedModels.filter((model) => model.id === managedProviderModelId)
    const sharedContext = props.default_generation_settings?.n_ctx
    const sharedMetadata = models.length === 1 ? props : {}
    return {
      serverId,
      ownership,
      health: "ready",
      models: models.map((model) => {
        const metadata = verifiedMetadata(model, sharedMetadata)
        return {
          providerModelId: model.id,
          modelPath: reportedModelPath(model, sharedMetadata),
          displayName: metadataString(model, "general.name")
            ?? metadataString(model, "general_name")
            ?? null,
          contextTokens: model.meta?.n_ctx !== undefined && model.meta.n_ctx > 0
            ? model.meta.n_ctx
            : sharedContext !== undefined && sharedContext > 0
              ? sharedContext
              : null,
          quantization: metadata.quantization,
          sizeBytes: metadata.sizeBytes,
        }
      }),
      build: props.build_info ?? null,
    }
  })

const normalizedConnectionUrl = (connection: LlamaCppConnection): string =>
  connection.baseUrl.trim().replace(/\/+$/, "")

const validatePresetValue = (label: string, value: string): Effect.Effect<string, LlamaCppRuntimeError> =>
  /[\r\n\[\]]/.test(value)
    ? Effect.fail(runtimeError("ensure_serving", "server_start_failed", `${label} contains unsupported preset characters`))
    : Effect.succeed(value)

const makePreset = (
  artifact: ResolvedModelArtifact,
  request: EnsureManagedServingRequest,
): Effect.Effect<string, LlamaCppRuntimeError> =>
  Effect.gen(function* () {
    const alias = yield* validatePresetValue("Provider model ID", request.providerModelId)
    const modelPath = yield* validatePresetValue("Model path", artifact.primaryPath)
    const projectorPath = artifact.projectorPath
      ? yield* validatePresetValue("Projector path", artifact.projectorPath)
      : null
    const lines = [
      "[*]",
      `LLAMA_ARG_N_GPU_LAYERS = ${request.fitPlan.gpuLayers}`,
      `LLAMA_ARG_CTX_SIZE = ${request.contextTokens}`,
      `LLAMA_ARG_N_PARALLEL = ${request.fitPlan.parallelSlots}`,
      "LLAMA_ARG_FLASH_ATTN = auto",
      "LLAMA_ARG_CONT_BATCHING = true",
      "LLAMA_ARG_JINJA = true",
      "",
      `[${alias}]`,
      `LLAMA_ARG_MODEL = ${modelPath}`,
      `LLAMA_ARG_ALIAS = ${alias}`,
      ...(projectorPath ? [`LLAMA_ARG_MMPROJ = ${projectorPath}`] : []),
      "__PRESET_LOAD_ON_STARTUP = true",
    ]
    return lines.join("\n")
  })

const stopProcess = (
  handle: ManagedServerHandle,
  fs: FileSystem.FileSystem,
): Effect.Effect<void> =>
  handle.process.kill("SIGTERM").pipe(
    Effect.ignore,
    Effect.zipRight(handle.process.exitCode.pipe(
      Effect.timeout("5 seconds"),
      Effect.catchAll(() => handle.process.kill("SIGKILL").pipe(Effect.ignore)),
    )),
    Effect.zipRight(fs.remove(handle.presetPath, { force: true }).pipe(Effect.ignore)),
    Effect.asVoid,
  )

const waitForReady = (
  connection: LlamaCppConnection,
  stderr: Ref.Ref<string>,
  process: CommandExecutor.Process,
): Effect.Effect<void, LlamaCppRuntimeError, HttpClient.HttpClient> => {
  const client = makeLlamaCppEndpointClient(connection)
  const check = Effect.gen(function* () {
    const health = yield* client.health
    if (health._tag === "Ready") return true
    yield* Effect.sleep("500 millis")
    return false
  })
  const ready = check.pipe(
    Effect.repeat({ until: (ready) => ready }),
    Effect.timeout("2 minutes"),
    Effect.catchAll((cause) => Ref.get(stderr).pipe(
      Effect.flatMap((output) => Effect.fail(runtimeError(
        "ensure_serving",
        output.toLowerCase().includes("out of memory") ? "server_start_failed" : "server_timeout",
        output.trim() || "llama-server did not become ready within two minutes",
        cause,
      ))),
    )),
    Effect.asVoid,
  )
  const exited = process.exitCode.pipe(
    Effect.mapError((cause) => runtimeError(
      "ensure_serving",
      "server_start_failed",
      "Could not observe llama-server process state",
      cause,
    )),
    Effect.flatMap((code) => Ref.get(stderr).pipe(
      Effect.flatMap((output) => Effect.fail(runtimeError(
        "ensure_serving",
        "server_start_failed",
        output.trim() || `llama-server exited with status ${code} before becoming ready`,
      ))),
    )),
  )
  return Effect.raceFirst(ready, exited)
}

const makeManagedController = (
  config: LlamaCppRuntimeConfig,
): Effect.Effect<ManagedServerController, never, FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient | Scope.Scope> =>
  Effect.gen(function* () {
    const context = yield* Effect.context<FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient | Scope.Scope>()
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const state = yield* Ref.make<ManagedServerHandle | null>(null)
    const lock = yield* Effect.makeSemaphore(1)

    const stop = lock.withPermits(1)(Effect.gen(function* () {
      const current = yield* Ref.getAndSet(state, null)
      if (current) yield* stopProcess(current, fs)
    }))

    const ensureRaw = (distribution: ResolvedDistribution, artifact: ResolvedModelArtifact, request: EnsureManagedServingRequest) =>
      lock.withPermits(1)(Effect.gen(function* () {
        const existing = yield* Ref.get(state)
        if (existing && existing.request.modelId === request.modelId
          && existing.request.providerModelId === request.providerModelId
          && existing.request.contextTokens === request.contextTokens
          && existing.request.fitPlan.parallelSlots === request.fitPlan.parallelSlots
          && existing.request.fitPlan.gpuLayers === request.fitPlan.gpuLayers
          && existing.request.fitPlan.splitMode === request.fitPlan.splitMode) {
          const health = yield* makeLlamaCppEndpointClient(existing.connection).health
          if (health._tag === "Ready") return existing
        }
        if (existing) {
          yield* Ref.set(state, null)
          yield* stopProcess(existing, fs)
        }

        yield* fs.makeDirectory(config.runtimeRoot, { recursive: true }).pipe(
          Effect.mapError((cause) => runtimeError("ensure_serving", "server_start_failed", "Could not create runtime directory", cause)),
        )
        const presetPath = path.join(config.runtimeRoot, `router-${randomUUID()}.ini`)
        const preset = yield* makePreset(artifact, request)
        yield* fs.writeFileString(presetPath, preset).pipe(
          Effect.mapError((cause) => runtimeError("ensure_serving", "server_start_failed", "Could not write router preset", cause)),
        )
        const pending = yield* Ref.make<ManagedServerHandle | null>(null)
        const cleanupPending = Ref.get(pending).pipe(
          Effect.flatMap((handle) => handle
            ? stopProcess(handle, fs)
            : fs.remove(presetPath, { force: true }).pipe(Effect.ignore)),
        )
        return yield* Effect.gen(function* () {
          const port = yield* findFreePort(config.preferredPort ?? 8080)
          if (port <= 0) return yield* runtimeError("ensure_serving", "server_start_failed", "Could not allocate a loopback port")
          const secret = Secret.fromString(randomBytes(32).toString("base64url"))
          const connection: LlamaCppConnection = {
            baseUrl: `http://127.0.0.1:${port}`,
            apiKey: Option.some(secret),
          }
          const command = Command.make(
            distribution.executablePath,
            "--models-preset", presetPath,
            "--host", "127.0.0.1",
            "--port", String(port),
          ).pipe(
            Command.workingDirectory(distribution.directory),
            Command.env({ ...process.env, LLAMA_API_KEY: Secret.value(secret) }),
          )
          const child = yield* Command.start(command).pipe(
            Effect.mapError((cause) => runtimeError("ensure_serving", "server_start_failed", "Could not start llama-server", cause)),
          )
          const stderr = yield* Ref.make("")
          yield* child.stderr.pipe(
            Stream.runForEach((chunk) => Ref.update(stderr, (current) => {
              const next = current + new TextDecoder().decode(chunk)
              return next.length > 32_768 ? next.slice(-32_768) : next
            })),
            Effect.ignore,
            Effect.forkScoped,
          )
          const handle: ManagedServerHandle = {
            serverId: `managed-${randomUUID()}`,
            process: child,
            connection,
            presetPath,
            request,
          }
          yield* Ref.set(pending, handle)
          yield* waitForReady(connection, stderr, child)
          yield* Effect.uninterruptible(
            Ref.set(state, handle).pipe(Effect.zipRight(Ref.set(pending, null))),
          )
          return handle
        }).pipe(
          Effect.onExit((exit) => Exit.isFailure(exit) ? cleanupPending : Effect.void),
        )
      }))

    return {
      current: Ref.get(state),
      ensure: (distribution, artifact, request) => ensureRaw(distribution, artifact, request).pipe(Effect.provide(context)),
      stop: stop.pipe(Effect.provide(context)),
    }
  })

const targetAt = (
  connection: LlamaCppConnection,
  ownership: "managed" | "external",
  serverId: string,
  request: Pick<EnsureServingRequest, "providerModelId" | "contextTokens">,
  expectedArtifact?: ResolvedModelArtifact,
): Effect.Effect<ServingTarget, LlamaCppRuntimeError, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const client = makeLlamaCppEndpointClient(connection)
    const health = yield* client.health
    if (health._tag !== "Ready") {
      return yield* runtimeError("ensure_serving", "endpoint_failed", `llama-server is not ready: ${health._tag}`)
    }
    const [models, props] = yield* Effect.all([
      client.models,
      client.propsForModel(request.providerModelId),
    ], { concurrency: 2 }).pipe(
      Effect.mapError((cause) => runtimeError("ensure_serving", "endpoint_failed", cause.reason, cause)),
    )
    const model = models.find((candidate) => candidate.id === request.providerModelId)
    if (!model) {
      return yield* runtimeError("ensure_serving", "identity_mismatch", `Server does not expose model ${request.providerModelId}`)
    }
    if (expectedArtifact) yield* verifyManagedIdentity(model, props, expectedArtifact)
    const context = props.default_generation_settings?.n_ctx
    const paddedRequestedContext = Math.ceil(request.contextTokens / 256) * 256
    if (context === undefined || context < request.contextTokens || context > paddedRequestedContext) {
      return yield* runtimeError(
        "ensure_serving",
        "context_mismatch",
        `Server configured ${context ?? "unknown"} context tokens; expected ${request.contextTokens}`,
      )
    }
    return {
      serverId,
      ownership,
      providerModelId: request.providerModelId,
      configuredContextTokens: context,
      metadata: verifiedMetadata(model, props),
      connection,
    }
  })

export const LlamaCppRuntimeLive = (
  config: LlamaCppRuntimeConfig,
): Layer.Layer<LlamaCppRuntime, never, RuntimeRequirements> => Layer.scoped(
  LlamaCppRuntime,
  Effect.gen(function* () {
    const context = yield* Effect.context<RuntimeRequirements | Scope.Scope>()
    const distribution = yield* LlamaCppDistribution
    const models = yield* LlamaCppModelStore
    const controller = yield* makeManagedController(config)
    yield* Effect.addFinalizer(() => controller.stop)

    const externalConnections = config.externalConnections ?? (() => Effect.succeed([]))
    const inspect = Effect.gen(function* () {
      const managed = yield* controller.current
      const external = yield* externalConnections()
      const visibleExternal = managed
        ? external.filter(({ connection }) =>
            normalizedConnectionUrl(connection) !== normalizedConnectionUrl(managed.connection))
        : external
      const [managedObservation, externalObservations] = yield* Effect.all([
        managed
          ? observeConnection(
              managed.connection,
              "managed",
              managed.serverId,
              managed.request.providerModelId,
            ).pipe(Effect.map(Option.some))
          : Effect.succeed(Option.none<LlamaCppServerObservation>()),
        Effect.all(visibleExternal.map(({ connectionId, connection }) => observeConnection(
          connection,
          "external",
          connectionId,
        )), { concurrency: 4 }),
      ], { concurrency: 2 })
      return {
        managed: Option.getOrNull(managedObservation),
        external: externalObservations,
      } satisfies LlamaCppRuntimeSnapshot
    })

    const ensureServing = (request: EnsureServingRequest) => Effect.gen(function* () {
      if (request.contextTokens <= 0) {
        return yield* runtimeError("ensure_serving", "model_unavailable", "Serving request does not have a viable fit plan")
      }
      if (request._tag === "External") {
        const external = (yield* externalConnections()).find(
          ({ connectionId }) => connectionId === request.connectionId,
        )
        if (!external) {
          return yield* runtimeError(
            "ensure_serving",
            "external_unavailable",
            `External llama.cpp connection ${request.connectionId} is unavailable`,
          )
        }
        const managed = yield* controller.current
        if (managed && normalizedConnectionUrl(managed.connection) === normalizedConnectionUrl(external.connection)) {
          return yield* runtimeError(
            "ensure_serving",
            "external_unavailable",
            `Connection ${request.connectionId} belongs to Magnitude's managed router`,
          )
        }
        return yield* targetAt(external.connection, "external", external.connectionId, request)
      }
      if (!request.fitPlan.fits || request.fitPlan.parallelSlots <= 0) {
        return yield* runtimeError("ensure_serving", "model_unavailable", "Serving request does not have a viable fit plan")
      }

      const distributionState = yield* distribution.inspect.pipe(
        Effect.mapError((cause) => runtimeError("ensure_serving", "distribution_unavailable", cause.reason, cause)),
      )
      if (distributionState._tag !== "Ready") {
        return yield* runtimeError("ensure_serving", "distribution_unavailable", `Distribution is ${distributionState._tag}`)
      }
      const artifact = yield* models.resolve(request.modelId).pipe(
        Effect.mapError((cause) => runtimeError("ensure_serving", "model_unavailable", cause.reason, cause)),
      )
      const handle = yield* controller.ensure(distributionState.distribution, artifact, request)
      return yield* targetAt(handle.connection, "managed", handle.serverId, request, artifact).pipe(
        Effect.tapError(() => controller.stop),
      )
    })

    return LlamaCppRuntime.of({
      inspect: inspect.pipe(Effect.provide(context)),
      ensureServing: (request) => ensureServing(request).pipe(Effect.provide(context)),
      stopManaged: controller.stop.pipe(Effect.provide(context)),
    })
  }),
)
