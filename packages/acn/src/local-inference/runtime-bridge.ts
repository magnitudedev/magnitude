import * as os from "node:os"
import { Context, Effect, Layer, Stream } from "effect"
import {
  SessionOperationFailed,
  type LocalInferenceCapabilities,
  type LocalModelChoice,
} from "@magnitudedev/protocol"
import { MagnitudeStorage } from "@magnitudedev/storage"
import { resolveLlamaCppAuth, type EndpointProviderAuthConfig } from "../shared-client"
import { LOCAL_MODEL_CATALOG } from "./catalog"
import type { LlamaCppRuntimeBridgeShape, QuantBitsClass } from "./types"

export class LlamaCppRuntimeBridge extends Context.Tag("LlamaCppRuntimeBridge")<
  LlamaCppRuntimeBridge,
  LlamaCppRuntimeBridgeShape
>() {}

const pending = (operation: string) => new SessionOperationFailed({
  operation,
  reason: "The daemon llama.cpp mechanism adapter has not landed yet. The onboarding policy and protocol are ready, but this operation must be delegated to the CTO-owned llama.cpp package.",
})

const failed = (operation: string, cause: unknown) => new SessionOperationFailed({
  operation,
  reason: cause instanceof Error ? cause.message : String(cause),
})

interface EndpointModel {
  readonly id: string
  readonly contextTokens: number
  readonly modelMaximumContextTokens?: number
  readonly sizeBytes?: number
  readonly totalParametersBillions?: number
  readonly quant?: string
}

interface EndpointProbe {
  readonly models: readonly EndpointModel[]
  readonly build?: string
}

const positiveNumber = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined

const quantBitsClass = (format: string): QuantBitsClass => {
  const normalized = format.toLowerCase()
  if (normalized.includes("mxfp4")) return "mxfp4"
  if (normalized.includes("fp8")) return "fp8"
  if (normalized.includes("q4")) return "q4"
  if (normalized.includes("q5")) return "q5"
  if (normalized.includes("q6")) return "q6"
  if (normalized.includes("q8")) return "q8"
  return "other"
}

const displayName = (modelId: string): string => {
  const knownArtifact = LOCAL_MODEL_CATALOG.find((entry) =>
    entry.files.some((file) => modelId === file.path || modelId.endsWith(`/${file.path}`)),
  )
  return knownArtifact?.displayName ?? modelId
}

const endpointHeaders = (config: EndpointProviderAuthConfig): Headers => {
  const headers = new Headers()
  if (config.apiKey) headers.set("authorization", `Bearer ${config.apiKey}`)
  return headers
}

const probeEndpoint = (
  config: EndpointProviderAuthConfig,
): Effect.Effect<EndpointProbe, SessionOperationFailed> => Effect.tryPromise({
  try: async () => {
    const endpoint = config.endpoint.replace(/\/$/, "")
    const getJson = async (path: string): Promise<unknown> => {
      const response = await fetch(`${endpoint}${path}`, {
        headers: endpointHeaders(config),
        signal: AbortSignal.timeout(5_000),
      })
      if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`)
      return response.json()
    }

    const health = await getJson("/health")
    if (typeof health !== "object" || health === null || (health as { status?: unknown }).status !== "ok") {
      throw new Error("/health did not report status ok")
    }
    const modelsResponse = await getJson("/v1/models")
    if (typeof modelsResponse !== "object" || modelsResponse === null) {
      throw new Error("/v1/models returned an invalid response")
    }
    const rawModels = (modelsResponse as { data?: unknown }).data
    if (!Array.isArray(rawModels) || rawModels.length === 0) {
      throw new Error("/v1/models reported no loaded models")
    }

    let props: Record<string, unknown> = {}
    for (const path of ["/props", "/v1/props"]) {
      try {
        const value = await getJson(path)
        if (typeof value === "object" && value !== null) props = value as Record<string, unknown>
        break
      } catch {
        // /props is optional and older/forked servers may expose /v1/props.
      }
    }
    const settings = typeof props.default_generation_settings === "object"
      && props.default_generation_settings !== null
      ? props.default_generation_settings as Record<string, unknown>
      : {}
    const propsContext = positiveNumber(settings.n_ctx)
    const propsQuant = typeof props.model_ftype === "string" ? props.model_ftype : undefined

    const models: EndpointModel[] = []
    for (const raw of rawModels) {
      if (typeof raw !== "object" || raw === null) continue
      const value = raw as Record<string, unknown>
      if (typeof value.id !== "string" || value.id.length === 0) continue
      const meta = typeof value.meta === "object" && value.meta !== null
        ? value.meta as Record<string, unknown>
        : {}
      const contextTokens = rawModels.length === 1
        ? propsContext ?? positiveNumber(meta.n_ctx)
        : positiveNumber(meta.n_ctx)
      if (!contextTokens) continue
      models.push({
        id: value.id,
        contextTokens,
        ...(positiveNumber(meta.n_ctx_train) !== undefined
          ? { modelMaximumContextTokens: positiveNumber(meta.n_ctx_train) }
          : {}),
        ...(positiveNumber(meta.size) !== undefined ? { sizeBytes: positiveNumber(meta.size) } : {}),
        ...(positiveNumber(meta.n_params) !== undefined
          ? { totalParametersBillions: positiveNumber(meta.n_params)! / 1_000_000_000 }
          : {}),
        ...(typeof meta.ftype === "string"
          ? { quant: meta.ftype }
          : propsQuant ? { quant: propsQuant } : {}),
      })
    }
    if (models.length === 0) {
      throw new Error("No loaded model reported a usable model ID and configured context")
    }
    return {
      models,
      ...(typeof props.build_info === "string" ? { build: props.build_info } : {}),
    }
  },
  catch: (cause) => failed("probe configured llama.cpp endpoint", cause),
})

const endpointChoice = (
  config: EndpointProviderAuthConfig,
  model: EndpointModel,
): LocalModelChoice => ({
  choiceId: `running:${encodeURIComponent(config.endpoint)}:${encodeURIComponent(model.id)}`,
  source: "running",
  displayName: displayName(model.id),
  providerModelId: model.id,
  serverId: config.endpoint,
  ...(model.quant ? {
    quantization: {
      format: model.quant,
      bitsClass: quantBitsClass(model.quant),
      quantAwareCheckpoint: false,
      fidelityLabel: `Server-reported ${model.quant}`,
      fidelityEvidence: "The running llama.cpp server reported this loaded quant. Artifact revision and quantization-aware-training status are not inferred from a filename.",
      fidelitySourceUrl: "https://github.com/ggml-org/llama.cpp",
    },
  } : {}),
  ...(model.sizeBytes !== undefined ? { sizeBytes: model.sizeBytes } : {}),
  ...(model.totalParametersBillions !== undefined
    ? { totalParametersBillions: model.totalParametersBillions }
    : {}),
  contextTokens: model.contextTokens,
  ...(model.modelMaximumContextTokens !== undefined
    ? { modelMaximumContextTokens: model.modelMaximumContextTokens }
    : {}),
  fitClass: "unknown",
  managed: false,
  compatible: model.contextTokens >= 16_384,
  explanation: "Discovered from an already-running configured llama.cpp endpoint. Selecting it attaches Magnitude without restarting or replacing the server.",
})

/**
 * Opt-in, attach-only bridge for exercising onboarding against a user's
 * configured llama.cpp server before the managed mechanism layer lands.
 * It never downloads models or starts, stops, or restarts a server.
 */
export const LlamaCppRuntimeBridgeEndpointTestLive: Layer.Layer<
  LlamaCppRuntimeBridge,
  never,
  MagnitudeStorage
> = Layer.effect(
  LlamaCppRuntimeBridge,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    const config = yield* resolveLlamaCppAuth(storage)
    if (!config) return LlamaCppRuntimeBridge.of({
      getReadiness: Effect.succeed({
        status: "error",
        canDownload: false,
        canActivate: false,
        diagnostic: "No llama.cpp endpoint is configured.",
      }),
      getCapabilities: Effect.succeed({
        binary: { identity: "llama.cpp-endpoint-attach-test" },
        system: { totalMemoryBytes: os.totalmem(), cpuModel: os.cpus()[0]?.model, logicalCores: os.cpus().length },
        accelerators: [],
        warnings: [],
      }),
      getInventory: Effect.succeed({ running: [], downloaded: [] }),
      startDownload: () => Effect.fail(pending("start local model download")),
      subscribeDownload: () => Stream.fail(pending("subscribe to local model download")),
      cancelDownload: () => Effect.fail(pending("cancel local model download")),
      activate: () => Effect.fail(pending("activate local model")),
    })

    const probe = probeEndpoint(config)
    const capabilities = (build?: string): LocalInferenceCapabilities => ({
      binary: {
        identity: `configured-llama.cpp-endpoint:${config.endpoint}`,
        ...(build ? { version: build } : {}),
      },
      system: {
        totalMemoryBytes: os.totalmem(),
        cpuModel: os.cpus()[0]?.model,
        logicalCores: os.cpus().length,
      },
      accelerators: [],
      warnings: [{
        code: "llamacpp_endpoint_attach_test_no_device_probe",
        message: "This running llama.cpp server does not report accelerator capacity.",
      }],
    })

    return LlamaCppRuntimeBridge.of({
      getReadiness: probe.pipe(Effect.match({
        onFailure: (error) => ({
          status: "error" as const,
          canDownload: false,
          canActivate: false,
          diagnostic: `Attach-only llama.cpp test failed: ${error.reason}`,
        }),
        onSuccess: () => ({
          status: "ready" as const,
          canDownload: false,
          canActivate: true,
          diagnostic: "Connected to the running llama.cpp server.",
        }),
      })),
      getCapabilities: probe.pipe(
        Effect.map((result) => capabilities(result.build)),
        Effect.catchAll(() => Effect.succeed(capabilities())),
      ),
      getInventory: probe.pipe(
        Effect.map((result) => ({
          running: result.models.map((model) => endpointChoice(config, model)),
          downloaded: [],
        })),
        Effect.catchAll(() => Effect.succeed({ running: [], downloaded: [] })),
      ),
      startDownload: () => Effect.fail(new SessionOperationFailed({
        operation: "start local model download",
        reason: "Attach-only endpoint test mode does not download artifacts. TODO(llamacpp-hf-download-integration): delegate the exact pinned Hugging Face source to the CTO-owned llama.cpp download/cache primitive.",
      })),
      subscribeDownload: () => Stream.fail(new SessionOperationFailed({
        operation: "subscribe to local model download",
        reason: "Attach-only endpoint test mode has no download operation.",
      })),
      cancelDownload: () => Effect.fail(new SessionOperationFailed({
        operation: "cancel local model download",
        reason: "Attach-only endpoint test mode has no download operation.",
      })),
      activate: (selection) => Effect.gen(function* () {
        if (!("source" in selection) || selection.source !== "running" || !selection.compatible) {
          return yield* new SessionOperationFailed({
            operation: "attach to running local model",
            reason: "Attach-only endpoint test mode can activate only a compatible model discovered from the configured running server.",
          })
        }
        const current = yield* probe
        const model = current.models.find((item) => item.id === selection.providerModelId)
        if (!model) {
          return yield* new SessionOperationFailed({
            operation: "attach to running local model",
            reason: "The selected model is no longer reported by the configured llama.cpp server.",
          })
        }
        return {
          providerId: "llamacpp",
          providerModelId: model.id,
          contextTokens: model.contextTokens,
        }
      }),
    })
  }),
)

/**
 * TODO(llamacpp-binary-bootstrap-integration, CTO-owned): Replace the pending
 * readiness value below with the llama.cpp package's binary bootstrap API.
 * That package owns this entire boundary:
 *
 * 1. Detect whether its correct managed llama.cpp binary exists for the host
 *    platform/architecture. Do not search PATH or infer readiness in ACN.
 * 2. When it is absent, report that state so onboarding can offer Install.
 * 3. After the user accepts, download through the package's platform-specific
 *    installer, including progress/cancellation, integrity verification,
 *    executable permissions, and a stable managed binary identity/version.
 * 4. Report ready only after the installed binary can execute its capability
 *    probe. ACN then continues into hardware detection and model choices.
 *
 * Do not implement a second binary URL table, HTTP downloader, archive
 * extractor, PATH scanner, or installation directory in onboarding/ACN. The
 * final package adapter should replace this stub at the composition seam in
 * server.ts; the attach-only endpoint test bridge remains independent.
 *
 * This fallback keeps recommendation policy inspectable while that work is
 * pending, without pretending that binary detection or installation occurred.
 */
export const LlamaCppRuntimeBridgePendingLive = Layer.succeed(
  LlamaCppRuntimeBridge,
  LlamaCppRuntimeBridge.of({
    getReadiness: Effect.succeed({
      status: "integration_pending",
      canDownload: false,
      canActivate: false,
      diagnostic: "Waiting for the CTO-owned managed llama.cpp binary detection and installation adapter. Model recommendations are available from stable system capacity, but installation, model download, and activation are intentionally disabled.",
    }),

    // TODO(llamacpp-package-integration): Replace this adapter with the final
    // exact-managed-binary capability API; do not add vendor-specific probes
    // here. The final result must combine os.totalmem() with normalized output
    // from the same managed llama-server binary's --list-devices operation.
    // Set modelSplitGroupId only when that backend can split one model across
    // the distinct reported physical devices.
    getCapabilities: Effect.sync(() => ({
      binary: { identity: "managed-llama.cpp-integration-pending" },
      system: {
        totalMemoryBytes: os.totalmem(),
        cpuModel: os.cpus()[0]?.model,
        logicalCores: os.cpus().length,
      },
      accelerators: [],
      warnings: [{
        code: "llamacpp_capability_integration_pending",
        message: "Accelerator capacity is not available yet.",
      }],
    })),

    // TODO(llamacpp-runtime-discovery): Replace the empty inventory with the
    // daemon's final discovered llama.cpp server and verified cache inventory.
    // Preserve arbitrary server-reported model IDs and structured metadata.
    getInventory: Effect.succeed({ running: [], downloaded: [] }),

    // TODO(llamacpp-hf-download-integration): Pass the catalog's exact Hugging
    // Face repo, immutable revision, quant tag, and file/hash expectations to
    // the final llama.cpp model download/cache API. Consume normalized progress
    // and returned cache identity; validate free space on the actual cache
    // filesystem and verify sizes/hashes before activation. Do not add an ACN
    // HTTP downloader or assume a cache path here.
    startDownload: () => Effect.fail(pending("start local model download")),
    subscribeDownload: () => Stream.fail(pending("subscribe to local model download")),
    cancelDownload: () => Effect.fail(pending("cancel local model download")),

    // TODO(llamacpp-lifecycle-integration): Delegate attach/start/restart to
    // the daemon's final llama.cpp lifecycle service; never spawn a second
    // server manager or restart an unrelated external server from onboarding.
    //
    // TODO(llamacpp-fit-integration): Start the selected model with its exact
    // catalog context through the final --fit lifecycle API; use
    // llama-fit-params only if the runtime bundle exposes it, and never
    // silently lower the selected context.
    //
    // TODO(llamacpp-model-metadata): Return verified provider/model identity,
    // loaded quantization, byte size, and actual configured context from the
    // final discovery/provider contract before onboarding is completed.
    activate: () => Effect.fail(pending("activate local model")),
  }),
)
