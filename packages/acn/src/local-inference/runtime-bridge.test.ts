import { Effect, Layer } from "effect"
import { afterEach, describe, expect, test } from "vitest"
import { MagnitudeStorage, type MagnitudeStorageShape } from "@magnitudedev/storage"
import {
  LlamaCppRuntimeBridge,
  LlamaCppRuntimeBridgeEndpointTestLive,
} from "./runtime-bridge"

const servers: Bun.Server<unknown>[] = []
afterEach(() => {
  for (const server of servers.splice(0)) server.stop(true)
})

const storageFor = (endpoint: string): MagnitudeStorageShape => ({
  auth: {
    loadAll: () => Effect.succeed({}),
    get: () => Effect.succeed({ type: "endpoint", endpoint }),
    set: () => Effect.void,
    remove: () => Effect.void,
  },
} as unknown as MagnitudeStorageShape)

const runBridge = <A>(
  endpoint: string,
  effect: Effect.Effect<A, unknown, LlamaCppRuntimeBridge>,
): Promise<A> => Effect.runPromise(effect.pipe(Effect.provide(
  LlamaCppRuntimeBridgeEndpointTestLive.pipe(
    Layer.provide(Layer.succeed(MagnitudeStorage, storageFor(endpoint))),
  ),
)))

describe("attach-only llama.cpp endpoint bridge", () => {
  test("discovers and attaches to a live model without managing the server", async () => {
    const server = Bun.serve({
      port: 0,
      fetch(request) {
        const path = new URL(request.url).pathname
        if (path === "/health") return Response.json({ status: "ok" })
        if (path === "/v1/models") return Response.json({
          object: "list",
          data: [{
            id: "/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf",
            object: "model",
            meta: {
              n_ctx: 65_536,
              n_ctx_train: 262_144,
              n_params: 35_505_251_456,
              size: 31_843_777_504,
              ftype: "Q6_K",
            },
          }],
        })
        if (path === "/props") return Response.json({
          build_info: "test-build",
          model_ftype: "Q6_K",
          default_generation_settings: { n_ctx: 65_536 },
        })
        return new Response("missing", { status: 404 })
      },
    })
    servers.push(server)
    const endpoint = `http://127.0.0.1:${server.port}`

    const result = await runBridge(endpoint, Effect.gen(function* () {
      const bridge = yield* LlamaCppRuntimeBridge
      const readiness = yield* bridge.getReadiness
      const inventory = yield* bridge.getInventory
      const activated = yield* bridge.activate(inventory.running[0]!)
      return { readiness, inventory, activated }
    }))

    expect(result.readiness).toMatchObject({ status: "ready", canDownload: false, canActivate: true })
    expect(result.inventory.running).toEqual([expect.objectContaining({
      displayName: "Qwen3.6 35B-A3B",
      providerModelId: "/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf",
      contextTokens: 65_536,
      totalParametersBillions: 35.505251456,
      managed: false,
      compatible: true,
      quantization: expect.objectContaining({
        format: "Q6_K",
        bitsClass: "q6",
        fidelityLabel: "Very high fidelity with minimal quality loss",
      }),
    })])
    expect(result.activated).toEqual({
      providerId: "llamacpp",
      providerModelId: "/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf",
      contextTokens: 65_536,
    })
  })

  test("reports an unavailable endpoint without breaking the onboarding snapshot", async () => {
    const result = await runBridge("http://127.0.0.1:1", Effect.gen(function* () {
      const bridge = yield* LlamaCppRuntimeBridge
      return {
        readiness: yield* bridge.getReadiness,
        inventory: yield* bridge.getInventory,
        capabilities: yield* bridge.getCapabilities,
      }
    }))
    expect(result.readiness.status).toBe("error")
    expect(result.inventory).toEqual({ running: [], downloaded: [] })
    expect(result.capabilities.system.totalMemoryBytes).toBeGreaterThan(0)
  })
})
