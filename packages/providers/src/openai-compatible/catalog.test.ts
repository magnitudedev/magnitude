import { afterEach, describe, expect, it } from "vitest"
import { FetchHttpClient } from "@effect/platform"
import { Effect } from "effect"
import { Auth, ModelCatalogError } from "@magnitudedev/ai"
import type { ModelsDevClient, ModelsDevModel, ModelsDevProvider } from "../catalog/models-dev"
import { createOpenAiCompatibleCatalog } from "./catalog"

const servers: Bun.Server<unknown>[] = []

afterEach(() => {
  for (const server of servers.splice(0)) server.stop(true)
})

function metadata(
  id: string,
  overrides?: Partial<ModelsDevModel>,
): ModelsDevModel {
  return {
    id,
    name: `Metadata ${id}`,
    attachment: false,
    reasoning: true,
    tool_call: true,
    structured_output: true,
    open_weights: true,
    modalities: { input: ["text"], output: ["text"] },
    limit: { context: 131_072, output: 16_384 },
    cost: { input: 1, output: 2 },
    ...overrides,
  }
}

function modelsDev(provider: ModelsDevProvider): ModelsDevClient {
  return {
    getProvider: (providerId) => Effect.succeed(providerId === provider.id ? provider : null),
    refresh: Effect.succeed({ [provider.id]: provider }),
  }
}

describe("OpenAI-compatible catalog enrichment", () => {
  it("prefers an exact model ID over a coarse models.dev family", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ data: [{ id: "glm-4.7" }] }),
    })
    servers.push(server)
    const provider: ModelsDevProvider = {
      id: "zai",
      name: "Z.AI",
      models: {
        "glm-4.7": metadata("glm-4.7", { family: "glm", name: "GLM-4.7" }),
      },
    }
    const catalog = createOpenAiCompatibleCatalog({
      providerId: "zai",
      endpoint: server.url.toString().replace(/\/$/, ""),
      auth: Auth.none,
      modelsDevProviderId: "zai",
      modelsDev: modelsDev(provider),
    })

    const result = await Effect.runPromise(catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))

    expect(result[0]?.modelFamilyId).toBe("glm-4")
    expect(result[0]?.upstreamFamily).toBe("glm")
    expect(result[0]?.displayName).toBe("GLM-4.7")
  })

  it("preserves common model acronyms in fallback display names", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ data: [{ id: "glm-4.6" }, { id: "openai/gpt-oss-20b" }] }),
    })
    servers.push(server)
    const emptyProvider: ModelsDevProvider = { id: "direct", name: "Direct", models: {} }
    const catalog = createOpenAiCompatibleCatalog({
      providerId: "direct",
      endpoint: server.url.toString().replace(/\/$/, ""),
      auth: Auth.none,
      modelsDevProviderId: "direct",
      modelsDev: modelsDev(emptyProvider),
    })

    const result = await Effect.runPromise(catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))

    expect(result.map((model) => model.displayName)).toEqual(["GLM 4.6", "GPT OSS 20B"])
  })

  it("keeps live direct-provider IDs when metadata has not caught up", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ data: [{ id: "brand-new-model" }] }),
    })
    servers.push(server)
    const unavailableMetadata: ModelsDevClient = {
      getProvider: () => Effect.fail(new ModelCatalogError({ message: "metadata unavailable" })),
      refresh: Effect.fail(new ModelCatalogError({ message: "metadata unavailable" })),
    }
    const catalog = createOpenAiCompatibleCatalog({
      providerId: "direct",
      endpoint: server.url.toString().replace(/\/$/, ""),
      auth: Auth.none,
      modelsDevProviderId: "direct",
      modelsDev: unavailableMetadata,
      requireToolCalls: true,
    })

    const result = await Effect.runPromise(catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))

    expect(result).toHaveLength(1)
    expect(result[0]).toMatchObject({
      providerModelId: "brand-new-model",
      displayName: "Brand New Model",
      modelFamilyId: "unknown",
      metadataSource: "provider",
    })
  })

  it("fails closed when open-weight metadata is unavailable", async () => {
    const unavailableMetadata: ModelsDevClient = {
      getProvider: () => Effect.fail(new ModelCatalogError({ message: "metadata unavailable" })),
      refresh: Effect.fail(new ModelCatalogError({ message: "metadata unavailable" })),
    }
    const catalog = createOpenAiCompatibleCatalog({
      providerId: "router",
      endpoint: "http://127.0.0.1:1",
      auth: Auth.none,
      modelsDevProviderId: "router",
      modelsDev: unavailableMetadata,
      requireOpenWeights: true,
      requireToolCalls: true,
    })

    const result = await Effect.runPromiseExit(
      catalog.list.pipe(Effect.provide(FetchHttpClient.layer)),
    )

    expect(result._tag).toBe("Failure")
  })

  it("preserves exact provider IDs and filters routers solely from exact models.dev metadata", async () => {
    let authorization = ""
    const liveModels = [
      {
        id: "openai/gpt-oss-120b:free",
        name: "GPT OSS Free",
        supported_parameters: ["tools", "tool_choice", "reasoning"],
        reasoning: {
          supported_efforts: ["high", "medium", "low", "none"],
          default_effort: "medium",
          default_enabled: true,
          mandatory: false,
        },
      },
      { id: "google/gemma-3-27b-it" },
      { id: "meta-llama/llama-4-scout" },
      { id: "open/provider-no-tools", supported_parameters: ["reasoning"] },
      { id: "closed/model" },
      { id: "open/no-tools" },
      { id: "open/non-text" },
      { id: "missing/exact-metadata" },
    ]
    const server = Bun.serve({
      port: 0,
      fetch(request) {
        authorization = request.headers.get("authorization") ?? ""
        return Response.json({ data: liveModels })
      },
    })
    servers.push(server)

    const provider: ModelsDevProvider = {
      id: "router",
      name: "Router",
      models: {
        "openai/gpt-oss-120b:free": metadata("openai/gpt-oss-120b:free", { family: "gpt-oss" }),
        "google/gemma-3-27b-it": metadata("google/gemma-3-27b-it", { family: "gemma-3" }),
        "meta-llama/llama-4-scout": metadata("meta-llama/llama-4-scout", { family: "llama-4" }),
        "open/provider-no-tools": metadata("open/provider-no-tools"),
        "closed/model": metadata("closed/model", { open_weights: false }),
        "open/no-tools": metadata("open/no-tools", { tool_call: false }),
        "open/non-text": metadata("open/non-text", { modalities: { input: ["text"], output: ["image"] } }),
      },
    }
    const catalog = createOpenAiCompatibleCatalog({
      providerId: "router",
      endpoint: server.url.toString().replace(/\/$/, ""),
      auth: Auth.bearer("router-key"),
      modelsDevProviderId: "router",
      modelsDev: modelsDev(provider),
      requireOpenWeights: true,
      requireToolCalls: true,
    })

    const result = await Effect.runPromise(catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))

    expect(result.map((model) => model.providerModelId)).toEqual([
      "openai/gpt-oss-120b:free",
      "google/gemma-3-27b-it",
      "meta-llama/llama-4-scout",
    ])
    expect(result[0]).toMatchObject({
      displayName: "GPT OSS Free",
      modelFamilyId: "gpt-oss",
      openWeightStatus: "open",
      metadataSource: "models.dev",
      capabilities: { grammar: false, toolCalls: true },
      reasoningEfforts: ["high", "medium", "low", "none"],
    })
    expect(authorization).toBe("Bearer router-key")
  })

  it("keeps direct-provider models with unknown family visible", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ data: [{
        id: "unusual/vendor-model.v9",
        context_window: 222_000,
        max_tokens: 44_000,
        description: "Provider description",
      }] }),
    })
    servers.push(server)
    const provider: ModelsDevProvider = {
      id: "direct",
      name: "Direct",
      models: { "unusual/vendor-model.v9": metadata("unusual/vendor-model.v9") },
    }
    const catalog = createOpenAiCompatibleCatalog({
      providerId: "direct",
      endpoint: server.url.toString().replace(/\/$/, ""),
      auth: Auth.bearer("key"),
      modelsDevProviderId: "direct",
      modelsDev: modelsDev(provider),
      requireToolCalls: true,
    })

    const result = await Effect.runPromise(catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))

    expect(result).toHaveLength(1)
    expect(result[0]?.providerModelId).toBe("unusual/vendor-model.v9")
    expect(result[0]?.modelFamilyId).toBe("unknown")
    expect(result[0]?.reasoningEfforts).toEqual(["default"])
    expect(result[0]).toMatchObject({
      contextWindow: 222_000,
      maxOutputTokens: 44_000,
      description: "Provider description",
    })
  })

  it("uses provider-authoritative Kimi capability flags when metadata lags", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ data: [{
        id: "kimi-k2.7-preview",
        supports_reasoning: true,
        supports_image_in: true,
        supports_video_in: true,
      }] }),
    })
    servers.push(server)
    const provider: ModelsDevProvider = {
      id: "kimi",
      name: "Kimi",
      models: {
        "kimi-k2.7-preview": metadata("kimi-k2.7-preview", {
          reasoning: true,
          attachment: false,
          modalities: { input: ["text"], output: ["text"] },
        }),
      },
    }
    const catalog = createOpenAiCompatibleCatalog({
      providerId: "kimi",
      endpoint: server.url.toString().replace(/\/$/, ""),
      auth: Auth.none,
      modelsDevProviderId: "kimi",
      modelsDev: modelsDev(provider),
    })

    const result = await Effect.runPromise(catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))

    expect(result[0]?.reasoningEfforts).toEqual(["none", "high"])
    expect(result[0]?.capabilities.vision).toBe(true)
    expect(result[0]?.modalities?.input).toEqual(["text", "image", "video"])
  })
})
