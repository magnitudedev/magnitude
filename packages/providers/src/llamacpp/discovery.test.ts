import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { Effect } from "effect"
import {
  GGUFValueType,
  serializeGgufMetadata,
  type GGUFTypedMetadata,
} from "@huggingface/gguf"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientError from "@effect/platform/HttpClientError"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import {
  checkServerHealth,
  deriveContextWindow,
  deriveDisplayName,
  deriveSourceModelPath,
  detectVision,
  fetchModelList,
  fetchServerProps,
} from "./discovery"
import { createLlamaCppProvider } from "./provider"
import { makeProviderRegistry } from "../registry"
import { createMagnitudeProvider } from "../magnitude/provider"

function responseClient(
  respond: (url: string) => Response,
): HttpClient.HttpClient {
  return HttpClient.make((request) =>
    Effect.succeed(HttpClientResponse.fromWeb(request, respond(request.url))),
  )
}

async function writeGgufFixture(
  path: string,
  values: Readonly<Record<string, string>>,
): Promise<void> {
  const entries = Object.fromEntries(
    Object.entries(values).map(([key, value]) => [
      key,
      { value, type: GGUFValueType.STRING },
    ]),
  )
  const metadata = {
    version: { value: 3, type: GGUFValueType.UINT32 },
    tensor_count: { value: 0n, type: GGUFValueType.UINT64 },
    kv_count: { value: BigInt(Object.keys(entries).length), type: GGUFValueType.UINT64 },
    ...entries,
  } as GGUFTypedMetadata
  await writeFile(path, serializeGgufMetadata(metadata))
}

describe("llama.cpp discovery", () => {
  it("maps ready and loading health responses", async () => {
    const readyClient = responseClient(() => new Response(
      JSON.stringify({ status: "ok" }),
      { status: 200 },
    ))
    const loadingClient = responseClient(() => new Response(
      JSON.stringify({ error: { message: "Loading model" } }),
      { status: 503 },
    ))

    const ready = await Effect.runPromise(
      checkServerHealth("http://127.0.0.1:8080").pipe(
        Effect.provideService(HttpClient.HttpClient, readyClient),
      ),
    )
    const loading = await Effect.runPromise(
      checkServerHealth("http://127.0.0.1:8080").pipe(
        Effect.provideService(HttpClient.HttpClient, loadingClient),
      ),
    )

    expect(ready).toEqual({ status: "ready", endpoint: "http://127.0.0.1:8080" })
    expect(loading).toEqual({ status: "loading", endpoint: "http://127.0.0.1:8080" })
  })

  it("maps transport failures to not_found", async () => {
    const client = HttpClient.make((request) =>
      Effect.fail(new HttpClientError.RequestError({
        request,
        reason: "Transport",
        cause: new Error("connection refused"),
      })),
    )

    const result = await Effect.runPromise(
      checkServerHealth("http://127.0.0.1:8080").pipe(
        Effect.provideService(HttpClient.HttpClient, client),
      ),
    )

    expect(result).toEqual({ status: "not_found", endpoint: "http://127.0.0.1:8080" })
  })

  it("uses the standard llama-server endpoint by default", async () => {
    let requestedUrl = ""
    const client = responseClient((url) => {
      requestedUrl = url
      return new Response("", { status: 503 })
    })

    const result = await Effect.runPromise(
      createLlamaCppProvider().checkStatus.pipe(
        Effect.provideService(HttpClient.HttpClient, client),
      ),
    )

    expect(requestedUrl).toBe("http://127.0.0.1:8080/health")
    expect(result.status).toBe("loading")
  })

  it("discovers absolute GGUF model paths", async () => {
    const modelId = "/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"
    const client = responseClient((url) => {
      switch (new URL(url).pathname) {
        case "/health":
          return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
        case "/v1/models":
          return new Response(JSON.stringify({
            object: "list",
            data: [{
              id: modelId,
              object: "model",
              meta: { n_ctx: 200_192, n_ctx_train: 262_144, ftype: "Q6_K" },
            }],
          }), { status: 200 })
        case "/props":
          return new Response(JSON.stringify({
            default_generation_settings: { n_ctx: 200_192 },
            model_alias: modelId,
            model_ftype: "Q6_K",
            model_path: modelId,
            modalities: { vision: false, audio: false },
          }), { status: 200 })
        default:
          return new Response("", { status: 404 })
      }
    })

    const result = await Effect.runPromise(
      createLlamaCppProvider().checkStatus.pipe(
        Effect.provideService(HttpClient.HttpClient, client),
      ),
    )

    expect(result.status).toBe("ok")
    expect(result.models).toEqual([
      expect.objectContaining({
        providerId: "llamacpp",
        providerModelId: modelId,
        displayName: "Qwen3.6-35B-A3B (UD-Q6_K_XL)",
        modelFamilyId: "qwen-3.5",
        contextWindow: 200_192,
      }),
    ])
  })

  it("reads embedded GGUF names and tokenizer-family evidence from renamed files", async () => {
    const directory = await mkdtemp(join(tmpdir(), "magnitude-gguf-"))
    const modelPath = join(directory, "opaque-local-build-Q4_K_M.gguf")
    await writeGgufFixture(modelPath, {
      "general.name": "Qwen3.6-35B-A3B",
      "general.basename": "Qwen3.6",
      "general.size_label": "35B-A3B",
      "general.architecture": "qwen35moe",
      "tokenizer.ggml.model": "gpt2",
      "tokenizer.ggml.pre": "qwen35",
      "general.base_model.0.name": "Qwen3.6 35B A3B",
      "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
    })

    try {
      const client = responseClient((url) => {
        switch (new URL(url).pathname) {
          case "/health":
            return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
          case "/v1/models":
            return new Response(JSON.stringify({
              object: "list",
              data: [{ id: "opaque-model-alias", object: "model", meta: { n_ctx: 32_768 } }],
            }), { status: 200 })
          case "/props":
            return new Response(JSON.stringify({
              default_generation_settings: { n_ctx: 32_768 },
              model_alias: "opaque-model-alias",
              model_ftype: "Q4_K",
              model_path: modelPath,
              modalities: { vision: false, audio: false },
            }), { status: 200 })
          default:
            return new Response("", { status: 404 })
        }
      })

      const result = await Effect.runPromise(
        createLlamaCppProvider().catalog.list.pipe(
          Effect.provideService(HttpClient.HttpClient, client),
        ),
      )

      expect(result).toEqual([
        expect.objectContaining({
          providerModelId: "opaque-model-alias",
          sourceModelPath: modelPath,
          displayName: "Qwen3.6-35B-A3B (Q4_K_M)",
          metadataName: "Qwen3.6-35B-A3B",
          modelArchitecture: "qwen35moe",
          tokenizerModel: "gpt2",
          tokenizerPre: "qwen35",
          modelFamilyId: "qwen-3.5",
          baseModelNames: ["Qwen3.6 35B A3B"],
          baseModelRepositories: ["https://huggingface.co/Qwen/Qwen3.6-35B-A3B"],
        }),
      ])
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  })

  it("does not trust a Qwen name when embedded tokenizer metadata conflicts", async () => {
    const directory = await mkdtemp(join(tmpdir(), "magnitude-gguf-"))
    const modelPath = join(directory, "renamed.gguf")
    await writeGgufFixture(modelPath, {
      "general.name": "Qwen3.6-35B-A3B",
      "general.architecture": "qwen35moe",
      "tokenizer.ggml.model": "gpt2",
      "tokenizer.ggml.pre": "qwen2",
    })

    try {
      const client = responseClient((url) => {
        switch (new URL(url).pathname) {
          case "/health":
            return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
          case "/v1/models":
            return new Response(JSON.stringify({
              object: "list",
              data: [{ id: "opaque", object: "model", meta: { n_ctx: 8_192 } }],
            }), { status: 200 })
          case "/props":
            return new Response(JSON.stringify({
              default_generation_settings: { n_ctx: 8_192 },
              model_path: modelPath,
              modalities: { vision: false, audio: false },
            }), { status: 200 })
          default:
            return new Response("", { status: 404 })
        }
      })

      const models = await Effect.runPromise(
        createLlamaCppProvider().catalog.list.pipe(
          Effect.provideService(HttpClient.HttpClient, client),
        ),
      )

      expect(models[0]).toEqual(expect.objectContaining({
        displayName: "Qwen3.6-35B-A3B",
        modelFamilyId: "unknown",
      }))
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  })

  it("keeps arbitrary unclassified aliases in the catalog", async () => {
    const modelId = "acme/custom.model:latest"
    const client = responseClient((url) => {
      switch (new URL(url).pathname) {
        case "/health":
          return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
        case "/v1/models":
          return new Response(JSON.stringify({
            object: "list",
            data: [{ id: modelId, object: "model", meta: { n_ctx: 8_192 } }],
          }), { status: 200 })
        case "/props":
          return new Response(JSON.stringify({
            default_generation_settings: { n_ctx: 8_192 },
            model_alias: modelId,
            model_path: "/models/private-build.gguf",
            modalities: { vision: false, audio: false },
          }), { status: 200 })
        default:
          return new Response("", { status: 404 })
      }
    })

    const result = await Effect.runPromise(
      createLlamaCppProvider().catalog.list.pipe(
        Effect.provideService(HttpClient.HttpClient, client),
      ),
    )

    expect(result).toEqual([
      expect.objectContaining({
        providerModelId: modelId,
        displayName: "custom.model:latest",
        modelFamilyId: "unknown",
      }),
    ])
  })

  it("formats local, Hugging Face, sharded, and metadata-backed names", () => {
    expect(deriveDisplayName({
      id: "/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf",
      object: "model",
      meta: { ftype: "Q6_K" },
    })).toBe("Qwen3.6-35B-A3B (UD-Q6_K_XL)")

    expect(deriveDisplayName({
      id: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M",
      object: "model",
    })).toBe("Qwen3.6-35B-A3B-MTP (UD-Q4_K_M)")

    expect(deriveDisplayName({
      id: "C:\\Models\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
      object: "model",
    })).toBe("Meta-Llama-3.1-8B-Instruct (Q4_K_M)")

    expect(deriveDisplayName({
      id: "/models/Kimi-K2-Thinking-UD-IQ1_S-00001-of-00006.gguf",
      object: "model",
    })).toBe("Kimi-K2-Thinking (UD-IQ1_S)")

    expect(deriveDisplayName({
      id: "/models/private-Q4_K_M.gguf",
      object: "model",
      meta: { "general.name": "Curated Model", ftype: "Q4_K" },
    })).toBe("Curated Model (Q4_K_M)")

    expect(deriveDisplayName({
      id: "acme/custom.model:v2",
      object: "model",
    })).toBe("custom.model:v2")

    expect(deriveDisplayName({
      id: "/models/private-build.gguf",
      aliases: ["team-friendly-name"],
      object: "model",
    })).toBe("team-friendly-name")
  })

  it("rejects malformed model-list envelopes as a discovery error", async () => {
    const client = responseClient(() => new Response(
      JSON.stringify({ object: "list", data: {} }),
      { status: 200 },
    ))

    const result = await Effect.runPromise(
      fetchModelList({ endpoint: "http://127.0.0.1:8080" }).pipe(
        Effect.either,
        Effect.provideService(HttpClient.HttpClient, client),
      ),
    )

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left.message).toContain("did not contain a data array")
    }
  })

  it("rejects non-empty model lists without valid model IDs", async () => {
    const client = responseClient(() => new Response(
      JSON.stringify({ object: "list", data: [{ object: "model" }] }),
      { status: 200 },
    ))

    const result = await Effect.runPromise(
      fetchModelList({ endpoint: "http://127.0.0.1:8080" }).pipe(
        Effect.either,
        Effect.provideService(HttpClient.HttpClient, client),
      ),
    )

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left.message).toContain("no valid model entries")
    }
  })

  it("uses router model arguments and modalities", () => {
    const raw = {
      id: "router-model",
      object: "model",
      status: { args: ["llama-server", "-m=/models/router-model.gguf"] },
      architecture: { input_modalities: ["text", "image"] },
    }

    expect(deriveSourceModelPath(raw, null, 2)).toBe("/models/router-model.gguf")
    expect(detectVision(raw, null)).toBe(true)
  })

  it("keeps an unavailable provider visible without failing its catalog", async () => {
    const client = HttpClient.make((request) =>
      Effect.fail(new HttpClientError.RequestError({
        request,
        reason: "Transport",
        cause: new Error("connection refused"),
      })),
    )
    const instance = createLlamaCppProvider()
    const registry = makeProviderRegistry({
      magnitude: null,
      discoverableProviders: [instance],
    })

    const [models, providers] = await Effect.runPromise(
      Effect.all([registry.aggregatedCatalog.list, registry.listProviders]).pipe(
        Effect.provideService(HttpClient.HttpClient, client),
      ),
    )

    expect(models).toEqual([])
    expect(providers).toEqual([{
      id: "llamacpp",
      displayName: "Llama.cpp",
      authStatus: { _tag: "no_auth_required" },
      status: "not_found",
      hint: "Start one with e.g. llama-server -m /path/to/model.gguf",
    }])
  })

  it("discovers local models when Magnitude authentication is not configured", async () => {
    const client = responseClient((url) => {
      const parsed = new URL(url)
      if (parsed.hostname !== "127.0.0.1") {
        return new Response("unauthorized", { status: 401 })
      }
      switch (parsed.pathname) {
        case "/health":
          return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
        case "/v1/models":
          return new Response(JSON.stringify({
            object: "list",
            data: [{ id: "private-build.gguf", object: "model", meta: { n_ctx: 8_192 } }],
          }), { status: 200 })
        case "/props":
          return new Response(JSON.stringify({
            default_generation_settings: { n_ctx: 8_192 },
            model_alias: "private-build.gguf",
            modalities: { vision: false, audio: false },
          }), { status: 200 })
        default:
          return new Response("", { status: 404 })
      }
    })
    const magnitude = createMagnitudeProvider({
      auth: () => {
        throw new Error("No Magnitude API key")
      },
    })
    const llamacpp = createLlamaCppProvider()
    const registry = makeProviderRegistry({
      magnitude,
      discoverableProviders: [llamacpp],
    })

    const models = await Effect.runPromise(
      registry.aggregatedCatalog.refresh.pipe(
        Effect.provideService(HttpClient.HttpClient, client),
      ),
    )

    expect(models).toEqual([
      expect.objectContaining({
        providerId: "llamacpp",
        providerModelId: "private-build.gguf",
      }),
    ])
  })

  it("falls back to versioned props and parses runtime metadata", async () => {
    const requested: string[] = []
    const client = responseClient((url) => {
      requested.push(url)
      if (new URL(url).pathname === "/props") return new Response("", { status: 404 })
      return new Response(JSON.stringify({
        default_generation_settings: { n_ctx: 16_384 },
        model_alias: "test-model",
        model_ftype: "Q4_K",
        model_path: "/models/test.gguf",
        chat_template: "chatml",
        modalities: { vision: true, audio: false },
      }), { status: 200 })
    })

    const result = await Effect.runPromise(
      fetchServerProps({ endpoint: "http://127.0.0.1:8080" }).pipe(
        Effect.provideService(HttpClient.HttpClient, client),
      ),
    )

    expect(requested).toEqual([
      "http://127.0.0.1:8080/props",
      "http://127.0.0.1:8080/v1/props",
    ])
    expect(result).toEqual({
      nCtx: 16_384,
      modelAlias: "test-model",
      modelFtype: "Q4_K",
      modelPath: "/models/test.gguf",
      chatTemplate: "chatml",
      modalities: { vision: true, audio: false },
    })
  })

  it("prefers runtime context and authoritative modality props", () => {
    const raw = {
      id: "gemma-3-vision.gguf",
      object: "model",
      meta: { n_ctx: 8_192, n_ctx_train: 32_768 },
    }

    expect(deriveContextWindow(raw, { nCtx: 4_096 })).toBe(4_096)
    expect(detectVision(raw, { modalities: { vision: false } })).toBe(false)
    expect(detectVision(raw, null)).toBe(true)
  })
})
