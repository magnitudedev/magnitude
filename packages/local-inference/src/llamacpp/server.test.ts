import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import { afterAll, beforeAll, describe, expect, it } from "vitest"
import { Effect, Option } from "effect"
import { LlamaInstanceId, makeLlamaServerClient } from "."

let loading = true
let server: ReturnType<typeof Bun.serve>

beforeAll(() => {
  server = Bun.serve({
    port: 0,
    fetch: (request) => {
      const url = new URL(request.url)
      if (url.pathname === "/health") return Response.json({ status: "ok" }, { status: loading ? 503 : 200 })
      if (url.pathname === "/models") return Response.json({
        data: [{
          id: "qwen-local",
          status: { value: "loaded" },
          meta: null,
          "general.name": "Qwen structured name",
          "general.architecture": "qwen2",
          ftype: "Q5_K_M",
          size: 12_345,
          architecture: { input_modalities: ["text"], output_modalities: ["text"] },
        }],
      })
      if (url.pathname === "/props") return Response.json({ model_path: "/models/qwen.gguf", model_ftype: "Q5_K_M", default_generation_settings: { n_ctx: 65_536 } })
      return new Response(null, { status: 404 })
    },
  })
})

afterAll(() => server.stop(true))

const client = () => makeLlamaServerClient({
  origin: new URL(`http://127.0.0.1:${server.port}`),
  authorization: Option.none(),
  timeout: Option.some("2 seconds" as const),
}).pipe(Effect.provide(FetchHttpClient.layer))

describe("llama.cpp server protocol", () => {
  it("treats a 503 health response as loading", async () => {
    const value = await Effect.runPromise(client().pipe(Effect.flatMap(({ observer }) => observer.health)))
    expect(value).toBe("loading")
  })

  it("normalizes current router data while preserving server-reported fields", async () => {
    loading = false
    const observation = await Effect.runPromise(client().pipe(Effect.flatMap(({ observer }) => observer.observe(LlamaInstanceId.make("test-router"), "external"))))
    const model = observation.models[0]

    expect(observation.mode).toBe("router")
    expect(model?.id).toBe("qwen-local")
    expect(Option.getOrNull(model?.serverDisplayName ?? Option.none())).toBe("Qwen structured name")
    expect(Option.getOrNull(model?.serverFileType ?? Option.none())).toBe("Q5_K_M")
    expect(Option.getOrNull(model?.serverReportedSizeBytes ?? Option.none())).toBe(12_345)
    expect(Option.getOrNull(model?.activeContextTokens ?? Option.none())).toBe(65_536)
    expect(Option.getOrNull(model?.architecture ?? Option.none())).toBe("qwen2")
    expect(Option.getOrNull(model?.inputModalities ?? Option.none())).toEqual(["text"])
  })
})
