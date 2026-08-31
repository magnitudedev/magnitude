import { describe, expect, test } from "vitest"
import {
  LOCAL_ANTHROPIC_MODEL_PREFIX,
  MAX_ANTHROPIC_ROUTING_BODY_BYTES,
  proxyAnthropicInferenceRequest,
} from "./inference-gateway"

const icn = {
  origin: new URL("http://127.0.0.1:9999"),
  clientOptions: {
    headers: { authorization: "Bearer private-icn-token" },
  },
}

describe("Anthropic inference gateway", () => {
  test("removes only the anthropic-local prefix from canonical hf identities", async () => {
    const model = "hf:unsloth/Qwen-GGUF/quants/Qwen-Q4.gguf"
    let forwarded: Request | undefined
    await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/messages", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: `${LOCAL_ANTHROPIC_MODEL_PREFIX}${model}`,
          messages: [],
        }),
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        return Response.json({ ok: true })
      },
    )

    expect(forwarded?.headers.get("magnitude-gateway-model")).toBe(
      `anthropic-local/${model}`,
    )
    expect(await forwarded?.json()).toMatchObject({ model })
  })

  test("rewrites reserved local aliases and replaces caller credentials", async () => {
    let forwarded: Request | undefined
    const result = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/messages", {
        method: "POST",
        headers: {
          authorization: "Bearer caller-token",
          "x-api-key": "caller-key",
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
          "magnitude-gateway-model": "forged",
        },
        body: JSON.stringify({
          model: `${LOCAL_ANTHROPIC_MODEL_PREFIX}canonical:model`,
          max_tokens: 32,
          messages: [{ role: "user", content: "hello" }],
        }),
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        return Response.json({ ok: true })
      },
    )

    expect(result.status).toBe(200)
    expect(forwarded?.url).toBe("http://127.0.0.1:9999/anthropic/v1/messages")
    expect(forwarded?.headers.get("authorization")).toBe("Bearer private-icn-token")
    expect(forwarded?.headers.get("x-api-key")).toBeNull()
    expect(forwarded?.headers.get("magnitude-gateway-model")).toBe(
      "anthropic-local/canonical:model",
    )
    expect(await forwarded?.json()).toMatchObject({ model: "canonical:model" })
  })

  test("changes only the local model string and leaves unknown JSON opaque", async () => {
    const body = `{
  "unknown_future_field": { "model": "nested", "raw": [1, 2, 3] },
  "model" : "anthropic-local/canonical:model",
  "messages": []
}`
    let forwardedBody = ""
    await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/messages", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
      }),
      icn,
      async (input, init) => {
        forwardedBody = await new Request(input, init).text()
        return new Response("ok")
      },
    )

    expect(forwardedBody).toBe(body.replace(
      '"anthropic-local/canonical:model"',
      '"canonical:model"',
    ))
  })

  test("forwards non-reserved models and exact request bytes upstream", async () => {
    const body = '{ "model": "claude-sonnet-4-5", "max_tokens": 4, "messages": [] }'
    let forwarded: Request | undefined
    await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": "anthropic-key",
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
        },
        body,
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        return new Response("ok")
      },
    )

    expect(forwarded?.url).toBe("https://api.anthropic.com/v1/messages")
    expect(forwarded?.headers.get("x-api-key")).toBe("anthropic-key")
    expect(await forwarded?.text()).toBe(body)
  })

  test("preserves count-token bytes upstream and refuses the local operation", async () => {
    const upstreamBody = '{"model":"claude-upstream","future":true}'
    let forwarded: Request | undefined
    const upstream = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/messages/count_tokens?beta=1", {
        method: "POST",
        body: upstreamBody,
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        return new Response("upstream", { status: 202, headers: { "x-upstream": "yes" } })
      },
    )
    expect(forwarded?.url).toBe(
      "https://api.anthropic.com/v1/messages/count_tokens?beta=1",
    )
    expect(await forwarded?.text()).toBe(upstreamBody)
    expect(upstream.status).toBe(202)
    expect(upstream.headers.get("x-upstream")).toBe("yes")

    let localForwarded: Request | undefined
    const local = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/messages/count_tokens", {
        method: "POST",
        body: '{"model":"anthropic-local/local:model"}',
      }),
      icn,
      async (input, init) => {
        localForwarded = new Request(input, init)
        return Response.json({ input_tokens: 7 })
      },
    )
    expect(local.status).toBe(200)
    expect(localForwarded?.url).toBe(
      "http://127.0.0.1:9999/anthropic/v1/messages/count_tokens",
    )
    expect(await localForwarded?.json()).toMatchObject({ model: "local:model" })
    expect(await local.json()).toEqual({ input_tokens: 7 })
  })

  test("bounds classification before parsing and rejects ambiguous model keys", async () => {
    const oversized = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/messages", {
        method: "POST",
        headers: {
          "content-length": String(MAX_ANTHROPIC_ROUTING_BODY_BYTES + 1),
        },
        body: '{"model":"claude-upstream"}',
      }),
      icn,
      async () => new Response("unexpected"),
    )
    expect(oversized.status).toBe(413)
    expect(await oversized.json()).toMatchObject({
      type: "error",
      error: { type: "request_too_large" },
    })

    const actualOversizedBody = new Uint8Array(
      MAX_ANTHROPIC_ROUTING_BODY_BYTES + 1,
    ).buffer
    const actualOversized = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/messages", {
        method: "POST",
        body: actualOversizedBody,
      }),
      icn,
      async () => new Response("unexpected"),
    )
    expect(actualOversized.status).toBe(413)

    const duplicate = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/messages", {
        method: "POST",
        body: '{"model":"claude-one","model":"claude-two"}',
      }),
      icn,
      async () => new Response("unexpected"),
    )
    expect(duplicate.status).toBe(400)
  })

  test("projects ICN models into Claude-compatible local aliases", async () => {
    const response = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/models", {
        headers: { "anthropic-version": "2023-06-01" },
      }),
      icn,
      async () => Response.json({
        object: "list",
        models: [{
          id: "local:model",
          name: "Local Model",
          description: "A local fixture.",
          contextWindow: 32_768,
          capabilities: {
            vision: false,
            tools: true,
            structuredOutput: true,
            reasoning: {
              supported: true,
              efforts: ["none", "high"],
              defaultEffort: "high",
            },
          },
        }],
        data: [{
          id: "local:model",
          object: "model",
          created: 0,
          owned_by: "magnitude",
        }],
      }),
    )
    expect(await response.json()).toMatchObject({
      data: [{
        id: "anthropic-local/local:model",
        type: "model",
        display_name: "Local Model",
        description: "A local fixture.",
      }],
      has_more: false,
    })
  })

  test("paginates the local alias catalog without an upstream dependency", async () => {
    const catalog = {
      object: "list",
      models: [],
      data: ["c", "a", "b"].map((id) => ({
        id,
        object: "model",
        created: 0,
        owned_by: "magnitude",
      })),
    }
    const fetchCatalog = async () => Response.json(catalog)
    const first = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/models?limit=2"),
      icn,
      fetchCatalog,
    )
    expect(await first.json()).toMatchObject({
      data: [
        { id: "anthropic-local/a" },
        { id: "anthropic-local/b" },
      ],
      has_more: true,
      last_id: "anthropic-local/b",
    })
    const second = await proxyAnthropicInferenceRequest(
      new Request(
        "http://127.0.0.1:10100/inference/anthropic/v1/models?limit=2&after_id=anthropic-local%2Fb",
      ),
      icn,
      fetchCatalog,
    )
    expect(await second.json()).toMatchObject({
      data: [{ id: "anthropic-local/c" }],
      has_more: false,
    })
  })

  test("answers the optional warm-up probe without touching either backend", async () => {
    let called = false
    const response = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/api/hello", {
        method: "HEAD",
      }),
      icn,
      async () => {
        called = true
        return new Response("unexpected")
      },
    )
    expect(response.status).toBe(204)
    expect(called).toBe(false)
  })
})
