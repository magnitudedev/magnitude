import { describe, expect, test } from "vitest"
import {
  LOCAL_CODEX_MODEL_PREFIX,
  LOCAL_ANTHROPIC_MODEL_PREFIX,
  MAX_CODEX_ROUTING_BODY_BYTES,
  MAX_ANTHROPIC_ROUTING_BODY_BYTES,
  codexWebSocketTarget,
  proxyAnthropicInferenceRequest,
  proxyCodexInferenceRequest,
  proxyLocalAnthropicInferenceRequest,
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
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/messages", {
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
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/messages", {
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
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/messages", {
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
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/messages", {
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
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/messages/count_tokens?beta=1", {
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
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/messages/count_tokens", {
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
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/messages", {
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
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/messages", {
        method: "POST",
        body: actualOversizedBody,
      }),
      icn,
      async () => new Response("unexpected"),
    )
    expect(actualOversized.status).toBe(413)

    const duplicate = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/messages", {
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
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/models", {
        headers: { "anthropic-version": "2023-06-01" },
      }),
      icn,
      async () => Response.json({
        object: "list",
        data: [{
          id: "local:model",
          object: "model",
          created: 0,
          owned_by: "magnitude",
          name: "Local Model",
          description: "A local fixture.",
          context_length: 32_768,
          architecture: { input_modalities: ["text"], output_modalities: ["text"] },
          supported_parameters: ["max_tokens", "tools", "tool_choice", "reasoning"],
          reasoning: {
            supported_efforts: ["none", "high"],
            default_effort: "high",
            default_enabled: true,
            mandatory: false,
          },
          top_provider: { context_length: 32_768, max_completion_tokens: 32_768 },
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
      data: ["c", "a", "b"].map((id) => ({
        id,
        object: "model",
        created: 0,
        owned_by: "magnitude",
        name: id,
        description: `Local ${id}.`,
        context_length: 32_768,
        architecture: { input_modalities: ["text"], output_modalities: ["text"] },
        supported_parameters: ["max_tokens"],
        top_provider: { context_length: 32_768, max_completion_tokens: 32_768 },
      })),
    }
    const fetchCatalog = async () => Response.json(catalog)
    const first = await proxyAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/models?limit=2"),
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
        "http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/v1/models?limit=2&after_id=anthropic-local%2Fb",
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
      new Request("http://127.0.0.1:10100/inference/anthropic/proxies/claude-code/api/hello", {
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

describe("Codex inference gateway", () => {
  test("forwards OpenAI request bytes and ChatGPT authentication unchanged", async () => {
    const body = '{ "model": "gpt-5.6-sol", "future": { "unknown": true } }'
    let forwarded: Request | undefined
    let decompress: boolean | undefined
    const response = await proxyCodexInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses?beta=true", {
        method: "POST",
        headers: {
          authorization: "Bearer chatgpt-token",
          "chatgpt-account-id": "account-1",
          "content-type": "application/json",
        },
        body,
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        decompress = init?.decompress
        return new Response("upstream", { status: 202 })
      },
    )

    expect(forwarded?.url).toBe("https://chatgpt.com/backend-api/codex/responses?beta=true")
    expect(forwarded?.headers.get("authorization")).toBe("Bearer chatgpt-token")
    expect(await forwarded?.text()).toBe(body)
    expect(decompress).toBe(false)
    expect(response.status).toBe(202)
  })

  test("routes API-key requests to the public OpenAI v1 origin", async () => {
    let forwarded = ""
    await proxyCodexInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses", {
        method: "POST",
        headers: { authorization: "Bearer sk-test" },
        body: '{"model":"gpt-5.6-sol"}',
      }),
      icn,
      async (input) => {
        forwarded = String(input)
        return new Response("ok")
      },
    )
    expect(forwarded).toBe("https://api.openai.com/v1/responses")
  })

  test("removes ICN-owned response references before an upstream HTTP turn", async () => {
    let forwarded: Request | undefined
    await proxyCodexInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses", {
        method: "POST",
        headers: { authorization: "Bearer sk-test", "content-type": "application/json" },
        body: JSON.stringify({
          model: "gpt-5.6-sol",
          previous_response_id: "resp_icn_1",
          input: [
            { type: "reasoning", id: "rs_icn_2", summary: [] },
            { type: "message", id: "msg_icn_2", role: "assistant", content: "local" },
          ],
        }),
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        return new Response("ok")
      },
    )
    expect(await forwarded?.json()).toEqual({
      model: "gpt-5.6-sol",
      input: [{ type: "message", role: "assistant", content: "local" }],
    })
    expect(forwarded?.headers.get("authorization")).toBe("Bearer sk-test")
  })

  test("rewrites only a local model alias and isolates credentials", async () => {
    const body = `{
  "unknown": { "model": "nested", "value": [1, 2, 3] },
  "model" : "${LOCAL_CODEX_MODEL_PREFIX}canonical:model",
  "input": []
}`
    let forwarded: Request | undefined
    await proxyCodexInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses", {
        method: "POST",
        headers: {
          authorization: "Bearer openai-secret",
          "x-api-key": "caller-secret",
          "content-type": "application/json",
        },
        body,
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        return new Response("local")
      },
    )

    expect(forwarded?.url).toBe("http://127.0.0.1:9999/v1/responses")
    expect(forwarded?.headers.get("authorization")).toBe("Bearer private-icn-token")
    expect(forwarded?.headers.get("x-api-key")).toBeNull()
    expect(await forwarded?.text()).toBe(body.replace(
      `"${LOCAL_CODEX_MODEL_PREFIX}canonical:model"`,
      '"canonical:model"',
    ))
  })

  test("classifies zstd without changing upstream compressed bytes", async () => {
    const decoded = Buffer.from('{"model":"gpt-5.6-sol","input":[]}')
    const compressed = Bun.zstdCompressSync(decoded)
    let forwarded: Request | undefined
    await proxyCodexInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses", {
        method: "POST",
        headers: { "content-encoding": "zstd" },
        body: Uint8Array.from(compressed).buffer,
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        return new Response("ok")
      },
    )

    expect(forwarded?.headers.get("content-encoding")).toBe("zstd")
    expect(Buffer.from(await forwarded!.arrayBuffer())).toEqual(compressed)
  })

  test("decodes local zstd content and removes stale encoding metadata", async () => {
    const decoded = Buffer.from(`{"model":"${LOCAL_CODEX_MODEL_PREFIX}canonical:model","input":[]}`)
    const compressed = Bun.zstdCompressSync(decoded)
    let forwarded: Request | undefined
    await proxyCodexInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses", {
        method: "POST",
        headers: { "content-encoding": "zstd" },
        body: Uint8Array.from(compressed).buffer,
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        return new Response("ok")
      },
    )

    expect(forwarded?.headers.get("content-encoding")).toBeNull()
    expect(await forwarded?.text()).toBe('{"model":"canonical:model","input":[]}')
  })

  test("rejects unsupported encodings and oversized routing bodies", async () => {
    const unsupported = await proxyCodexInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses", {
        method: "POST",
        headers: { "content-encoding": "br" },
        body: '{"model":"gpt-5.6-sol"}',
      }),
      icn,
      async () => new Response("unexpected"),
    )
    expect(unsupported.status).toBe(415)

    const oversized = await proxyCodexInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses", {
        method: "POST",
        headers: { "content-length": String(MAX_CODEX_ROUTING_BODY_BYTES + 1) },
        body: '{"model":"gpt-5.6-sol"}',
      }),
      icn,
      async () => new Response("unexpected"),
    )
    expect(oversized.status).toBe(413)
  })

  test("rejects malformed, ambiguous, and empty local model selectors", async () => {
    for (const body of [
      '{"input":[]}',
      '{"model":42}',
      '{"model":"first","model":"second"}',
      '{"model":',
      `{"model":"${LOCAL_CODEX_MODEL_PREFIX}"}`,
    ]) {
      let forwarded = false
      const response = await proxyCodexInferenceRequest(
        new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses", {
          method: "POST",
          body,
        }),
        icn,
        async () => {
          forwarded = true
          return new Response("unexpected")
        },
      )
      expect(response.status, body).toBe(400)
      expect(forwarded, body).toBe(false)
    }
  })

  test("passes non-routed Codex operations through without parsing their bodies", async () => {
    const body = new Uint8Array([0, 1, 2, 3])
    let forwarded: Request | undefined
    await proxyCodexInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses/compact?mode=remote", {
        method: "POST",
        headers: { authorization: "Bearer token", "content-type": "application/octet-stream" },
        body,
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        return new Response("ok")
      },
    )
    expect(forwarded?.url).toBe("https://api.openai.com/v1/responses/compact?mode=remote")
    expect(forwarded?.headers.get("authorization")).toBe("Bearer token")
    expect(new Uint8Array(await forwarded!.arrayBuffer())).toEqual(body)
  })

  test("preserves encoded upstream response bytes and headers", async () => {
    const encoded = new Uint8Array([0xce, 0xb2, 0xcf, 0x81])
    const response = await proxyCodexInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/v1/proxies/codex/responses", {
        method: "POST",
        body: '{"model":"gpt-5.6-sol"}',
      }),
      icn,
      async (_input, init) => {
        expect(init?.decompress).toBe(false)
        return new Response(encoded, { headers: { "content-encoding": "br" } })
      },
    )
    expect(response.headers.get("content-encoding")).toBe("br")
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(encoded)
  })
})

describe("Codex WebSocket gateway", () => {
  test("routes OpenAI models upstream without changing the first frame", () => {
    const firstMessage = '{"type":"response.create","model":"gpt-5.6-sol","input":[]}'
    const target = codexWebSocketTarget(
      firstMessage,
      new Headers({ authorization: "Bearer openai-token" }),
      icn,
    )
    expect(target._tag).toBe("Target")
    if (target._tag !== "Target") return
    expect(target.url.href).toBe("wss://api.openai.com/v1/responses")
    expect(target.headers.get("authorization")).toBe("Bearer openai-token")
    expect(target.firstMessage).toBe(firstMessage)
  })

  test("inlines local message history and omits non-portable local reasoning upstream", () => {
    const target = codexWebSocketTarget(
      JSON.stringify({
        type: "response.create",
        model: "gpt-5.6-luna",
        previous_response_id: "resp_icn_1",
        input: [
          { type: "reasoning", id: "rs_icn_2", summary: [], encrypted_content: null },
          { type: "message", id: "msg_icn_2", role: "assistant", content: [{ type: "output_text", text: "local" }] },
          { type: "message", id: "msg_user", role: "user", content: [{ type: "input_text", text: "next" }] },
        ],
      }),
      new Headers({ authorization: "Bearer openai-token" }),
      icn,
    )
    expect(target._tag).toBe("Target")
    if (target._tag !== "Target") return
    const frame = JSON.parse(new TextDecoder().decode(target.firstMessage as Uint8Array))
    expect(frame.previous_response_id).toBeUndefined()
    expect(frame.input).toEqual([
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "local" }] },
      { type: "message", id: "msg_user", role: "user", content: [{ type: "input_text", text: "next" }] },
    ])
  })

  test("routes local models to ICN and changes only the model selector", () => {
    const firstMessage = `{
  "type": "response.create",
  "future": {"untouched": true},
  "model" : "${LOCAL_CODEX_MODEL_PREFIX}canonical:model",
  "input": []
}`
    const target = codexWebSocketTarget(
      firstMessage,
      new Headers({ authorization: "Bearer caller-token" }),
      icn,
    )
    expect(target._tag).toBe("Target")
    if (target._tag !== "Target") return
    expect(target.url.href).toBe("ws://127.0.0.1:9999/v1/responses")
    expect(target.headers.get("authorization")).toBe("Bearer private-icn-token")
    expect(new TextDecoder().decode(target.firstMessage as Uint8Array)).toBe(
      firstMessage.replace(`${LOCAL_CODEX_MODEL_PREFIX}canonical:model`, "canonical:model"),
    )
  })

  test("uses the ChatGPT Codex WebSocket origin and rejects an invalid first frame", () => {
    const upstream = codexWebSocketTarget(
      '{"type":"response.create","model":"gpt-5.6-luna","input":[]}',
      new Headers({ "chatgpt-account-id": "account" }),
      icn,
    )
    expect(upstream._tag).toBe("Target")
    if (upstream._tag === "Target") {
      expect(upstream.url.href).toBe("wss://chatgpt.com/backend-api/codex/responses")
    }
    expect(codexWebSocketTarget("{}", new Headers(), icn)).toMatchObject({
      _tag: "Invalid",
    })
  })
})

describe("generic Anthropic inference proxy", () => {
  test("keeps the ordinary Anthropic endpoint separate from Claude Code discovery", async () => {
    let forwarded: Request | undefined
    const response = await proxyLocalAnthropicInferenceRequest(
      new Request("http://127.0.0.1:10100/inference/anthropic/v1/models?source=generic", {
        headers: { authorization: "Bearer caller" },
      }),
      icn,
      async (input, init) => {
        forwarded = new Request(input, init)
        return new Response('{"generic":true}', { headers: { "content-type": "application/json" } })
      },
    )
    expect(forwarded?.url).toBe("http://127.0.0.1:9999/anthropic/v1/models?source=generic")
    expect(forwarded?.headers.get("authorization")).toBe("Bearer private-icn-token")
    expect(await response.json()).toEqual({ generic: true })
  })
})
