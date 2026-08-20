import { afterAll, describe, expect, it } from "vitest"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import type { PlannedRequest } from "../src/domain"
import { EndpointClientLive, executeRequest, parseTerminalEvidence, validateToolCalls } from "../src/transport"
import { startMemoryProbe } from "../src/memory"
import { evaluate } from "../src/benchmark"
import { TargetLauncherLive } from "../src/target"
import { digestObject } from "../src/hash"
import type { TrialPlan } from "../src/domain"

const receivedBodies: Record<string, unknown>[] = []

const server = Bun.serve({
  port: 0,
  routes: {
    "/health": () => Response.json({ status: "ok" }),
    "/v1/chat/completions": async (request) => {
      const body = await request.json() as Record<string, unknown> & { model: string }
      receivedBodies.push(body)
      const encoder = new TextEncoder()
      const chunks = [
        { id: "1", object: "chat.completion.chunk", created: 0, model: body.model, choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call_", type: "function", function: { name: "lookup", arguments: "{\"key\":" } }] }, finish_reason: null }] },
        { id: "1", object: "chat.completion.chunk", created: 0, model: body.model, choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "1", function: { name: "", arguments: "\"value\"}" } }] }, finish_reason: "tool_calls" }] },
        {
          id: "1",
          object: "chat.completion.chunk",
          created: 0,
          model: body.model,
          choices: [],
          usage: { prompt_tokens: 20, completion_tokens: 6, total_tokens: 26, prompt_tokens_details: { cached_tokens: 4 } },
          timings: { cache_n: 4, prompt_n: 16, prompt_ms: 8, predicted_n: 6, predicted_ms: 12 },
        },
      ]
      return new Response(new ReadableStream({
        async start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
            await Bun.sleep(2)
          }
          controller.enqueue(encoder.encode("data: [DONE]\n\n"))
          controller.close()
        },
      }), { headers: { "content-type": "text/event-stream" } })
    },
  },
})

afterAll(() => server.stop(true))

const planned: PlannedRequest = {
  id: "request-1",
  fixtureId: "fixture-1",
  messages: [{ role: "user", content: "Look up value" }],
  tools: [{ type: "function", function: { name: "lookup", description: "lookup", parameters: { type: "object", properties: { key: { type: "string" } }, required: ["key"] } } }],
  expected: [{ name: "lookup", arguments: { key: ["value"] } }],
  releaseOffsetMs: 0,
  dependsOn: [],
  maxOutputTokens: 32,
}

const RuntimeLive = TargetLauncherLive.pipe(
  Layer.provideMerge(EndpointClientLive),
  Layer.provideMerge(FetchHttpClient.layer),
  Layer.provideMerge(BunContext.layer),
)

describe("endpoint and memory observation", () => {
  it("assembles streamed tool calls and validates BFCL allowed arguments", async () => {
    const observation = await Effect.runPromise(
      executeRequest({ endpoint: server.url.toString(), servedModel: "test" }, planned).pipe(
        Effect.provide(EndpointClientLive),
      ),
    )
    expect(observation.outcome).toBe("valid")
    expect(observation.toolCalls).toEqual([{ id: "call_1", name: "lookup", arguments: "{\"key\":\"value\"}" }])
    expect(observation.ttftMs).toBeGreaterThanOrEqual(0)
    expect(observation.terminal?.usage.promptTokens).toBe(20)
    expect(observation.terminal?.usage.completionTokens).toBe(6)
  })

  it("preserves JSON request extensions through finalization", async () => {
    receivedBodies.length = 0
    await Effect.runPromise(
      executeRequest({
        endpoint: server.url.toString(),
        servedModel: "test",
        requestBody: { provider_extension: { enabled: true } },
      }, planned).pipe(Effect.provide(EndpointClientLive)),
    )

    expect(receivedBodies.at(-1)).toMatchObject({
      provider_extension: { enabled: true },
    })
  })

  it("rejects request extensions that own standard fields", async () => {
    receivedBodies.length = 0
    const observation = await Effect.runPromise(
      executeRequest({
        endpoint: server.url.toString(),
        servedModel: "test",
        requestBody: { tool_choice: "none" },
      }, planned).pipe(Effect.provide(EndpointClientLive)),
    )

    expect(observation.outcome).toBe("error")
    expect(receivedBodies).toHaveLength(0)
  })

  it("rejects malformed request messages before fetch", async () => {
    receivedBodies.length = 0
    const observation = await Effect.runPromise(
      executeRequest(
        { endpoint: server.url.toString(), servedModel: "test" },
        {
          ...planned,
          messages: [{ role: "assistant", content: null }] as never,
        },
      ).pipe(Effect.provide(EndpointClientLive)),
    )

    expect(observation.outcome).toBe("error")
    expect(receivedBodies).toHaveLength(0)
  })

  it("rejects semantically incorrect arguments", () => {
    expect(validateToolCalls(planned.expected, [{ id: "x", name: "lookup", arguments: "{\"key\":\"wrong\"}" }])).toContain("allowed values")
  })

  it("rejects missing or inconsistent native evidence", () => {
    expect(() => parseTerminalEvidence({
      choices: [],
      usage: { prompt_tokens: 20, completion_tokens: 6, total_tokens: 26, prompt_tokens_details: { cached_tokens: 4 } },
      timings: { cache_n: 4, prompt_n: 15, prompt_ms: 8, predicted_n: 6, predicted_ms: 12 },
    })).toThrow("prompt_tokens does not equal")
    expect(() => parseTerminalEvidence({
      choices: [],
      usage: { prompt_tokens: 20, completion_tokens: 6, total_tokens: 26, prompt_tokens_details: { cached_tokens: 4 } },
    })).toThrow('"timings"]')
  })

  it("samples the live managed process without restarting it", async () => {
    const probe = await Effect.runPromise(startMemoryProbe({ rootPid: process.pid, intervalMs: 20 }))
    const allocation = new Uint8Array(2 * 1024 * 1024)
    allocation.fill(1)
    await Bun.sleep(50)
    const result = await Effect.runPromise(probe.stop)
    expect(result.supported).toBe(true)
    expect(result.samples.length).toBeGreaterThanOrEqual(2)
    expect(result.peakBytes).toBeGreaterThan(0)
  })

  it("evaluates an existing endpoint through the library", async () => {
    const identity = {
      profile: "smoke" as const,
      model: { id: "test", artifactPath: "/models/test.gguf", artifactSha256: "model", chatTemplateDigest: "template", contextLimit: 4096 },
      corpusDigest: "corpus",
      servingPolicy: { contextTokensPerSequence: 4096, parallelSequences: 1 },
      warmup: { ...planned, id: "warmup" },
      trials: [{
        id: "trial-1",
        pattern: "single-request" as const,
        criteria: ["responsiveness", "prefill", "decode", "memory-usage", "distribution"] as const,
        checkpoint: "small",
        repetition: 0,
        state: "cache-disjoint" as const,
        requests: [planned],
      }],
    }
    const plan: TrialPlan = { ...identity, createdAt: new Date(0).toISOString(), digest: digestObject(identity) }
    const result = await Effect.runPromise(evaluate(plan, {
      kind: "existing",
      id: "mock",
      endpoint: server.url.toString(),
      servedModel: "test",
      apiKey: "do-not-record",
      parallelSequences: 1,
    }).pipe(Effect.provide(RuntimeLive)))
    expect(result.planDigest).toBe(plan.digest)
    expect(result.trials[0]?.requests[0]?.outcome).toBe("valid")
    expect(result.target.apiKey).toBe("[redacted]")
    expect(result.trials[0]?.memory).toBeUndefined()
  })
})
