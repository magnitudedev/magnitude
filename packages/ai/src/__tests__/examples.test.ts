import { describe, it, expect } from "vitest"
import { Effect, Layer, Stream } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import {
  Model,
  NativeChatCompletions,
  Auth,
  Option,
  Prompt,
} from "../index"
import type {
  ToolDefinition,
} from "../index"

const emptyPrompt = Prompt.from({ system: "You are helpful.", messages: [{ _tag: "UserMessage" as const, parts: [{ _tag: "TextPart" as const, text: "hello" }] }] })
const noTools: readonly ToolDefinition[] = []

// ---------------------------------------------------------------------------
// 1. Fireworks MiniMax with reasoning effort
// ---------------------------------------------------------------------------
describe("Fireworks MiniMax with reasoning effort", () => {
  const minimaxM27 = NativeChatCompletions.model({
    modelId: "accounts/fireworks/models/minimax-m2.7",
    endpoint: "https://api.fireworks.ai/inference/v1",
    options: {
      ...NativeChatCompletions.options,
      reasoningEffort: Option.define(
        (val: "none" | "low" | "medium") => ({ reasoning_effort: val }),
      ),
    },
  })

  it("creates a valid model spec", () => {
    expect(minimaxM27.modelId).toBe("accounts/fireworks/models/minimax-m2.7")
  })

  it("binds with auth and accepts typed options", () => {
    const model = minimaxM27.bind({ auth: Auth.bearer("test-key") })
    const effect = model.stream(emptyPrompt, noTools, {
      maxTokens: 4000,
      reasoningEffort: "low",
    })
    expect(effect).toBeDefined()
  })

  it("accepts no options (all optional)", () => {
    const model = minimaxM27.bind({ auth: Auth.bearer("test-key") })
    const effect = model.stream(emptyPrompt, noTools)
    expect(effect).toBeDefined()
  })
})

// ---------------------------------------------------------------------------
// 2. Magnitude gateway — Kimi K2.6 with grammar
// ---------------------------------------------------------------------------
describe("Magnitude gateway — Kimi K2.6 with grammar", () => {
  const kimiK26 = NativeChatCompletions.model({
    modelId: "kimi-k2.6",
    endpoint: "https://app.magnitude.dev/api/v1",
    options: {
      ...NativeChatCompletions.options,
      grammar: Option.define(
        (g: string) => ({ response_format: { type: "grammar", grammar: g } }),
      ),
    },
  })

  it("accepts grammar option", () => {
    const model = kimiK26.bind({ auth: Auth.bearer("test-key") })
    const effect = model.stream(emptyPrompt, noTools, {
      maxTokens: 8000,
      grammar: "<some-grammar>",
    })
    expect(effect).toBeDefined()
  })
})

// ---------------------------------------------------------------------------
// 3. DeepSeek — thinking toggle with compose
// ---------------------------------------------------------------------------
describe("DeepSeek — thinking toggle with compose", () => {
  type DeepSeekThinking = { type: "enabled" } | { type: "disabled" }

  const deepseekV4 = NativeChatCompletions.model({
    modelId: "deepseek-v4-pro",
    endpoint: "https://api.deepseek.com/v1",
    options: {
      ...NativeChatCompletions.options,
      thinking: Option.define(
        (val: DeepSeekThinking) => ({ thinking: val }),
      ),
    },
    compose: (wire, callOpts) => {
      if (callOpts.thinking?.type === "enabled") {
        return { ...wire, temperature: undefined }
      }
      return wire
    },
  })

  it("creates spec with compose", () => {
    expect(deepseekV4.modelId).toBe("deepseek-v4-pro")
  })

  it("accepts thinking option", () => {
    const model = deepseekV4.bind({ auth: Auth.bearer("test-key") })
    const effect = model.stream(emptyPrompt, noTools, {
      thinking: { type: "enabled" },
    })
    expect(effect).toBeDefined()
  })
})

// ---------------------------------------------------------------------------
// 4. Required option
// ---------------------------------------------------------------------------
describe("Required option", () => {
  const deepseekReasoner = NativeChatCompletions.model({
    modelId: "deepseek-reasoner",
    endpoint: "https://api.deepseek.com/v1",
    options: {
      ...NativeChatCompletions.options,
      thinking: Option.required(
        (val: { type: "enabled" }) => ({ thinking: val }),
      ),
    },
  })

  it("requires thinking option — omitting it is a type error", () => {
    const model = deepseekReasoner.bind({ auth: Auth.bearer("test-key") })

    // @ts-expect-error — thinking is required but not provided
    const _ignored = model.stream(emptyPrompt, noTools, { maxTokens: 4000 })

    // This should compile: thinking is provided
    const effect = model.stream(emptyPrompt, noTools, {
      maxTokens: 4000,
      thinking: { type: "enabled" },
    })
    expect(effect).toBeDefined()
  })
})

// ---------------------------------------------------------------------------
// 5. Testing with mock HttpClient
// ---------------------------------------------------------------------------
describe("Testing with mock HttpClient", () => {
  const spec = NativeChatCompletions.model({
    modelId: "test-model",
    endpoint: "http://localhost:8080",
    options: { ...NativeChatCompletions.options },
  })

  it("demonstrates layer substitution pattern", () => {
    const ssePayload = [
      'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"test","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
      "",
      "data: [DONE]",
      "",
    ].join("\n")

    const mockClient = HttpClient.make((req) =>
      Effect.succeed(
        HttpClientResponse.fromWeb(
          req,
          new Response(ssePayload, {
            status: 200,
            headers: { "content-type": "text/event-stream" },
          }),
        ),
      ),
    )
    const MockLayer = Layer.succeed(HttpClient.HttpClient, mockClient)

    const model = spec.bind({ auth: Auth.bearer("test-token") })

    const program = Effect.gen(function* () {
      const result = yield* model.stream(emptyPrompt, noTools, {
        maxTokens: 100,
      })
      const events = yield* Stream.runCollect(result.events)
      return events
    })

    const runnable = Effect.provide(program, MockLayer)
    expect(runnable).toBeDefined()
  })
})
