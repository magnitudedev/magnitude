import { Effect, Option as EffectOption, Schema, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { describe, expect, it } from "vitest"
import { Auth } from "../auth/auth"
import type { Codec } from "../codec/codec"
import { Model } from "../model/define"
import { Option as CallOption } from "../options/option"
import { Prompt } from "../prompt/prompt"
import { defineTool } from "../tools/tool-definition"
import { encodeChatCompletionsRequest } from "../wire/chat-completions"
import { NativeChatCompletions } from "./native-chat-completions"

const prompt = Prompt.from({
  messages: [{
    _tag: "UserMessage",
    parts: [{ _tag: "TextPart", text: "Hello" }],
  }],
})

describe("NativeChatCompletions request construction", () => {
  it("emits only required protocol fields for a minimal request", async () => {
    const request = await Effect.runPromise(NativeChatCompletions.buildRequest(
      {
        call: {
          provider: "test-provider",
          model: "test-model",
          method: "POST",
          url: "https://provider.example/chat/completions",
        },
        modelId: "test-model",
        options: {},
      },
      prompt,
      [],
      {},
    ).pipe(Effect.flatMap(encodeChatCompletionsRequest)))

    expect(request).toEqual({
      model: "test-model",
      messages: [{ role: "user", content: "Hello" }],
      stream: true,
      stream_options: { include_usage: true },
    })
  })

  it("preserves omission of tool choice when tools are present", async () => {
    const tool = defineTool({
      name: "lookup",
      description: "Look up a value",
      inputSchema: Schema.Struct({ query: Schema.String }),
      outputSchema: Schema.String,
    })
    const request = await Effect.runPromise(NativeChatCompletions.buildRequest(
      {
        call: {
          provider: "test-provider",
          model: "test-model",
          method: "POST",
          url: "https://provider.example/chat/completions",
        },
        modelId: "test-model",
        options: {},
      },
      prompt,
      [tool],
      {},
    ))

    expect(EffectOption.isNone(request.tool_choice)).toBe(true)
  })

  it("reports an option mapper throw through the typed failure channel", async () => {
    const spec = NativeChatCompletions.model({
      modelId: "test-model",
      endpoint: "http://127.0.0.1:1",
      options: {
        broken: CallOption.define(Schema.Struct({ broken: Schema.Boolean }), (_value: boolean) => {
          throw new Error("mapper exploded")
        }),
      },
    })

    const failure = await Effect.runPromise(spec.bind({ auth: Auth.none }).stream(
      prompt,
      [],
      { broken: true },
    ).pipe(
      Effect.flip,
      Effect.provide(FetchHttpClient.layer),
    ))

    expect(failure._tag).toBe("StreamStartClientCorrectnessViolation")
    if (failure._tag !== "StreamStartClientCorrectnessViolation") return
    expect(failure.component).toBe("request_builder")
    expect(failure.evidence._tag).toBe("UnexpectedDefectCaught")
  })

  it("rejects duplicate properties contributed by separate options", async () => {
    const spec = NativeChatCompletions.model({
      modelId: "test-model",
      endpoint: "http://127.0.0.1:1",
      options: {
        first: CallOption.field("temperature", Schema.Number),
        second: CallOption.field("temperature", Schema.Number),
      },
    })

    const failure = await Effect.runPromise(spec.bind({ auth: Auth.none }).stream(
      prompt,
      [],
      { first: 0.1, second: 0.2 },
    ).pipe(
      Effect.flip,
      Effect.provide(FetchHttpClient.layer),
    ))

    expect(failure._tag).toBe("StreamStartClientCorrectnessViolation")
    if (failure._tag !== "StreamStartClientCorrectnessViolation") return
    expect(failure.evidence).toEqual({
      _tag: "RequestContributionCollision",
      property: "temperature",
    })
  })

  it("rejects undefined produced by an option mapper before transport", async () => {
    const spec = NativeChatCompletions.model({
      modelId: "test-model",
      endpoint: "http://127.0.0.1:1",
      options: {
        invalid: CallOption.define(
          Schema.Struct({ provider_extension: Schema.String }),
          (_value: boolean) => ({ provider_extension: undefined as never }),
        ),
      },
    })

    const failure = await Effect.runPromise(spec.bind({ auth: Auth.none }).stream(
      prompt,
      [],
      { invalid: true },
    ).pipe(
      Effect.flip,
      Effect.provide(FetchHttpClient.layer),
    ))

    expect(failure._tag).toBe("StreamStartClientCorrectnessViolation")
    if (failure._tag !== "StreamStartClientCorrectnessViolation") return
    expect(failure.evidence._tag).toBe("RequestSchemaValidationFailed")
  })

  it("classifies a synchronous compose throw as a typed request failure", async () => {
    const spec = NativeChatCompletions.model({
      modelId: "test-model",
      endpoint: "http://127.0.0.1:1",
      options: {},
      compose: () => {
        throw new Error("compose exploded")
      },
    })

    const failure = await Effect.runPromise(spec.bind({ auth: Auth.none }).stream(
      prompt,
      [],
      {},
    ).pipe(
      Effect.flip,
      Effect.provide(FetchHttpClient.layer),
    ))

    expect(failure._tag).toBe("StreamStartClientCorrectnessViolation")
    if (failure._tag !== "StreamStartClientCorrectnessViolation") return
    expect(failure.component).toBe("request_builder")
    expect(failure.evidence._tag).toBe("UnexpectedDefectCaught")
  })

  it("preserves compose and validates its result before transport", async () => {
    let requestBody: unknown
    const server = Bun.serve({
      port: 0,
      fetch: async (request) => {
        requestBody = await request.json()
        return new Response("data: [DONE]\n\n", {
          headers: { "content-type": "text/event-stream" },
        })
      },
    })

    try {
      const spec = NativeChatCompletions.model({
        modelId: "test-model",
        endpoint: `http://127.0.0.1:${server.port}`,
        options: { temperature: NativeChatCompletions.options.temperature },
        compose: (request) => Effect.succeed({
          ...request,
          temperature: EffectOption.none(),
          extensions: { ...request.extensions, provider_extension: "enabled" },
        }),
      })

      await Effect.runPromise(spec.bind({ auth: Auth.none }).stream(prompt, [], { temperature: 0.5 }).pipe(
        Effect.provide(FetchHttpClient.layer),
      ))

      expect(requestBody).toMatchObject({
        model: "test-model",
        provider_extension: "enabled",
      })
      expect(requestBody).not.toHaveProperty("temperature")
    } finally {
      server.stop(true)
    }
  })
})

describe("Model.define", () => {
  it("remains usable by a non-Chat-Completions protocol", () => {
    const CustomRequestSchema = Schema.Struct({ input: Schema.String })
    type CustomRequest = Schema.Schema.Encoded<typeof CustomRequestSchema>
    interface CustomChunk {
      readonly output: string
    }

    const codec = {
      id: "custom-protocol",
      promptSchema: CustomRequestSchema,
      encodePrompt: () => Effect.succeed({ input: "encoded" }),
      decode: () => ({
        events: Stream.empty,
        parsers: new Map(),
        logprobs: [],
      }),
    } satisfies Codec<CustomRequest, CustomRequest, CustomChunk, never>

    const spec = Model.define({
      modelId: "custom-model",
      endpoint: "https://provider.example",
      path: "/generate",
      codec,
      requestSchema: CustomRequestSchema,
      buildRequest: () => Effect.succeed({ input: "encoded" }),
      decodePayload: () => Effect.succeed({ output: "decoded" }),
    })

    expect(spec.modelId).toBe("custom-model")
  })
})
