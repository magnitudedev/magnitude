import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import {
  PromptBuilder,
  Prompt,
  ProviderModelIdSchema,
  defineTool,
} from "@magnitudedev/ai"
import { Effect, Exit, Layer, Option, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient } from "../client.js"
import { makeIcnApiClient } from "@magnitudedev/icn-protocol/client"
import * as Generated from "@magnitudedev/icn-protocol/schemas"
import { IcnProvider, IcnProviderModelResolver, makeIcnProvider } from "./source.js"
import type { IcnModelPreparation } from "./contract.js"

const TEST_BASE_URL = "http://icn.test"

const makeTestLayer = (
  http: HttpClient.HttpClient,
  runtimeModelId?: ReturnType<typeof ProviderModelIdSchema.make>,
) => {
  const httpLayer = Layer.succeed(HttpClient.HttpClient, http)
  const clientLayer = Layer.effect(
    IcnClient,
    makeIcnApiClient({ baseUrl: TEST_BASE_URL }),
  ).pipe(Layer.provide(httpLayer))
  const resolverLayer = Layer.succeed(IcnProviderModelResolver, IcnProviderModelResolver.of({
    resolve: () => Effect.succeed(
      runtimeModelId === undefined
        ? Option.none()
        : Option.some({ runtimeModelId }),
    ),
  }))
  const dependencies = Layer.merge(clientLayer, resolverLayer)

  return makeIcnProvider().pipe(
    Layer.provide(dependencies),
    Layer.merge(dependencies),
    Layer.merge(httpLayer),
  )
}

const jsonResponse = (
  request: Parameters<Parameters<typeof HttpClient.make>[0]>[0],
  body: string,
  status = 200,
) => HttpClientResponse.fromWeb(
  request,
  new Response(body, {
    status,
    headers: { "content-type": "application/json" },
  }),
)

const sseResponse = (
  request: Parameters<Parameters<typeof HttpClient.make>[0]>[0],
  events: readonly object[],
) => HttpClientResponse.fromWeb(
  request,
  new Response(
    `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`,
    { status: 200, headers: { "content-type": "text/event-stream" } },
  ),
)

const sseErrorResponse = (
  request: Parameters<Parameters<typeof HttpClient.make>[0]>[0],
  error: { readonly code: string; readonly message: string; readonly type: string },
  preceding: readonly object[] = [],
) => HttpClientResponse.fromWeb(
  request,
  new Response(
    `${preceding.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}event: error\ndata: ${JSON.stringify({ error })}\n\n`,
    { status: 200, headers: { "content-type": "text/event-stream" } },
  ),
)

describe("ICN local provider", () => {
  it("keeps the local provider catalog product-owned", async () => {
    const http = HttpClient.make((request) =>
      Effect.succeed(jsonResponse(request, '{"object":"list","data":[]}')))

    await Effect.runPromise(Effect.gen(function* () {
      const provider = yield* IcnProvider
      expect(yield* provider.catalog.list).toEqual([])
      expect(yield* provider.catalog.refresh).toEqual([])
    }).pipe(Effect.provide(makeTestLayer(http))))
  })

  it("fails before inference when a public model has no native association", async () => {
    let chatRequests = 0
    const http = HttpClient.make((request) => {
      if (request.url.endsWith("/v1/chat/completions")) chatRequests += 1
      return Effect.succeed(jsonResponse(request, '{"object":"list","data":[]}'))
    })
    const modelId = ProviderModelIdSchema.make("mdl_test")

    const result = await Effect.runPromiseExit(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId)
      return yield* bound.stream(PromptBuilder.empty().user("hello").build(), [])
    }).pipe(Effect.provide(makeTestLayer(http))))

    expect(Exit.isFailure(result)).toBe(true)
    expect(chatRequests).toBe(0)
  })

  it("sends canonical reasoning-only assistant history through the generated client", async () => {
    const modelId = ProviderModelIdSchema.make("mdl_test")
    let requestJson: string | undefined
    const http = HttpClient.make((request) => {
      if (request.url.endsWith("/v1/chat/completions") && request.body._tag === "Uint8Array") {
        requestJson = new TextDecoder().decode(request.body.body)
      }
      return Effect.succeed(sseResponse(request, []))
    })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId)
      yield* bound.stream(Prompt.from({
        messages: [
          {
            _tag: "AssistantMessage",
            reasoning: Option.some("thinking"),
            text: Option.none(),
            toolCalls: Option.none(),
          },
          { _tag: "UserMessage", parts: [{ _tag: "TextPart", text: "Continue" }] },
        ],
      }), [])
    }).pipe(Effect.provide(makeTestLayer(http, modelId)))))

    const requestBody = Schema.decodeUnknownSync(
      Schema.parseJson(Schema.encodedSchema(Generated.ChatCompletionRequest)),
    )(requestJson)
    expect(requestBody.messages).toContainEqual({
      role: "assistant",
      content: "",
      reasoning_content: "thinking",
    })
  })

  it("omits tool choice when the caller does not supply it", async () => {
    const modelId = ProviderModelIdSchema.make("mdl_test")
    let requestJson: string | undefined
    const http = HttpClient.make((request) => {
      if (request.url.endsWith("/v1/chat/completions") && request.body._tag === "Uint8Array") {
        requestJson = new TextDecoder().decode(request.body.body)
      }
      return Effect.succeed(sseResponse(request, []))
    })
    const tool = defineTool({
      name: "lookup",
      description: "Look up a value",
      inputSchema: Schema.Struct({ query: Schema.String }),
      outputSchema: Schema.String,
    })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId)
      yield* bound.stream(PromptBuilder.empty().user("hello").build(), [tool])
    }).pipe(Effect.provide(makeTestLayer(http, modelId)))))

    const requestBody = Schema.decodeUnknownSync(
      Schema.parseJson(Schema.encodedSchema(Generated.ChatCompletionRequest)),
    )(requestJson)
    expect(requestBody).not.toHaveProperty("tool_choice")
  })

  it("reports native request progress without exposing control chunks as model output", async () => {
    const modelId = ProviderModelIdSchema.make("mdl_test")
    const chunk = {
      id: "request-1",
      object: "chat.completion.chunk",
      created: 1,
      model: modelId,
    }
    const http = HttpClient.make((request) => Effect.succeed(sseResponse(request, [
      { ...chunk, choices: [], progress: { phase: "queued" } },
      { ...chunk, choices: [], progress: { phase: "prefill", completed_tokens: 1, total_tokens: 2, cached_tokens: 0 } },
      { ...chunk, choices: [], progress: { phase: "generating" } },
      {
        ...chunk,
        choices: [{
          index: 0,
          delta: { role: "assistant", content: null },
        }],
      },
      {
        ...chunk,
        choices: [{
          index: 0,
          delta: { content: "hello" },
        }],
      },
      {
        ...chunk,
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      },
      {
        ...chunk,
        choices: [],
        usage: {
          prompt_tokens: 1,
          completion_tokens: 1,
          total_tokens: 2,
          prompt_tokens_details: { cached_tokens: 0 },
        },
        timings: {
          cache_n: 0,
          prompt_n: 1,
          prompt_ms: 3,
          time_to_first_token_ms: 6,
          prompt_per_token_ms: 3,
          prompt_per_second: 333.3,
          predicted_n: 1,
          predicted_ms: 50,
          predicted_per_token_ms: 50,
          predicted_per_second: 20,
          sampler_ms: 0.1,
          parser_ms: 0.1,
        },
      },
    ])))
    const output = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId, {
        requestAttribution: {
          key: "test",
          requestStarted: Effect.void,
        },
      })
      const result = yield* bound.stream(
        PromptBuilder.empty().user("hello").build(),
        [],
      )
      return yield* Stream.runCollect(result.events)
    }).pipe(Effect.provide(makeTestLayer(http, modelId)))))

    expect(Array.from(output).map((event) => event._tag)).toEqual([
      "preparation_update",
      "preparation_update",
      "message_start",
      "message_delta",
      "message_end",
      "stream_end",
    ])
    expect(Array.from(output).at(-1)).toMatchObject({
      performance: {
        generatedTokens: 1,
        decodeDurationMs: 50,
        decodeTokensPerSecond: 20,
        timeToFirstTokenMs: 6,
      },
    })
    expect(Array.from(output).slice(0, 2)).toEqual([
      { _tag: "preparation_update", preparation: { phase: "queued" }, requestId: "request-1" },
      {
        _tag: "preparation_update",
        preparation: {
          phase: "prefill",
          completed_tokens: 1,
          total_tokens: 2,
          cached_tokens: 0,
        },
        requestId: "request-1",
      },
    ])
  })

  it("emits preparation progress before any model response event exists", async () => {
    const modelId = ProviderModelIdSchema.make("mdl_test")
    const progress = {
      id: "request-1",
      object: "chat.completion.chunk",
      created: 1,
      model: modelId,
      choices: [],
      progress: { phase: "model_loading", fraction: 0.25 },
    }
    const http = HttpClient.make((request) => Effect.succeed(HttpClientResponse.fromWeb(
      request,
      new Response(new ReadableStream({
        start: (controller) => {
          controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(progress)}\n\n`))
        },
      }), {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
    )))

    const first = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId)
      const result = yield* bound.stream(PromptBuilder.empty().user("hello").build(), [])
      return yield* Stream.runHead(result.events).pipe(Effect.timeout("1 second"))
    }).pipe(Effect.provide(makeTestLayer(http, modelId)))))

    expect(first).toEqual(Option.some({
      _tag: "preparation_update",
      preparation: { phase: "model_loading", fraction: 0.25 },
      requestId: "request-1",
    }))
  })

  it("reports attribution without fabricating lifecycle phases for a rejected request", async () => {
    const modelId = ProviderModelIdSchema.make("mdl_test")
    const http = HttpClient.make((request) => Effect.succeed(jsonResponse(
      request,
      JSON.stringify({
        error: {
          code: "model_unavailable",
          message: "model unavailable",
          retryable: true,
          type: "server_error",
        },
      }),
      500,
    )))
    const lifecycle: string[] = []

    const result = await Effect.runPromiseExit(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId, {
        requestAttribution: {
          key: "test",
          requestStarted: Effect.sync(() => lifecycle.push("started")),
        },
      })
      return yield* bound.stream(PromptBuilder.empty().user("hello").build(), [])
    }).pipe(Effect.provide(makeTestLayer(http, modelId))))

    expect(Exit.isFailure(result)).toBe(true)
    expect(lifecycle).toEqual(["started"])
  })

  it("maps an explicit instance stop before streaming to ModelInstanceStopped", async () => {
    const modelId = ProviderModelIdSchema.make("mdl_test")
    const http = HttpClient.make((request) => Effect.succeed(sseErrorResponse(request, {
      code: "model_instance_stopped",
      message: "model instance was stopped",
      type: "model_error",
    }, [
      {
        id: "request-1",
        object: "chat.completion.chunk",
        created: 1,
        model: modelId,
        choices: [],
        progress: { phase: "queued" },
      },
    ])))

    const events = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId)
      const result = yield* bound.stream(PromptBuilder.empty().user("hello").build(), [])
      return Array.from(yield* Stream.runCollect(result.events))
    }).pipe(Effect.provide(makeTestLayer(http, modelId)))))

    expect(events.at(-1)).toMatchObject({
      _tag: "stream_end",
      terminal: {
        _tag: "ModelInstanceStopped",
      },
    })
  })

  it("maps an explicit instance stop after streaming begins to the same terminal", async () => {
    const modelId = ProviderModelIdSchema.make("mdl_test")
    const base = {
      id: "request-1",
      object: "chat.completion.chunk",
      created: 1,
      model: modelId,
      choices: [],
    }
    const http = HttpClient.make((request) => Effect.succeed(sseErrorResponse(request, {
      code: "model_instance_stopped",
      message: "model instance was stopped",
      type: "model_error",
    }, [
      { ...base, progress: { phase: "generating" } },
    ])))

    const events = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId)
      const result = yield* bound.stream(PromptBuilder.empty().user("hello").build(), [])
      return Array.from(yield* Stream.runCollect(result.events))
    }).pipe(Effect.provide(makeTestLayer(http, modelId)))))

    expect(events.at(-1)).toMatchObject({
      _tag: "stream_end",
      terminal: { _tag: "ModelInstanceStopped" },
    })
  })

  it("does not classify another model error as an explicit request stop", async () => {
    const modelId = ProviderModelIdSchema.make("mdl_test")
    const http = HttpClient.make((request) => Effect.succeed(sseErrorResponse(request, {
      code: "model_unavailable",
      message: "temporary admission failure",
      type: "model_error",
    })))

    const events = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId)
      const result = yield* bound.stream(PromptBuilder.empty().user("hello").build(), [])
      return Array.from(yield* Stream.runCollect(result.events))
    }).pipe(Effect.provide(makeTestLayer(http, modelId)))))

    expect(events.at(-1)).toMatchObject({
      _tag: "stream_end",
      terminal: {
        _tag: "StreamFailed",
        cause: { _tag: "StreamProviderError" },
      },
    })
  })

  it("does not fabricate retry hints absent from the OpenAI error contract", async () => {
    const modelId = ProviderModelIdSchema.make("mdl_test")
    const http = HttpClient.make((request) => Effect.succeed(sseErrorResponse(request, {
      code: "low_memory",
      message: "Not enough memory to load model",
      type: "model_error",
    })))

    const events = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId)
      const result = yield* bound.stream(PromptBuilder.empty().user("hello").build(), [])
      return Array.from(yield* Stream.runCollect(result.events))
    }).pipe(Effect.provide(makeTestLayer(http, modelId)))))

    expect(events.at(-1)).toMatchObject({
      _tag: "stream_end",
      terminal: {
        _tag: "StreamFailed",
        cause: {
          _tag: "StreamProviderError",
          providerError: {
            code: "low_memory",
            retryable: null,
            type: "model_error",
          },
        },
      },
    })
  })
})
