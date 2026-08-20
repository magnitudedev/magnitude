import { Effect, Exit, Option, Schema } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { describe, expect, it } from "vitest"
import {
  Prompt,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/ai"
import {
  CustomEndpointDeclarationSchema,
  CustomEndpointNameSchema,
} from "@magnitudedev/storage"
import {
  createCustomEndpointProvider,
  customEndpointProviderId,
} from "./provider"

const endpoint = (
  authentication: unknown,
  baseUrl = "https://openrouter.ai/api/v1",
) =>
  Schema.decodeUnknownSync(CustomEndpointDeclarationSchema)({
    displayName: "OpenRouter",
    connection: {
      baseUrl,
      authentication,
    },
    models: {
      "z-ai/glm-5.2": {
        displayName: "GLM 5.2",
        contextWindow: 1048576,
        maxOutputTokens: 128000,
        capabilities: {
          vision: true,
          reasoning: {
            efforts: ["high", "xhigh"],
            defaultEffort: "high",
          },
        },
      },
    },
  })

const endpointWithoutReasoning = (baseUrl: string) =>
  Schema.decodeUnknownSync(CustomEndpointDeclarationSchema)({
    displayName: "Custom",
    connection: {
      baseUrl,
      authentication: { type: "none" },
    },
    models: {
      model: {
        displayName: "Model",
        contextWindow: 128000,
        maxOutputTokens: 8192,
      },
    },
  })

const CapturedRequestSchema = Schema.Struct({
  messages: Schema.Array(Schema.Record({ key: Schema.String, value: Schema.Unknown })),
})

describe("custom endpoint provider", () => {
  it("publishes authored models as custom provider catalog entries", async () => {
    const name = CustomEndpointNameSchema.make("openrouter")
    const instance = createCustomEndpointProvider(
      name,
      endpoint({ type: "none" }),
      {},
    )
    const models = await Effect.runPromise(
      instance.provider.catalog.list.pipe(Effect.provide(FetchHttpClient.layer)),
    )

    expect(instance.kind).toBe("Custom")
    expect(instance.provider.id).toBe(customEndpointProviderId(name))
    expect(instance.authStatus).toEqual({ _tag: "no_auth_required" })
    expect(models).toHaveLength(1)
    expect(models[0]).toMatchObject({
      providerId: "custom:openrouter",
      providerModelId: "z-ai/glm-5.2",
      displayName: "GLM 5.2",
      contextWindow: 1048576,
      maxOutputTokens: 128000,
      servingCapabilities: { tools: true, structuredOutput: false },
    })
    expect(models[0]?.pricing).toEqual(Option.none())
    expect(models[0]?.properties.vision).toMatchObject({ _tag: "Resolved", value: true })
    expect(models[0]?.properties.reasoning).toMatchObject({
      _tag: "Resolved",
      value: ["high", "xhigh"],
    })
  })

  it("keeps an endpoint declared but not configured when its credential is absent", () => {
    const instance = createCustomEndpointProvider(
      CustomEndpointNameSchema.make("openrouter"),
      endpoint({
        type: "bearer",
        credential: { type: "environment", variable: "OPENROUTER_API_KEY" },
      }),
      {},
    )

    expect(instance.authStatus).toEqual({
      _tag: "not_configured",
      reason: "Set OPENROUTER_API_KEY and restart Magnitude",
    })
  })

  it("omits reasoning_effort when the model does not declare reasoning", async () => {
    let requestBody: Record<string, unknown> | undefined
    const server = Bun.serve({
      port: 0,
      fetch: async (request) => {
        requestBody = await request.json() as Record<string, unknown>
        return new Response("data: [DONE]\n\n", {
          headers: { "content-type": "text/event-stream" },
        })
      },
    })

    try {
      const instance = createCustomEndpointProvider(
        CustomEndpointNameSchema.make("custom"),
        endpointWithoutReasoning(`http://127.0.0.1:${server.port}`),
        {},
      )
      const model = await Effect.runPromise(instance.provider.bindModel(
        ProviderModelIdSchema.make("model"),
        {
          defaults: {
            maxTokens: 8192,
            reasoningEffort: ReasoningEffortSchema.make("none"),
          },
        },
      ))
      const prompt = Prompt.from({
        system: "System",
        messages: [{
          _tag: "UserMessage",
          parts: [{ _tag: "TextPart", text: "Hello" }],
        }],
      })

      await Effect.runPromise(model.stream(prompt, [], {}).pipe(
        Effect.provide(FetchHttpClient.layer),
      ))

      expect(requestBody).toMatchObject({ model: "model", max_tokens: 8192 })
      expect(requestBody).not.toHaveProperty("reasoning_effort")
    } finally {
      server.stop(true)
    }
  })

  it("sends string content for reasoning-only assistant history", async () => {
    let assistantMessage: Record<string, unknown> | undefined
    const server = Bun.serve({
      port: 0,
      fetch: async (request) => {
        const requestBody = Schema.decodeUnknownSync(CapturedRequestSchema)(await request.json())
        assistantMessage = requestBody.messages.find(
          (message) => message.role === "assistant",
        )

        // Ollama rejects chat-completions requests when assistant content is null,
        // including reasoning-only turns replayed as conversation history.
        if (typeof assistantMessage?.content !== "string") {
          return new Response("assistant content must be a string", { status: 400 })
        }

        return new Response("data: [DONE]\n\n", {
          headers: { "content-type": "text/event-stream" },
        })
      },
    })

    try {
      const instance = createCustomEndpointProvider(
        CustomEndpointNameSchema.make("ollama"),
        endpoint(
          { type: "none" },
          `http://127.0.0.1:${server.port}`,
        ),
        {},
      )
      const model = await Effect.runPromise(instance.provider.bindModel(
        ProviderModelIdSchema.make("z-ai/glm-5.2"),
        {
          defaults: {
            maxTokens: 8192,
            reasoningEffort: ReasoningEffortSchema.make("high"),
          },
        },
      ))
      const prompt = Prompt.from({
        messages: [
          {
            _tag: "AssistantMessage",
            reasoning: Option.some("thinking"),
            text: Option.none(),
            toolCalls: Option.none(),
          },
          {
            _tag: "UserMessage",
            parts: [{ _tag: "TextPart", text: "Continue" }],
          },
        ],
      })

      const result = await Effect.runPromiseExit(model.stream(prompt, [], {}).pipe(
        Effect.provide(FetchHttpClient.layer),
      ))

      expect(assistantMessage).toEqual({
        role: "assistant",
        content: "",
        reasoning_content: "thinking",
      })
      expect(Exit.isSuccess(result)).toBe(true)
    } finally {
      server.stop(true)
    }
  })
})
