import { Either, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  ChatCompletionsModelNameSchema,
  CustomEndpointDeclarationSchema,
} from "./custom-endpoints"

const declaration = (overrides: Record<string, unknown> = {}) => ({
  displayName: "OpenRouter",
  connection: {
    baseUrl: "https://openrouter.ai/api/v1",
    authentication: {
      type: "bearer",
      credential: { type: "environment", variable: "OPENROUTER_API_KEY" },
    },
  },
  models: {
    "z-ai/glm-5.2": {
      displayName: "GLM 5.2",
      contextWindow: 1048576,
      maxOutputTokens: 128000,
    },
  },
  ...overrides,
})

const decode = Schema.decodeUnknownEither(CustomEndpointDeclarationSchema)

describe("custom endpoint declarations", () => {
  it("decodes a minimal OpenAI-compatible endpoint into explicit optional values", () => {
    const decoded = Either.getOrThrowWith(decode(declaration()), String)

    expect(decoded.displayName).toBe("OpenRouter")
    expect(decoded.connection.headers).toEqual(Option.none())
    const modelName = ChatCompletionsModelNameSchema.make("z-ai/glm-5.2")
    expect(decoded.models[modelName]?.capabilities).toEqual(Option.none())
  })

  it("rejects a completed chat-completions route instead of an API root", () => {
    const result = decode(declaration({
      connection: {
        baseUrl: "https://openrouter.ai/api/v1/chat/completions",
        authentication: { type: "none" },
      },
    }))

    expect(Either.isLeft(result)).toBe(true)
  })

  it("rejects literal headers that replace the configured authentication header", () => {
    const result = decode(declaration({
      connection: {
        baseUrl: "https://openrouter.ai/api/v1",
        authentication: {
          type: "bearer",
          credential: { type: "environment", variable: "OPENROUTER_API_KEY" },
        },
        headers: { Authorization: "literal secret" },
      },
    }))

    expect(Either.isLeft(result)).toBe(true)
  })

  it("rejects inconsistent model token limits", () => {
    const result = decode(declaration({
      models: {
        "z-ai/glm-5.2": {
          displayName: "GLM 5.2",
          contextWindow: 1000,
          maxOutputTokens: 2000,
        },
      },
    }))

    expect(Either.isLeft(result)).toBe(true)
  })

  it("rejects a reasoning default that is not supported", () => {
    const result = decode(declaration({
      models: {
        "z-ai/glm-5.2": {
          displayName: "GLM 5.2",
          contextWindow: 1048576,
          maxOutputTokens: 128000,
          capabilities: {
            reasoning: {
              efforts: ["high", "xhigh"],
              defaultEffort: "medium",
            },
          },
        },
      },
    }))

    expect(Either.isLeft(result)).toBe(true)
  })
})
