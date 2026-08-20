import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  decodeChatCompletionsRequest,
  encodeChatCompletionsRequest,
  finalizeChatCompletionsRequest,
} from "./request"

const baseRequest = {
  model: "test-model",
  messages: [{ role: "user" as const, content: "Hello" }],
  stream: true as const,
}

describe("ChatCompletionsRequestSchema", () => {
  it("accepts every request message form and preserves their encoded representation", async () => {
    const messages = [
      { role: "system" as const, content: "System" },
      { role: "user" as const, content: "Hello" },
      {
        role: "user" as const,
        content: [
          { type: "text" as const, text: "Look" },
          { type: "image_url" as const, image_url: { url: "data:image/png;base64,AA==" } },
        ],
      },
      { role: "assistant" as const, content: "" },
      { role: "tool" as const, tool_call_id: "call-1", content: "result" },
    ]

    const encoded = await Effect.runPromise(finalizeChatCompletionsRequest({
      ...baseRequest,
      messages,
    }))

    expect(encoded.messages).toEqual(messages)
  })

  it.each([
    "auto",
    "none",
    "required",
    { type: "function", function: { name: "lookup" } },
    {
      type: "allowed_tools",
      allowed_tools: {
        mode: "auto",
        tools: [{ type: "function", function: { name: "lookup" } }],
      },
    },
    { type: "grammar", grammar: "root ::= 'ok'" },
  ] as const)("accepts tool choice %#", async (toolChoice) => {
    const encoded = await Effect.runPromise(finalizeChatCompletionsRequest({
      ...baseRequest,
      tool_choice: toolChoice,
    }))

    expect(encoded.tool_choice).toEqual(toolChoice)
  })

  it("omits every absent optional request property", async () => {
    const encoded = await Effect.runPromise(finalizeChatCompletionsRequest(baseRequest))
    expect(encoded).toEqual(baseRequest)
  })

  it("does not impose provider policy on reasoning effort", async () => {
    const encoded = await Effect.runPromise(finalizeChatCompletionsRequest({
      ...baseRequest,
      reasoning_effort: "",
    }))

    expect(encoded.reasoning_effort).toBe("")
  })

  it("decodes optional fields to Option and isolates provider extensions", async () => {
    const decoded = await Effect.runPromise(decodeChatCompletionsRequest({
      ...baseRequest,
      temperature: 0.5,
      provider_extension: { enabled: true },
    }))

    expect(decoded.temperature).toEqual(Option.some(0.5))
    expect(decoded.max_tokens).toEqual(Option.none())
    expect(decoded.extensions).toEqual({
      provider_extension: { enabled: true },
    })

    const encoded = await Effect.runPromise(encodeChatCompletionsRequest({
      ...decoded,
      temperature: Option.none(),
    }))
    expect(encoded).not.toHaveProperty("temperature")
    expect(encoded).toHaveProperty("provider_extension", { enabled: true })
    expect(JSON.stringify(encoded)).not.toContain("undefined")
  })

  it("rejects a present undefined property rather than treating it as absent", async () => {
    const result = await Effect.runPromiseExit(finalizeChatCompletionsRequest({
      ...baseRequest,
      temperature: undefined,
    }))

    expect(result._tag).toBe("Failure")
  })

  it("rejects extensions that attempt to replace standard request properties", async () => {
    const decoded = await Effect.runPromise(decodeChatCompletionsRequest(baseRequest))
    const result = await Effect.runPromiseExit(encodeChatCompletionsRequest({
      ...decoded,
      extensions: { temperature: 0.5 },
    }))

    expect(result._tag).toBe("Failure")
  })

  it.each([
    { ...baseRequest, messages: [{ role: "assistant", content: null }] },
    { ...baseRequest, max_tokens: null },
    { ...baseRequest, tool_choice: "sometimes" },
    { ...baseRequest, provider_extension: () => "not JSON" },
  ])("rejects malformed or non-JSON request input %#", async (input) => {
    const result = await Effect.runPromiseExit(finalizeChatCompletionsRequest(input))
    expect(result._tag).toBe("Failure")
  })
})
