import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { Prompt } from "../../../prompt/prompt"
import { ProviderToolCallIdSchema, ToolCallIdSchema } from "../../../prompt/ids"
import type { AssistantMessage } from "../../../prompt/messages"
import { finalizeChatCompletionsRequest } from "../../../wire/chat-completions"
import { encodePrompt } from "../encode"

const requestFor = (assistant: AssistantMessage) => Effect.runPromise(
  encodePrompt("test-model", Prompt.from({
      messages: [
        assistant,
        { _tag: "UserMessage", parts: [{ _tag: "TextPart", text: "Continue" }] },
      ],
    }), []).pipe(
    Effect.flatMap((encodedPrompt) =>
      finalizeChatCompletionsRequest({ ...encodedPrompt, stream: true })
    ),
  ),
)

describe("native Chat Completions request encoding", () => {
  it("canonicalizes reasoning-only assistant history", async () => {
    const request = await requestFor({
      _tag: "AssistantMessage",
      text: Option.none(),
      reasoning: Option.some("thinking"),
      toolCalls: Option.none(),
    })

    expect(request.messages[0]).toEqual({
      role: "assistant",
      content: "",
      reasoning_content: "thinking",
    })
  })

  it("canonicalizes tool-only assistant history", async () => {
    const request = await requestFor({
      _tag: "AssistantMessage",
      text: Option.none(),
      reasoning: Option.none(),
      toolCalls: Option.some([{
        _tag: "ToolCallPart",
        id: Schema.decodeUnknownSync(ToolCallIdSchema)("semantic-call"),
        providerToolCallId: Schema.decodeUnknownSync(ProviderToolCallIdSchema)("provider-call"),
        name: "lookup",
        input: { query: "magnitude" },
      }]),
    })

    expect(request.messages[0]).toEqual({
      role: "assistant",
      content: "",
      tool_calls: [{
        id: "provider-call",
        type: "function",
        function: { name: "lookup", arguments: "{\"query\":\"magnitude\"}" },
      }],
    })
  })

  it("omits absent optional assistant fields", async () => {
    const request = await requestFor({
      _tag: "AssistantMessage",
      text: Option.some("answer"),
      reasoning: Option.none(),
      toolCalls: Option.none(),
    })

    expect(request.messages[0]).toEqual({ role: "assistant", content: "answer" })
  })

  it("preserves present empty reasoning", async () => {
    const request = await requestFor({
      _tag: "AssistantMessage",
      text: Option.some("answer"),
      reasoning: Option.some(""),
      toolCalls: Option.none(),
    })

    expect(request.messages[0]).toEqual({
      role: "assistant",
      content: "answer",
      reasoning_content: "",
    })
  })

  it("preserves an empty assistant turn with canonical string content", async () => {
    const request = await requestFor({
      _tag: "AssistantMessage",
      text: Option.none(),
      reasoning: Option.none(),
      toolCalls: Option.none(),
    })

    expect(request.messages[0]).toEqual({ role: "assistant", content: "" })
  })
})
