import { describe, it, expect } from "vitest"
import { imagePlaceholder, normalizeVision } from "../normalize-vision"
import { Prompt } from "../prompt"
import type { TextPart, ImagePart } from "../parts"
import type { ToolCallId, ProviderToolCallId } from "../ids"
import { Option } from "effect"
import type { UserMessage, AssistantMessage, ToolResultMessage } from "../messages"

const text = (t: string): TextPart => ({ _tag: "TextPart", text: t })
const image = (opts?: { mediaType?: ImagePart["mediaType"]; dimensions?: ImagePart["dimensions"] }): ImagePart => ({
  _tag: "ImagePart",
  data: "base64data",
  mediaType: opts?.mediaType ?? "image/png",
  dimensions: opts?.dimensions,
})

describe("imagePlaceholder", () => {
  it("includes mediaType when no dimensions", () => {
    expect(imagePlaceholder(image({ mediaType: "image/png" }))).toBe(
      "[Image placeholder: current model does not support images — image/png]",
    )
  })

  it("includes dimensions when provided", () => {
    expect(imagePlaceholder(image({ dimensions: { width: 1920, height: 1080 } }))).toBe(
      "[Image placeholder: current model does not support images — 1920x1080]",
    )
  })

  it("prefers dimensions over mediaType", () => {
    expect(
      imagePlaceholder(image({ mediaType: "image/png", dimensions: { width: 800, height: 600 } })),
    ).toBe("[Image placeholder: current model does not support images — 800x600]")
  })
})

describe("normalizeVision", () => {
  const basePrompt = (messages: Prompt["messages"]) =>
    Prompt.from({ system: "sys", messages })

  it("replaces ImageParts in UserMessage", () => {
    const prompt = basePrompt([
      { _tag: "UserMessage", parts: [image({ mediaType: "image/jpeg" })] } as UserMessage,
    ])
    const result = normalizeVision(prompt)
    expect(result.messages[0]).toMatchObject({
      _tag: "UserMessage",
      parts: [{ _tag: "TextPart", text: expect.stringContaining("image/jpeg") }],
    })
  })

  it("replaces ImageParts in ToolResultMessage", () => {
    const prompt = basePrompt([
      {
        _tag: "ToolResultMessage",
        toolCallId: "tc1" as ToolCallId,
        providerToolCallId: "tc1" as ProviderToolCallId,
        toolName: "view",
        parts: [image({ mediaType: "image/webp" })],
      } as ToolResultMessage,
    ])
    const result = normalizeVision(prompt)
    expect(result.messages[0]).toMatchObject({
      _tag: "ToolResultMessage",
      parts: [{ _tag: "TextPart", text: expect.stringContaining("image/webp") }],
    })
  })

  it("leaves AssistantMessage unchanged", () => {
    const assistant: AssistantMessage = { _tag: "AssistantMessage", reasoning: Option.none(), text: Option.some("hello"), toolCalls: Option.none() }
    const prompt = basePrompt([
      { _tag: "UserMessage", parts: [text("hi")] } as UserMessage,
      assistant,
      { _tag: "UserMessage", parts: [text("bye")] } as UserMessage,
    ])
    const result = normalizeVision(prompt)
    // No images so same prompt returned
    expect(result).toBe(prompt)
  })

  it("returns same prompt reference if no images present", () => {
    const prompt = basePrompt([
      { _tag: "UserMessage", parts: [text("just text")] } as UserMessage,
    ])
    expect(normalizeVision(prompt)).toBe(prompt)
  })

  it("handles mixed parts — only replaces ImageParts", () => {
    const prompt = basePrompt([
      {
        _tag: "UserMessage",
        parts: [text("before"), image(), text("after")],
      } as UserMessage,
    ])
    const result = normalizeVision(prompt)
    const parts = (result.messages[0] as UserMessage).parts
    expect(parts).toHaveLength(3)
    expect(parts[0]).toEqual({ _tag: "TextPart", text: "before" })
    expect(parts[1]).toMatchObject({ _tag: "TextPart", text: expect.stringContaining("placeholder") })
    expect(parts[2]).toEqual({ _tag: "TextPart", text: "after" })
  })
})
