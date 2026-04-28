import { JSONSchema } from "effect"
import type { EncodeOptions } from "../codec"
import type { Prompt, PromptShape } from "../../prompt/prompt"
import type { ToolDefinition } from "../../tools/tool-definition"
import type {
  ChatCompletionsRequest,
  ChatContentPart,
  ChatMessage,
  ChatTool,
  ChatToolCall,
} from "../../wire/chat-completions"

function toPromptShape(prompt: Prompt | PromptShape): PromptShape {
  return prompt as PromptShape
}

function encodeImageUrl(data: string, mediaType: string): string {
  return `data:${mediaType};base64,${data}`
}

function encodeUserContent(message: PromptShape["messages"][number] & { readonly _tag: "UserMessage" }): string | readonly ChatContentPart[] {
  if (message.parts.every((part) => part._tag === "TextPart")) {
    return message.parts.map((part) => part.text).join("\n")
  }

  return message.parts.map((part): ChatContentPart =>
    part._tag === "TextPart"
      ? { type: "text", text: part.text }
      : {
          type: "image_url",
          image_url: {
            url: encodeImageUrl(part.data, part.mediaType),
          },
        },
  )
}

function encodeAssistantToolCall(toolCall: {
  readonly id: string
  readonly name: string
  readonly input: unknown
}): ChatToolCall {
  return {
    id: toolCall.id,
    type: "function",
    function: {
      name: toolCall.name,
      arguments: JSON.stringify(toolCall.input),
    },
  }
}

function encodeAssistantMessage(
  message: PromptShape["messages"][number] & { readonly _tag: "AssistantMessage" },
): ChatMessage {
  const content = message.text ?? null
  const reasoningContent = message.reasoning ?? null
  const toolCalls = message.toolCalls?.map(encodeAssistantToolCall)

  return {
    role: "assistant",
    content,
    ...(reasoningContent !== null ? { reasoning_content: reasoningContent } : {}),
    ...(toolCalls && toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  }
}

function encodeToolResultContent(
  message: PromptShape["messages"][number] & { readonly _tag: "ToolResultMessage" },
): string | readonly ChatContentPart[] {
  if (message.parts.every((part) => part._tag === "TextPart")) {
    return message.parts.map((part) => part.text).join("\n")
  }

  return message.parts.map((part): ChatContentPart =>
    part._tag === "TextPart"
      ? { type: "text", text: part.text }
      : {
          type: "image_url",
          image_url: {
            url: encodeImageUrl(part.data, part.mediaType),
          },
        },
  )
}

function encodeMessage(message: PromptShape["messages"][number]): ChatMessage {
  switch (message._tag) {
    case "UserMessage":
      return {
        role: "user",
        content: encodeUserContent(message),
      }
    case "AssistantMessage":
      return encodeAssistantMessage(message)
    case "ToolResultMessage":
      return {
        role: "tool",
        tool_call_id: message.toolCallId,
        content: encodeToolResultContent(message),
      }
  }
}

function schemaToJsonSchema(schema: ToolDefinition<any, any>["inputSchema"]): Record<string, unknown> {
  return { ...JSONSchema.make(schema) }
}

function encodeTool(tool: ToolDefinition<any, any>): ChatTool {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: schemaToJsonSchema(tool.inputSchema),
    },
  }
}

export function encode(
  model: string,
  prompt: Prompt | PromptShape,
  tools: readonly ToolDefinition<any, any>[],
  options: EncodeOptions,
): ChatCompletionsRequest {
  const promptShape = toPromptShape(prompt)
  const messages: ChatMessage[] = []

  if (promptShape.system.length > 0) {
    messages.push({
      role: "system",
      content: promptShape.system.join("\n"),
    })
  }

  for (const message of promptShape.messages) {
    messages.push(encodeMessage(message))
  }

  const encodedTools = tools.map(encodeTool)

  return {
    model,
    messages,
    ...(encodedTools.length > 0
      ? {
          tools: encodedTools,
          tool_choice: "auto" as const,
        }
      : {}),
    ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
    ...(options.stop && options.stop.length > 0 ? { stop: options.stop } : {}),
    ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
    ...(options.topP !== undefined ? { top_p: options.topP } : {}),
    ...(options.reasoningEffort !== undefined
      ? { reasoning_effort: options.reasoningEffort }
      : {}),
    stream: true,
    stream_options: { include_usage: true },
  }
}
