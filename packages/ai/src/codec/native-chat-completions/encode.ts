import { JSONSchema } from "effect"
import { Prompt } from "../../prompt/prompt"
import type { ToolDefinition } from "../../tools/tool-definition"
import type {
  ChatCompletionsRequest,
  ChatContentPart,
  ChatMessage,
  ChatTool,
  ChatToolCall,
} from "../../wire/chat-completions"

function encodeImageUrl(data: string, mediaType: string): string {
  return `data:${mediaType};base64,${data}`
}

function encodeUserContent(message: { readonly parts: readonly ({ readonly _tag: "TextPart"; readonly text: string } | { readonly _tag: "ImagePart"; readonly data: string; readonly mediaType: string })[] }): string | readonly ChatContentPart[] {
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
  message: { readonly _tag: "AssistantMessage"; readonly text?: string | null; readonly reasoning?: string | null; readonly toolCalls?: readonly { readonly id: string; readonly name: string; readonly input: unknown }[] },
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
  message: { readonly parts: readonly ({ readonly _tag: "TextPart"; readonly text: string } | { readonly _tag: "ImagePart"; readonly data: string; readonly mediaType: string })[] },
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

function encodeMessage(message: any): ChatMessage {
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
    default:
      throw new Error(`Unknown message tag: ${(message as any)._tag}`)
  }
}

function schemaToJsonSchema(schema: ToolDefinition["inputSchema"]): Record<string, unknown> {
  return { ...JSONSchema.make(schema) }
}

function encodeTool(tool: ToolDefinition): ChatTool {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: schemaToJsonSchema(tool.inputSchema),
    },
  }
}

export function encodePrompt(
  model: string,
  prompt: Prompt,
  tools: readonly ToolDefinition[],
): Partial<ChatCompletionsRequest> {
  const messages: ChatMessage[] = []

  if (prompt.system.length > 0) {
    messages.push({
      role: "system",
      content: prompt.system,
    })
  }

  for (const message of prompt.messages) {
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
  }
}
