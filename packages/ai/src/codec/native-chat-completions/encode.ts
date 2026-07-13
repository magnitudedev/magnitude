import type { JsonSchemaObject } from "@magnitudedev/utils/schema"
import { Option } from "effect"
import { Prompt } from "../../prompt/prompt"
import type { ProviderToolCallId } from "../../prompt/ids"
import type { ToolDefinition } from "../../tools/tool-definition"
import type {
  ChatCompletionsRequest,
  ChatContentPart,
  ChatMessage,
  ChatTool,
  ChatToolCall,
} from "../../wire/chat-completions"
import { makeNativeToolParametersJsonSchema } from "./tool-json-schema"

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
  readonly providerToolCallId: ProviderToolCallId
  readonly name: string
  readonly input: unknown
}): ChatToolCall {
  return {
    id: toolCall.providerToolCallId,
    type: "function",
    function: {
      name: toolCall.name,
      arguments: JSON.stringify(toolCall.input),
    },
  }
}

function encodeAssistantMessage(
  message: { readonly _tag: "AssistantMessage"; readonly text: Option.Option<string>; readonly reasoning: Option.Option<string>; readonly toolCalls: Option.Option<readonly { readonly id: string; readonly providerToolCallId: ProviderToolCallId; readonly name: string; readonly input: unknown }[]> },
): ChatMessage {
  const content = Option.getOrElse(message.text, () => null)
  const reasoningContent = Option.getOrElse(message.reasoning, () => null)
  const toolCalls = Option.map(message.toolCalls, (tcs) => tcs.map(encodeAssistantToolCall))

  return {
    role: "assistant",
    content,
    ...(reasoningContent !== null ? { reasoning_content: reasoningContent } : {}),
    ...(Option.isSome(toolCalls) && toolCalls.value.length > 0 ? { tool_calls: toolCalls.value } : {}),
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
        tool_call_id: message.providerToolCallId,
        content: encodeToolResultContent(message),
      }
    default:
      throw new Error(`Unknown message tag: ${(message as any)._tag}`)
  }
}

function schemaToJsonSchema(schema: ToolDefinition["inputSchema"]): JsonSchemaObject {
  return makeNativeToolParametersJsonSchema(schema)
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
        }
      : {}),
  }
}
