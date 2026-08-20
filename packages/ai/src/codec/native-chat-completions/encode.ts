import type { JsonSchemaObject } from "@magnitudedev/utils/schema"
import { Data, Effect, Option, Schema } from "effect"
import type * as ParseResult from "effect/ParseResult"
import type { Prompt } from "../../prompt/prompt"
import type {
  AssistantMessage,
  Message,
  ToolResultMessage,
  UserMessage,
} from "../../prompt/messages"
import type { ToolCallPart } from "../../prompt/parts"
import type { ToolDefinition } from "../../tools/tool-definition"
import { ChatCompletionsPromptSchema } from "../../wire/chat-completions"
import type {
  ChatContentPart,
  ChatCompletionsPrompt,
  ChatTool,
  ChatToolCall,
} from "../../wire/chat-completions"
import { makeNativeToolParametersJsonSchema } from "./tool-json-schema"
import { toCauseInfo, type CauseInfo } from "../../errors/failure"

export class PromptContributionError extends Data.TaggedError("PromptContributionError")<{
  readonly failure:
    | { readonly _tag: "PromptMappingFailed"; readonly cause: CauseInfo }
    | { readonly _tag: "PromptEncodingFailed"; readonly error: ParseResult.ParseError }
}> {}

function encodeImageUrl(data: string, mediaType: string): string {
  return `data:${mediaType};base64,${data}`
}

function encodeUserContent(message: UserMessage): string | readonly ChatContentPart[] {
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

function encodeAssistantToolCall(toolCall: ToolCallPart): ChatToolCall {
  return {
    id: toolCall.providerToolCallId,
    type: "function",
    function: {
      name: toolCall.name,
      arguments: JSON.stringify(toolCall.input),
    },
  }
}

type ChatCompletionsPromptValue = Schema.Schema.Type<typeof ChatCompletionsPromptSchema>
type ChatCompletionsRequestMessageValue = ChatCompletionsPromptValue["messages"][number]

function encodeAssistantMessage(message: AssistantMessage): ChatCompletionsRequestMessageValue {
  const content = Option.getOrElse(message.text, () => "")
  const reasoningContent = message.reasoning
  const toolCalls = message.toolCalls.pipe(
    Option.map((calls) => calls.map(encodeAssistantToolCall)),
    Option.filter((calls) => calls.length > 0),
  )

  return {
    role: "assistant",
    content,
    reasoning_content: reasoningContent,
    tool_calls: toolCalls,
  }
}

function encodeToolResultContent(message: ToolResultMessage): string | readonly ChatContentPart[] {
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

function absurdMessage(message: never): never {
  throw new Error(`Unknown prompt message: ${String(message)}`)
}

function encodeMessage(message: Message): ChatCompletionsRequestMessageValue {
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
      return absurdMessage(message)
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
): Effect.Effect<ChatCompletionsPrompt, PromptContributionError> {
  return Effect.try({
    try: (): ChatCompletionsPromptValue => {
      const messages: ChatCompletionsRequestMessageValue[] = []

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
        tools: encodedTools.length > 0 ? Option.some(encodedTools) : Option.none(),
      }
    },
    catch: (cause) => new PromptContributionError({
      failure: { _tag: "PromptMappingFailed", cause: toCauseInfo(cause) },
    }),
  }).pipe(
    Effect.flatMap((value) => Schema.encode(ChatCompletionsPromptSchema)(value).pipe(
      Effect.mapError((error) => new PromptContributionError({
        failure: { _tag: "PromptEncodingFailed", error },
      })),
    )),
  )
}
