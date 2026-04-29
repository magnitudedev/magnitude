// Namespaces
export { Model } from "./model/define"
export { NativeChatCompletions } from "./protocol/native-chat-completions"
export { Auth, type AuthApplicator } from "./auth/auth"
export { Option } from "./options/option"

// Core types
export type { ModelSpec } from "./model/model-spec"
export type { BoundModel } from "./model/bound-model"
export type { OptionDef, InferCallOptions } from "./options/option"

// Prompt
export { Prompt, type TerminalMessages } from "./prompt/prompt"
export { PromptBuilder } from "./prompt/prompt-builder"
export type {
  UserMessage,
  AssistantMessage,
  ToolResultMessage,
  Message,
  TerminalMessage,
} from "./prompt/messages"
export type { TextPart, ImagePart, ToolCallPart, JsonValue } from "./prompt/parts"
export type { ToolCallId } from "./prompt/ids"

// Tools
export type { ToolDefinition } from "./tools/tool-definition"
export { defineTool } from "./tools/tool-definition"

// Response
export type { ResponseStreamEvent } from "./response/events"
export type { ResponseUsage } from "./response/usage"

// Errors
export {
  AuthFailed,
  RateLimited,
  UsageLimitExceeded,
  ContextLimitExceeded,
  InvalidRequest,
  TransportError,
  ParseError,
} from "./errors/model-error"
export type { ConnectionError, StreamError } from "./errors/model-error"
export { defaultClassifyConnectionError, defaultClassifyStreamError } from "./errors/classify"
export type { HttpConnectionFailure, StreamFailure } from "./errors/failure"

// Wire types
export type { ChatCompletionsRequest, ChatCompletionsStreamChunk } from "./wire/chat-completions"

// Codec
export type { Codec } from "./codec/codec"
export { nativeChatCompletionsCodec } from "./codec/native-chat-completions/index"

// Jsonish
export { createStreamingJsonParser } from "./jsonish/parser"
export type { StreamingJsonParser, ParsedValue } from "./jsonish/types"
