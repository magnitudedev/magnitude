// Namespaces
export { Model } from "./model/define"
export {
  NativeChatCompletions,
} from "./protocol/native-chat-completions"
export { Auth, type AuthApplicator } from "./auth/auth"
export { Option } from "./options/option"

// Core types
export type { ProviderModelCapabilities, ImagePlaceholderConfig } from "./model/capabilities"
export type {
  ModelSpec,
  ModelStreamEvent,
  ModelStreamResult,
} from "./model/model-spec"
export type { BoundModel } from "./model/bound-model"
export type { OptionDef, InferCallOptions } from "./options/option"

// Prompt
export { Prompt, type TerminalMessages } from "./prompt/prompt"
export { PromptBuilder } from "./prompt/prompt-builder"
export type {
  UserMessage,
  AssistantMessage,
  ToolResultMessage,
  UserPart,
  ToolResultPart,
  Message,
  TerminalMessage,
} from "./prompt/messages"
export {
  UserPartSchema,
  UserMessageSchema,
  AssistantMessageSchema,
  ToolResultMessageSchema,
  MessageSchema,
} from "./prompt/messages"
export type { TextPart, ImagePart, ImageMediaType, ToolCallPart, JsonValue } from "./prompt/parts"
export {
  TextPartSchema,
  ImagePartSchema,
  ToolCallPartSchema,
  JsonValueSchema,
} from "./prompt/parts"
export { normalizeVision, imagePlaceholder } from "./prompt/normalize-vision"
export { createToolCallId, ProviderToolCallIdSchema, ToolCallIdSchema } from "./prompt/ids"
export type { ToolCallId, ProviderToolCallId } from "./prompt/ids"

// Tools
export type { ToolDefinition } from "./tools/tool-definition"
export { defineTool } from "./tools/tool-definition"

// Response
export type { ResponseStreamEvent, ValidationIssue, FinishReason, StreamEnd, RawLogprobEntry, RawInputToken, RawOutputToken } from "./response/events"
export type { ResponseUsage } from "./response/usage"
export { GenerationPerformanceSchema } from "./response/performance"
export type { GenerationPerformance } from "./response/performance"
export { formatValidationIssue } from "./response/validation-issue"

export {
  formatModelAttemptFailureMessage,
  formatStreamStartFailureMessage,
  formatStreamFailureMessage,
  snapshotModelAttemptFailure,
  streamStartFailureFromRejectedResponse,
} from "./errors/classify"
export type { ModelAttemptFailureSnapshot } from "./errors/classify"
export {
  AiRetryPolicy,
  AiRetryPolicyLive,
  AiBillingPolicy,
  AiBillingPolicyLive,
  UpstreamRetryability,
  BillingDisposition,
  RetryAfter,
  defaultRetryabilityForFailure,
} from "./errors/policy"
export {
  acceptedHttpResponse,
  getHeader,
  headersFromHeaderList,
  payloadSample,
  toCauseInfo,
  rejectedHttpResponse,
  StreamClientCorrectnessViolation,
  StreamOperationalFailure,
  StreamProviderError,
  StreamProviderCorrectnessViolation,
  StreamStartClientCorrectnessViolation,
  StreamStartOperationalFailure,
  StreamStartProviderCorrectnessViolation,
  StreamStartProviderRejection,
} from "./errors/failure"
export {
  ModelRequestTerminal,
} from "./errors/failure"
export type {
  AcceptedHttpResponse,
  BodyReadError,
  CauseInfo,
  DecoderExpectation,
  HeaderList,
  InvalidConstrainedOutput,
  InvalidProviderChunkProblem,
  StreamStartFailure,
  StreamStartClientComponent,
  StreamStartClientCorrectnessEvidence,
  StreamStartOperationalFailureReason,
  StreamStartProviderCorrectnessViolationReason,
  LastStreamActivity,
  ModelAttemptFailure,
  ModelRequestTerminal as ModelRequestTerminalType,
  PayloadSample,
  ProviderCall,
  ProviderErrorEnvelope,
  ProviderRejection,
  RejectedHttpResponse,
  RetryPolicy,
  SchemaIssue,
  StreamClientComponent,
  StreamClientCorrectnessEvidence,
  StreamFailure,
  StreamFailureContext,
  StreamOperationalFailureReason,
  StreamProgress,
  StreamProviderCorrectnessViolationReason,
  UsageAtTermination,
  UsageMissingReason,
} from "./errors/failure"

// Wire types
export {
  ChatChunkDeltaSchema,
  ChatCompletionsStreamChunk,
  chatCompletionsStreamChunkFields,
  ChatCompletionsRequestSchema,
  ChatCompletionsRequestExtensionsSchema,
  ChatCompletionsPromptSchema,
  ChatToolChoiceSchema,
  ChatToolSchema,
  decodeChatCompletionsRequest,
  encodeChatCompletionsRequest,
  finalizeChatCompletionsRequest,
} from "./wire/chat-completions"
export type {
  ChatCompletionsRequest,
  ChatCompletionsPrompt,
  ChatCompletionsRequestMessage,
  ChatMessage,
  ChatMessageRole,
  ChatTool,
  ChatToolCall,
  ChatToolChoice,
  ChatNamedFunctionToolChoice,
  ChatAllowedToolsToolChoice,
} from "./wire/chat-completions"

// Codec
export type { Codec } from "./codec/codec"
export { nativeChatCompletionsCodec } from "./codec/native-chat-completions/index"
export {
  normalizeChatCompletionsChunk,
  type NormalizedChatCompletionsStreamChunk,
} from "./codec/native-chat-completions/chunk"
export {
  decodeChatCompletionsJson,
  decodeChatCompletionsPayload,
  standardChatCompletionChunkDecoder,
  type ChatCompletionChunkDecoder,
} from "./codec/native-chat-completions/chunk-decoder"
export { makeNativeToolParametersJsonSchema } from "./codec/native-chat-completions/tool-json-schema"

// Trace
export { TraceListener } from "./trace"
export type { ModelCallTrace, AssembledToolCall, TokenLogprob } from "./trace"

// Streaming field parser
export { createStreamingFieldParser } from "./streaming/field-parser"
export type { StreamingFieldParser } from "./streaming/field-parser"
export type { FieldEvent, StreamingPartial, StreamingLeaf } from "./streaming/types"

// Provider-agnostic contract (merged from packages/provider)
export * from "./provider"
