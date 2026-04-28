// =============================================================================
// Codec interface + errors + encode options
// =============================================================================
export type { Codec, EncodeOptions } from './codec'
export { CodecEncodeError, CodecDecodeError } from './codec'

// =============================================================================
// TurnPart — memory representation of one assistant-turn part
// =============================================================================
export type { TurnPart, ThoughtPart, MessagePart, ToolCallPart } from './memory/turn-part'

// =============================================================================
// TurnPartEvent — canonical streaming event vocabulary from codec.decode
// =============================================================================
export type { ResponseStreamEvent, ResponseUsage } from './events/turn-part-event'

// =============================================================================
// ID constructors
// =============================================================================
export { newThoughtId, newMessageId, newToolCallId } from './memory/ids'

// =============================================================================
// ToolDef — tool declaration passed to codec.encode
// =============================================================================
export { ToolDef } from './tools/tool-def'

// =============================================================================
// ModelAdapter — completions-paradigm adapter interface (declared, no impls)
// =============================================================================
export type { ModelAdapter } from './adapters/model-adapter'

// =============================================================================
// NativeChatCompletionsCodec — native OpenAI-format chat completions codec
// =============================================================================
export { NativeChatCompletionsCodec } from './impls/native-chat-completions'
export type { EncodeConfig } from './impls/native-chat-completions'
