// Driver interface and call options
export type { Driver, DriverCallOptions } from './driver'

// Implementations
export { OpenAIChatCompletionsDriver } from './openai-chat-completions'
export { sseChunks } from './openai-chat-completions'

// Errors
export { DriverError } from './errors'

// Wire types — chat completions
export type {
  ChatMessageRole,
  ChatMessage,
  ChatTool,
  ChatToolCall,
  ChatCompletionsRequest,
} from './wire/chat-completions'
export { ChatCompletionsStreamChunk } from './wire/chat-completions'

// Wire types — completions (future)
export type { CompletionsRequest } from './wire/completions'
export { CompletionsStreamChunk } from './wire/completions'
