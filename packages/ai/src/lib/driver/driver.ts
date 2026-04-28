import type { Stream } from "effect"
import type {
  ChatCompletionsRequest,
  ChatCompletionsStreamChunk,
} from "../wire/chat-completions"
import type { ModelError } from "../errors/model-error"

export interface Driver {
  readonly id: string
  readonly stream: (
    request: ChatCompletionsRequest,
    endpoint: string,
    authToken: string,
  ) => Stream.Stream<ChatCompletionsStreamChunk, ModelError>
}
