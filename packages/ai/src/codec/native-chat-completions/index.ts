import type { Codec } from "../codec"
import type {
  ChatCompletionsRequest,
  ChatCompletionsStreamChunk,
} from "../../wire/chat-completions"
import { encodePrompt } from "./encode"
import { decode } from "./decode"

export const nativeChatCompletionsCodec: Codec<ChatCompletionsRequest, ChatCompletionsStreamChunk> = {
  id: "native-chat-completions",
  encodePrompt,
  decode,
}
