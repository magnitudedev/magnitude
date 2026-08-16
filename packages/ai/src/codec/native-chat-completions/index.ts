
import type { Codec } from "../codec"
import type {
  ChatCompletionsRequest,
} from "../../wire/chat-completions"
import type { NormalizedChatCompletionsStreamChunk } from "./chunk"
import { encodePrompt } from "./encode"
import { decode } from "./decode"

export const nativeChatCompletionsCodec: Codec<ChatCompletionsRequest, NormalizedChatCompletionsStreamChunk> = {
  id: "native-chat-completions",
  encodePrompt,
  decode,
}
