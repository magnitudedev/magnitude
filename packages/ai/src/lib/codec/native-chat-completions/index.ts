import type { Codec } from "../codec"
import { decode } from "./decode"
import { encode } from "./encode"

export const nativeChatCompletionsCodec: Codec = {
  id: "native-chat-completions",
  encode,
  decode,
}
