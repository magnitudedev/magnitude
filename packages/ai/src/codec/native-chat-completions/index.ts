
import type { Codec } from "../codec"
import type { Schema } from "effect"
import type { NormalizedChatCompletionsStreamChunk } from "./chunk"
import type { ChatCompletionsPrompt } from "../../wire/chat-completions"
import { ChatCompletionsPromptSchema } from "../../wire/chat-completions"
import { encodePrompt } from "./encode"
import type { PromptContributionError } from "./encode"
import { decode } from "./decode"

export const nativeChatCompletionsCodec = {
  id: "native-chat-completions",
  promptSchema: ChatCompletionsPromptSchema,
  encodePrompt,
  decode,
} satisfies Codec<
  Schema.Schema.Type<typeof ChatCompletionsPromptSchema>,
  ChatCompletionsPrompt,
  NormalizedChatCompletionsStreamChunk,
  PromptContributionError
>
