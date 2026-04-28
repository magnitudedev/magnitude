import type { Codec } from '../../codec'
import type { ChatCompletionsRequest, ChatCompletionsStreamChunk } from '@magnitudedev/drivers'
import { encode, type EncodeConfig } from './encode'
import { decode } from './decode'

/**
 * NativeChatCompletionsCodec — factory returning a Codec that encodes agent
 * memory to an OpenAI-compatible chat completions request and decodes the
 * SSE chunk stream to canonical TurnPartEvents.
 *
 * @param config — model-specific capabilities and defaults
 */
export const NativeChatCompletionsCodec = (
  config: EncodeConfig,
): Codec<ChatCompletionsRequest, ChatCompletionsStreamChunk> => ({
  id: 'native-chat-completions',
  encode: (memory, tools, options) => encode(memory, tools, options, config),
  decode,
})
