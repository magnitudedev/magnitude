import type { Stream } from 'effect'
import type { DriverError } from '@magnitudedev/drivers'
import type { ResponseStreamEvent } from '../events/turn-part-event'
import type { EncodeOptions, CodecDecodeError } from '../codec'
import type { ToolDef } from '../tools/tool-def'

/**
 * ModelAdapter — adapter interface for the completions paradigm (Phase 3+).
 *
 * A ModelAdapter knows how to render a prompt and parse a raw text stream
 * for a specific model family (Kimi K2, Qwen, DeepSeek, etc.). It is used
 * by CompletionsCodec to:
 *
 *   1. Render the model's native chat template into a raw prompt string.
 *   2. Parse the raw text stream back into ResponseStreamEvents.
 *
 * This is declared in Phase 0 as a forward-looking contract. No concrete
 * implementations are created in this plan — the completions paradigm is
 * a follow-up effort.
 *
 * encodeTools  — render tool declarations into the model's native format
 *                (some models use special JSON schemas, some use XML, etc.)
 * encodePrompt — render the full prompt string from memory + tools.
 *                Consumes the model's chat_template via @huggingface/jinja.
 * decode       — parse a raw text stream into ResponseStreamEvents.
 *                The adapter understands the model's native thinking tokens
 *                and tool-call delimiters.
 */
export interface ModelAdapter {
  readonly id: string

  readonly encodeTools: (
    tools: readonly ToolDef[],
  ) => string

  readonly encodePrompt: (
    memory:  readonly unknown[],
    tools:   readonly ToolDef[],
    options: EncodeOptions,
  ) => string

  readonly decode: (
    text: Stream.Stream<string, DriverError>,
  ) => Stream.Stream<ResponseStreamEvent, CodecDecodeError | DriverError>
}
