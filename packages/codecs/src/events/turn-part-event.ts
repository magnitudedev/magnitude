/**
 * ResponseStreamEvent — plain TypeScript discriminated union.
 *
 * These are the events emitted by Codec.decode as it processes the raw
 * SSE chunk stream from the provider. They form the canonical vocabulary
 * for the assistant-turn execution path:
 *
 *   thought_*         — reasoning block (streaming)
 *   message_*         — user-facing text content (streaming)
 *   tool_call_*       — a tool invocation (streaming input, then resolved)
 *   response_done     — terminal response metadata with finish reason + usage
 *
 * Plain TS (no Schema) — produced inside the codec with statically-known
 * shapes; consumed by TurnEngine and projections. Not a wire-boundary type.
 */

export interface ResponseUsage {
  readonly inputTokens: number
  readonly outputTokens: number
  readonly cacheReadTokens: number | null
  readonly cacheWriteTokens: number | null
}

export type ResponseStreamEvent =
  // ── Reasoning ──────────────────────────────────────────────────────────────
  | { readonly type: 'thought_start';   readonly level: 'low' | 'medium' | 'high' }
  | { readonly type: 'thought_delta';   readonly text: string }
  | { readonly type: 'thought_end' }

  // ── User-facing text response ──────────────────────────────────────────────
  | { readonly type: 'message_start' }
  | { readonly type: 'message_delta';   readonly text: string }
  | { readonly type: 'message_end' }

  // ── Tool calls ─────────────────────────────────────────────────────────────
  // toolCallId distinguishes parallel tool calls within a turn.
  // Codec drives a per-call StreamingJsonParser to surface field boundaries
  // inside the JSON arguments stream. Field events fire for leaves
  // (string/number/boolean/null) and containers (object/array). Leaves get
  // deltas as their text accumulates; containers don't (deltas are routed to
  // descendant leaves). Path is decimal-string-encoded for array indices.
  // The full tool input is derivable by accumulating field_end values.
  | { readonly type: 'tool_call_start';        readonly toolCallId: string; readonly toolName: string }
  | { readonly type: 'tool_call_field_start';  readonly toolCallId: string; readonly path: readonly string[] }
  | { readonly type: 'tool_call_field_delta';  readonly toolCallId: string; readonly path: readonly string[]; readonly delta: string }
  | { readonly type: 'tool_call_field_end';    readonly toolCallId: string; readonly path: readonly string[]; readonly value: unknown }
  | { readonly type: 'tool_call_end';          readonly toolCallId: string }

  // ── Completion ─────────────────────────────────────────────────────────────
  | {
      readonly type: 'response_done'
      /** finish_reason normalised across providers. */
      readonly reason: 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'other'
      readonly usage: ResponseUsage
    }
