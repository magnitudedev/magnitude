import type { ProviderToolCallId, ToolCallId } from "../prompt/ids"
import type { JsonValue } from "../prompt/parts"
import type { ModelStreamTerminal } from "../errors/failure"
import type { ResponseUsage } from "./usage"

/** A single logprob candidate for a generated token. */
export type RawLogprobEntry = {
  /** The token text */
  text: string
  /** Log probability of this token */
  logprob: number
}

/** A single token in raw_input — prompt token with its text and ID. */
export type RawInputToken = {
  /** The token text as decoded from the tokenizer */
  text: string
  /** The token ID in the model's vocabulary */
  id: number
}

/** A single token in raw_output — generated token with its text, ID, and top logprob candidates. */
export type RawOutputToken = {
  /** The token text */
  text: string
  /** The token ID in the model's vocabulary */
  id: number
  /** Top-N logprob candidates for this token position. Null for special tokens (e.g. BOS/EOS) that lack valid probability data. */
  logprobs: readonly RawLogprobEntry[] | null
}

export interface ValidationIssue {
  readonly path: readonly (string | number)[]
  readonly message: string
}

export type FinishReason = "stop" | "tool_calls" | "end_turn" | "length" | "content_filter" | "unknown"

export type StreamEnd = {
  readonly _tag: "stream_end"
  readonly terminal: ModelStreamTerminal
  readonly rawInput?: ReadonlyArray<RawInputToken>
  readonly rawOutput?: ReadonlyArray<RawOutputToken>
}

export type ResponseStreamEvent =
  | { readonly _tag: "thought_start"; readonly level: "low" | "medium" | "high" }
  | { readonly _tag: "thought_delta"; readonly text: string }
  | { readonly _tag: "reasoning_details"; readonly details: readonly JsonValue[] }
  | { readonly _tag: "thought_end" }
  | { readonly _tag: "message_start" }
  | { readonly _tag: "message_delta"; readonly text: string }
  | { readonly _tag: "message_end" }
  | { readonly _tag: "tool_call_start"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string }
  | { readonly _tag: "tool_call_field_start"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly path: readonly string[] }
  | { readonly _tag: "tool_call_field_delta"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly path: readonly string[]; readonly delta: string }
  | { readonly _tag: "tool_call_field_end"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly path: readonly string[]; readonly value: JsonValue }
  | { readonly _tag: "tool_call_ready"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId }
  | StreamEnd
