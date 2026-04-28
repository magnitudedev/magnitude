import { Schema } from 'effect'

// =============================================================================
// Completions API — for the completions paradigm (Phase 3+)
// Raw-text completions endpoint (/v1/completions) with GBNF grammar support.
// =============================================================================

/** Full completions request object. */
export interface CompletionsRequest {
  readonly model:           string
  readonly prompt:          string
  readonly grammar?:        string  // GBNF grammar string for constrained decoding
  readonly max_tokens?:     number
  readonly temperature?:    number
  readonly stop?:           readonly string[]
  readonly stream:          true
  readonly stream_options?: { readonly include_usage: boolean }
}

/** A single choice delta in a completions stream chunk. */
const CompletionsChunkChoice = Schema.Struct({
  index:         Schema.Number,
  text:          Schema.String,
  finish_reason: Schema.optional(Schema.NullOr(Schema.String)),
})

const CompletionsChunkUsage = Schema.Struct({
  prompt_tokens:     Schema.Number,
  completion_tokens: Schema.Number,
})

/**
 * CompletionsStreamChunk — the decoded SSE data payload for /v1/completions.
 * Declared for the future completions paradigm; no driver implementation yet.
 */
export class CompletionsStreamChunk extends Schema.Class<CompletionsStreamChunk>(
  'CompletionsStreamChunk',
)({
  id:      Schema.String,
  object:  Schema.String,
  created: Schema.Number,
  model:   Schema.String,
  choices: Schema.Array(CompletionsChunkChoice),
  usage:   Schema.optional(CompletionsChunkUsage),
}) {}
