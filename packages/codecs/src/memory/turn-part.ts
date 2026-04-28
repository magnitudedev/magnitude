/**
 * TurnPart — plain TypeScript discriminated union representing one part of an
 * assistant turn in memory.
 *
 * Three variants:
 *   thought   — internal reasoning block (never shown to the user directly)
 *   message   — user-facing text response emitted by the model
 *   tool_call — a single tool invocation with fully-parsed input
 *
 * Plain TS (no Schema) because this is internal projection state. Producers
 * construct it with statically-known shapes; no untrusted boundary validation
 * is needed.
 */

export type ThoughtPart = {
  readonly type:  'thought'
  readonly id:    string
  readonly level: 'low' | 'medium' | 'high'
  readonly text:  string
}

export type MessagePart = {
  readonly type: 'message'
  readonly id:   string
  readonly text: string
}

export type ToolCallPart = {
  readonly type:     'tool_call'
  readonly id:       string
  readonly toolName: string
  readonly input:    unknown
}

export type TurnPart = ThoughtPart | MessagePart | ToolCallPart
