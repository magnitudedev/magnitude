/**
 * Local memory type definitions for use in encode.ts.
 *
 * These mirror the types in packages/agent but are defined here to avoid
 * a circular dependency (codecs → agent). They are lightweight structural
 * types used only for runtime-narrowing of the `unknown[]` memory parameter.
 *
 * When inbox-types is split into its own package (Phase 4), these will be
 * replaced with imports from @magnitudedev/inbox-types.
 */

// ---------------------------------------------------------------------------
// ContentPart (minimal; mirrors packages/tools/src/image-types.ts)
// ---------------------------------------------------------------------------

export type ContentPart =
  | { readonly type: 'text'; readonly text: string }
  | {
      readonly type:      'image'
      readonly base64:    string
      readonly mediaType: string
      readonly width:     number
      readonly height:    number
    }

// ---------------------------------------------------------------------------
// TurnPart (mirrors packages/codecs/src/memory/turn-part.ts)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// ResultEntry / TurnResultItem
// ---------------------------------------------------------------------------

export type ToolObservationResultItem = {
  readonly kind:       'tool_observation'
  readonly toolCallId: string                   // added in Phase 4; required here
  readonly tagName:    string
  readonly content:    readonly ContentPart[]
}

export type ToolErrorResultItem = {
  readonly kind:       'tool_error'
  readonly toolCallId: string                   // added in Phase 4; required here
  readonly tagName:    string
  readonly message?:   string
}

export type OtherResultItem = {
  readonly kind: string
  readonly [key: string]: unknown
}

export type TurnResultItem = ToolObservationResultItem | ToolErrorResultItem | OtherResultItem

export type ResultEntry =
  | { readonly kind: 'turn_results'; readonly items: readonly TurnResultItem[] }
  | { readonly kind: string; readonly [key: string]: unknown }

// ---------------------------------------------------------------------------
// TimelineEntry (minimal subset for encoding)
// ---------------------------------------------------------------------------

export type TimelineEntry =
  | { readonly kind: 'user_message';  readonly text: string; readonly timestamp: number; readonly attachments: readonly unknown[] }
  | { readonly kind: 'parent_message'; readonly text: string; readonly timestamp: number }
  | { readonly kind: 'user_bash_command'; readonly command: string; readonly stdout: string; readonly stderr: string; readonly exitCode: number; readonly timestamp: number; readonly cwd: string }
  | { readonly kind: 'observation';   readonly parts: readonly ContentPart[]; readonly timestamp: number }
  | { readonly kind: 'agent_block';   readonly agentId: string; readonly role: string; readonly atoms: readonly unknown[]; readonly timestamp: number; readonly firstAtomTimestamp: number; readonly lastAtomTimestamp: number }
  | { readonly kind: string;          readonly [key: string]: unknown }

// ---------------------------------------------------------------------------
// Memory Message types
// ---------------------------------------------------------------------------

export type SessionContextMessage = {
  readonly type:    'session_context'
  readonly content: readonly ContentPart[]
}

export type ForkContextMessage = {
  readonly type:    'fork_context'
  readonly content: readonly ContentPart[]
}

export type CompactedMessage = {
  readonly type:    'compacted'
  readonly content: readonly ContentPart[]
}

export type AssistantTurnMessage = {
  readonly type:  'assistant_turn'
  readonly parts: readonly TurnPart[]
}

export type InboxMessage = {
  readonly type:     'inbox'
  readonly results:  readonly ResultEntry[]
  readonly timeline: readonly TimelineEntry[]
}

export type MemoryMessage =
  | SessionContextMessage
  | ForkContextMessage
  | CompactedMessage
  | AssistantTurnMessage
  | InboxMessage

// ---------------------------------------------------------------------------
// Runtime narrowing helpers
// ---------------------------------------------------------------------------

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function hasType(v: unknown, t: string): boolean {
  return isRecord(v) && (v as Record<string, unknown>)['type'] === t
}

export function asMemoryMessage(raw: unknown): MemoryMessage | null {
  if (!isRecord(raw)) return null
  const type = (raw as Record<string, unknown>)['type']
  if (type === 'session_context') return raw as SessionContextMessage
  if (type === 'fork_context')    return raw as ForkContextMessage
  if (type === 'compacted')       return raw as CompactedMessage
  if (type === 'assistant_turn')  return raw as AssistantTurnMessage
  if (type === 'inbox')           return raw as InboxMessage
  return null
}

export function isToolObservation(item: TurnResultItem): item is ToolObservationResultItem {
  return item.kind === 'tool_observation'
}

export function isToolError(item: TurnResultItem): item is ToolErrorResultItem {
  return item.kind === 'tool_error'
}
