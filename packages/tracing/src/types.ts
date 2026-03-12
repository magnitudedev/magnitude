/**
 * Trace types — standalone, no dependencies on other Magnitude packages.
 */

// Inlined from providers — keeps tracing dependency-free from providers
export type ModelSlot = 'primary' | 'secondary' | 'browser'

export interface CallUsage {
  inputTokens: number | null
  outputTokens: number | null
  cacheReadTokens: number | null
  cacheWriteTokens: number | null
  inputCost: number | null
  outputCost: number | null
  totalCost: number | null
}

export interface CollectorData {
  /** Full HTTP response body from the provider */
  rawResponseBody: unknown | null
  /** Full HTTP request body from the provider */
  rawRequestBody: unknown | null
  /** SSE events for streaming calls */
  sseEvents: unknown[] | null
}

/**
 * Transport-level trace input emitted by the driver.
 * Contains only what the driver knows — no agent-level context.
 */
export interface TraceInput {
  timestamp: string
  model: string | null
  provider: string | null
  slot: ModelSlot
  request: { messages?: unknown[]; input?: unknown }
  response: { rawBody: unknown | null; sseEvents: unknown[] | null; rawOutput?: string }
  usage: CallUsage
  durationMs: number
}

/**
 * Full trace data for an LLM call — driver data enriched with agent context.
 * M is the metadata type — narrow it for type-safe access to call-specific fields.
 */
export interface TraceData<M extends Record<string, unknown> = Record<string, unknown>> extends TraceInput {
  callType: string
  metadata: M
  /** Strategy that produced this trace (null for non-strategy calls like compact/title) */
  strategyId: string | null
  /** Full system prompt text */
  systemPrompt: string | null
}

/**
 * Agent-specific trace metadata — discriminated union on callType.
 */
export type AgentTraceMeta =
  | { callType: 'chat'; forkId: string | null; forkName: string; turnId: string; chainId: string }
  | { callType: 'compact'; forkId: string | null }
  | { callType: 'autopilot'; forkId: string | null }
  | { callType: 'title'; forkId: string | null }
  | { callType: 'extract-memory-diff'; forkId: string | null }

/** A trace emitted by the Magnitude agent — TraceData narrowed with agent metadata */
export type AgentTrace = TraceData<AgentTraceMeta>

/**
 * Trace session metadata written to meta.json
 */
export interface TraceSessionMeta {
  sessionId: string
  created: string
  cwd: string | null
  platform: string | null
  gitBranch: string | null
}
