/**
 * Parser types — internal frame types, ParserConfig, and field state.
 */

import type { RegisteredTool } from '../types'
import type { StreamingJsonParser } from '../jsonish/types'

// =============================================================================
// ParserConfig
// =============================================================================

/**
 * Configuration for creating a new mact parser.
 * The parser needs the tool registry to validate invocations and parameters at parse time.
 */
export interface ParserConfig {
  /**
   * Tool registry — maps tagName → RegisteredTool.
   * Used to validate tool names and parameter names at parse time.
   */
  readonly tools: ReadonlyMap<string, RegisteredTool>

  /**
   * ID generator — used for tool call IDs, message IDs.
   * If not provided, a default counter-based generator is used.
   * Pass a custom generator for replay (to reproduce prior IDs in order).
   */
  readonly generateId?: () => string

  /**
   * Default destination for prose content (when not inside a message tag).
   */
  readonly defaultProseDest?: string
}

// =============================================================================
// Frame Types (parser-internal)
// =============================================================================

export type Frame =
  | ProseFrame
  | ThinkFrame
  | MessageFrame
  | InvokeFrame
  | ParameterFrame
  | FilterFrame

export interface ProseFrame {
  readonly type: 'prose'
  readonly body: string
  readonly pendingNewlines: number
  readonly hasContent: boolean
}

export interface ThinkFrame {
  readonly type: 'think'
  readonly name: string
  readonly content: string
  readonly hasContent: boolean
  readonly pendingNewlines: number
}

export interface MessageFrame {
  readonly type: 'message'
  readonly id: string
  readonly to: string | null
  readonly content: string
  readonly pendingNewlines: number
}

export interface InvokeFrame {
  readonly type: 'invoke'
  readonly toolCallId: string
  readonly toolTag: string
  readonly toolName: string
  readonly group: string
  /** Whether this tool was found in the registry */
  readonly known: boolean
  /** Whether this invoke is dead (unknown tool or fatal error) */
  readonly dead: boolean
  /** Whether a filter has been started */
  readonly hasFilter: boolean
  /** Per-field state */
  readonly fieldStates: Map<string, FieldState>
  /** Parameters seen so far (for duplicate detection) */
  readonly seenParams: Set<string>
}

export interface ParameterFrame {
  readonly type: 'parameter'
  readonly toolCallId: string
  readonly paramName: string
  /** Whether this parameter frame is dead (unknown param, duplicate, etc.) */
  readonly dead: boolean
  /** Accumulated raw text value */
  rawValue: string
  /** Jsonish parser for JSON-type fields, null for scalar fields */
  readonly jsonishParser: StreamingJsonParser | null
  /** Parameter type derived from schema */
  readonly fieldType: FieldType
}

export interface FilterFrame {
  readonly type: 'filter'
  readonly toolCallId: string
  readonly filterType: string
  query: string
}

// =============================================================================
// Field Types
// =============================================================================

export type FieldType = 'string' | 'number' | 'boolean' | 'json' | 'unknown'

export interface FieldState {
  readonly paramName: string
  rawValue: string
  coercedValue: unknown
  errored: boolean
  errorDetail: string | undefined
  complete: boolean
}
