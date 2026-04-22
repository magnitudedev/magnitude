/**
 * ops.ts — ParserOp type alias and helper constructors.
 *
 * All handler effects — stack changes, events, errors — go through the op system.
 * Handlers return ParserOp[] arrays; the parser loop applies them via machine.apply().
 * Handlers are pure functions with no direct side effects.
 *
 * Helper constructors (emitEvent, emitStructuralError, emitToolError) reduce
 * boilerplate in handler implementations.
 */

import type { Op } from '../machine'
import type { Frame } from './types'
import type {
  TurnEngineEvent,
  StructuralParseError,
  ToolParseError,
  StructuralParseErrorEvent,
  ToolParseErrorEvent,
} from '../types'

// =============================================================================
// ParserOp — the op type for the parser's stack machine
// =============================================================================

/**
 * ParserOp — Op<Frame, TurnEngineEvent>.
 *
 * The full set of ops the parser loop applies to the stack machine:
 *   push    — push a new frame
 *   pop     — pop the top frame
 *   replace — replace the top frame (used for immutable frame updates)
 *   emit    — emit a TurnEngineEvent to downstream consumers
 *   done    — signal turn completion
 *   observe — signal yield/observe (turn control)
 */
export type ParserOp = Op<Frame, TurnEngineEvent>

// =============================================================================
// Helper constructors
// =============================================================================

/**
 * emitEvent — wrap a TurnEngineEvent in an emit op.
 *
 * Usage:
 *   return [emitEvent({ _tag: 'LensStart', name: 'alignment' }), { type: 'push', frame: ... }]
 */
export const emitEvent = (event: TurnEngineEvent): ParserOp =>
  ({ type: 'emit', event })

/**
 * emitStructuralError — wrap a StructuralParseError in a StructuralParseErrorEvent emit op.
 *
 * Structural errors are not routable to any specific tool. They indicate problems
 * with the turn's overall structure (stray tags, unclosed elements, etc.).
 *
 * Usage:
 *   return [emitStructuralError({ _tag: 'StrayCloseTag', tagName: 'invoke', detail: '...' })]
 */
export const emitStructuralError = (error: StructuralParseError): ParserOp =>
  ({
    type: 'emit',
    event: { _tag: 'StructuralParseError', error } satisfies StructuralParseErrorEvent,
  })

/**
 * emitToolError — wrap a ToolParseError in a ToolParseErrorEvent emit op.
 *
 * Tool errors are routable to a specific tool call via toolCallId.
 * The context fields (tagName, toolName, group, correctToolShape) are included
 * so consumers can surface well-formed error messages.
 *
 * Usage:
 *   return [emitToolError(
 *     { _tag: 'UnknownParameter', toolCallId, tagName, parameterName, detail },
 *     { toolCallId, tagName, toolName, group },
 *   )]
 */
export const emitToolError = (
  error: ToolParseError,
  context: {
    readonly toolCallId: string
    readonly tagName: string
    readonly toolName: string
    readonly group: string
    readonly correctToolShape?: string
  },
): ParserOp =>
  ({
    type: 'emit',
    event: {
      _tag: 'ToolParseError',
      error,
      ...context,
    } satisfies ToolParseErrorEvent,
  })
