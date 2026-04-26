/**
 * handler-context.ts — minimal context passed to all handlers.
 *
 * Intentionally minimal by design:
 *   - No emit functions    — effects go through returned Op[] arrays
 *   - No endCurrentProse  — loop handles prose ending before calling handlers
 *   - No findFrame        — parent frames stored on child frames at open time
 *   - No machine access   — handlers never touch the stack directly
 *   - No apply            — handlers return ops; the loop applies them
 *
 * This keeps handlers as pure functions: given attrs + parent + ctx, return ops.
 * All side effects are deferred to the parser loop via the op system.
 */

import type { RegisteredTool, FilterReady } from '../types'
import type { deriveParameters } from '../engine/parameter-schema'

// =============================================================================
// InvokeContext — tool registry access for invoke/parameter/filter handlers
// =============================================================================

/**
 * InvokeContext — schema and registry access for invoke-related handlers.
 *
 * onFilterReady remains as a callback (not an op) because it is an out-of-band
 * notification to the engine, not a turn event in the TurnEngineEvent union.
 * It is the only side effect that legitimately stays outside the op system.
 */
export interface InvokeContext {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly toolSchemas: ReadonlyMap<string, ReturnType<typeof deriveParameters>>
  readonly onFilterReady?: (event: FilterReady) => void
}

// =============================================================================
// HandlerContext
// =============================================================================

/**
 * HandlerContext — everything a handler needs beyond what's captured in its closure.
 *
 * Passed to every handler's open/close/selfClose method.
 * Handlers must not mutate this object.
 */
export interface HandlerContext {
  /**
   * Generate a unique ID for a new tool call or message.
   * Injected so tests can provide deterministic IDs.
   */
  generateId(): string

  /**
   * Tool registry and schema access for invoke-related handlers.
   * Think, message, and prose handlers do not use this.
   */
  invokeCtx: InvokeContext
}
