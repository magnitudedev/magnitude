/**
 * Visual Reducer Registry
 *
 * Module-level singleton that holds per-tool visual reducers.
 * The DisplayProjection reads this to reduce visual state as ToolCallEvents stream in.
 * The CLI sets it at startup via `setVisualRegistry()`.
 *
 * This lives in packages/agent (not CLI) so the projection can import it without
 * creating a circular dependency. Renderers stay in CLI.
 */

import type { ToolCallEvent } from '@magnitudedev/xml-act'

// =============================================================================
// Types
// =============================================================================

/** Reducer-only visual definition for a tool (no rendering).
 * Not generic — state is `unknown` at the registry/projection boundary.
 * Type safety is enforced at definition time via `defineToolReducer`. */
export interface ToolVisualReducer {
  readonly toolKey: string
  readonly cluster?: string
  readonly initial: unknown
  readonly reduce: (state: unknown, event: ToolCallEvent) => unknown
}

/** Registry mapping toolKey → reducer. */
export interface VisualReducerRegistry {
  readonly get: (toolKey: string) => ToolVisualReducer | undefined
}

// =============================================================================
// Singleton
// =============================================================================

let _registry: VisualReducerRegistry | null = null

/** Set the visual reducer registry. Called once at startup by the CLI layer. */
export function setVisualRegistry(registry: VisualReducerRegistry): void {
  _registry = registry
}

/** Get the visual reducer registry. Returns null if not set. */
export function getVisualRegistry(): VisualReducerRegistry | null {
  return _registry
}
