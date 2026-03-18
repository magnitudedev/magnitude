/**
 * Tool Visual System — CLI Render Definitions
 *
 * The agent package owns state machines (reducers). The CLI only renders
 * pre-reduced state from DisplayProjection. This file defines the render
 * function type and the render registry.
 */

import type { ReactNode } from 'react'

// =============================================================================
// Types
// =============================================================================

/** Props passed to a tool's render function */
export interface ToolRenderProps<TState = unknown> {
  readonly state: TState
  readonly isExpanded: boolean
  readonly onToggle: () => void
  /** The tool result from the ThinkBlockStep (includes display data, rejection reason, etc.) */
  readonly stepResult?: import('@magnitudedev/agent').ToolResult
  /** Callback to open artifact panel (artifact tools only) */
  readonly onArtifactClick?: (name: string, section?: string) => void
}

/** A render function for a specific tool's visual state.
 * Non-generic at the registry boundary; type safety is enforced at
 * definition time via `render()`. */
export type ToolVisualRenderer = (props: ToolRenderProps) => ReactNode

/** Registry mapping toolKey → render function */
export interface RenderRegistry {
  readonly get: (toolKey: string) => ToolVisualRenderer | undefined
}

export type ToolLiveTextGetter = (args: {
  readonly state: unknown
  readonly step?: import('@magnitudedev/agent').ThinkBlockStep
}) => string | null | undefined

export interface LiveTextRegistry {
  readonly get: (toolKey: string) => ToolLiveTextGetter | undefined
}

// =============================================================================
// Helpers
// =============================================================================

/** Define a typed render function and erase to ToolVisualRenderer.
 * The single place where the unknown cast happens, hidden from consumers. */
export function render<TState>(fn: (props: ToolRenderProps<TState>) => ReactNode): ToolVisualRenderer {
  return fn as ToolVisualRenderer
}

// =============================================================================
// Registry builder
// =============================================================================

/** Build a render registry from a record of toolKey → render function. */
export function createRenderRegistry(
  renderers: Record<string, ToolVisualRenderer>,
): RenderRegistry {
  const map = new Map(Object.entries(renderers))
  return {
    get: (toolKey: string) => map.get(toolKey),
  }
}

export function createLiveTextRegistry(
  getters: Record<string, ToolLiveTextGetter>,
): LiveTextRegistry {
  const map = new Map(Object.entries(getters))
  return {
    get: (toolKey: string) => map.get(toolKey),
  }
}

// =============================================================================
// Cluster Rendering
// =============================================================================

/** Step data passed to a cluster renderer — one entry per tool step in the group. */
export interface ClusterStepData<TState = unknown> {
  readonly id: string
  readonly visualState: TState
  readonly result?: import('@magnitudedev/agent').ToolResult
}

/** Props passed to a cluster render function */
export interface ClusterRenderProps<TState = unknown> {
  readonly steps: readonly ClusterStepData<TState>[]
  readonly expandedSteps: ReadonlySet<string>
  readonly onToggleStep: (id: string) => void
  readonly onArtifactClick?: (name: string, section?: string) => void
}

/** A render function for a cluster group.
 * Non-generic at the registry boundary; type safety is enforced at
 * definition time via `clusterRender()`. */
export type ClusterRenderer = (props: ClusterRenderProps) => ReactNode

/** Define a typed cluster render function and erase to ClusterRenderer. */
export function clusterRender<TState>(
  fn: (props: ClusterRenderProps<TState>) => ReactNode,
): ClusterRenderer {
  return fn as ClusterRenderer
}

/** Registry mapping cluster key → cluster render function */
export interface ClusterRenderRegistry {
  readonly get: (cluster: string) => ClusterRenderer | undefined
}

/** Build a cluster render registry from a record of cluster → render function. */
export function createClusterRenderRegistry(
  renderers: Record<string, ClusterRenderer>,
): ClusterRenderRegistry {
  const map = new Map(Object.entries(renderers))
  return {
    get: (cluster: string) => map.get(cluster),
  }
}
