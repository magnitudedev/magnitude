/**
 * Tool Visual Reducer Definition
 *
 * Provides `defineToolReducer` — a typed identity function that constrains
 * the reduce callback to receive `ToolCallEvent` parameterized with the
 * tool's input, output, and binding types.
 *
 * Reducers live in the agent package so DisplayProjection can reduce visual
 * state without depending on CLI rendering code.
 */

import type { Tool } from '@magnitudedev/tools'
import type { ToolCallEvent } from '@magnitudedev/xml-act'
import type { ToolVisualReducer } from './registry'

/** Config for defineToolReducer — fully typed against a specific tool */
export interface ToolReducerConfig<T extends Tool.Any, TState> {
  /** The tool object — used for type inference only */
  readonly tool: T
  /** Tool key as registered in agent definitions (e.g., 'shell', 'fileRead') */
  readonly toolKey: string
  /** Visual clustering key — consecutive tools with the same cluster share a container */
  readonly cluster?: string
  /** Initial state for new tool calls */
  readonly initial: TState
  /** Typed reducer — event union is narrowed to the tool's specific types */
  readonly reduce: (
    state: TState,
    event: ToolCallEvent<Tool.Input<T>, Tool.Output<T>, Tool.XmlInputBinding<T>>,
  ) => TState
}

/**
 * Define a tool visual reducer with full type safety.
 *
 * The `reduce` callback receives `ToolCallEvent` parameterized with the tool's
 * input, output, and binding types. Impossible event variants (e.g.,
 * ToolInputFieldValue for a body-only tool) resolve to `never` and drop out.
 */
export function defineToolReducer<T extends Tool.Any, TState>(
  config: ToolReducerConfig<T, TState>,
): ToolVisualReducer {
  return {
    toolKey: config.toolKey,
    cluster: config.cluster ?? 'generic',
    initial: config.initial,
    reduce: config.reduce as (state: unknown, event: ToolCallEvent) => unknown,
  }
}

/** Simple reducer config — typed state, unparameterized events. */
export interface SimpleReducerConfig<TState> {
  readonly toolKey: string
  readonly cluster?: string
  readonly initial: TState
  readonly reduce: (state: TState, event: ToolCallEvent) => TState
}

/**
 * Define a tool visual reducer with typed state but unparameterized events.
 * Use this for tools that don't benefit from event type narrowing.
 */
export function reducer<TState>(config: SimpleReducerConfig<TState>): ToolVisualReducer {
  return {
    toolKey: config.toolKey,
    cluster: config.cluster ?? 'generic',
    initial: config.initial,
    reduce: config.reduce as (state: unknown, event: ToolCallEvent) => unknown,
  }
}

// =============================================================================
// Cluster — typed factory for tools that share visual state
// =============================================================================

/**
 * A cluster groups consecutive tool calls under a shared visual state type.
 * All tools in a cluster produce the same TState, enabling cluster renderers
 * to render the group as a unit (e.g., combining consecutive edits to the same file).
 *
 * Usage:
 *   const editCluster = defineCluster<EditState>({ cluster: 'edit', initial: { ... } })
 *   export const editReducer = editCluster.tool(editTool, 'fileEdit', (state, event) => { ... })
 */
export interface ClusterFactory<TState> {
  readonly cluster: string

  /** Add a tool with full event type narrowing (parameterized by tool's input/output/binding). */
  tool<T extends Tool.Any>(
    tool: T,
    toolKey: string,
    reduce: (
      state: TState,
      event: ToolCallEvent<Tool.Input<T>, Tool.Output<T>, Tool.XmlInputBinding<T>>,
    ) => TState,
  ): ToolVisualReducer

  /** Add a tool with simple (unparameterized) events. */
  simpleTool(
    toolKey: string,
    reduce: (state: TState, event: ToolCallEvent) => TState,
  ): ToolVisualReducer
}

export function defineCluster<TState>(config: {
  readonly cluster: string
  readonly initial: TState
}): ClusterFactory<TState> {
  return {
    cluster: config.cluster,

    tool<T extends Tool.Any>(
      _tool: T,
      toolKey: string,
      reduce: (
        state: TState,
        event: ToolCallEvent<Tool.Input<T>, Tool.Output<T>, Tool.XmlInputBinding<T>>,
      ) => TState,
    ): ToolVisualReducer {
      return {
        toolKey,
        cluster: config.cluster,
        initial: config.initial,
        reduce: reduce as (state: unknown, event: ToolCallEvent) => unknown,
      }
    },

    simpleTool(
      toolKey: string,
      reduce: (state: TState, event: ToolCallEvent) => TState,
    ): ToolVisualReducer {
      return {
        toolKey,
        cluster: config.cluster,
        initial: config.initial,
        reduce: reduce as (state: unknown, event: ToolCallEvent) => unknown,
      }
    },
  }
}
