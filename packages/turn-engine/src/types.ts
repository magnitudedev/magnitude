/**
 * Turn engine type vocabulary.
 *
 * Format-neutral event language produced by the engine.
 * Native codec response events are translated into this vocabulary
 * inside the engine runtime.
 *
 * Phase 1: types only. Engine logic lands in Phase 2.
 */

import { Context, Effect, Layer } from "effect"
import type { ResponseUsage } from "@magnitudedev/codecs"
import type { ToolDefinition, ContentPart, DeepPaths, StreamingPartial } from "@magnitudedev/tools"

export type { ContentPart, DeepPaths, StreamingPartial }

// =============================================================================
// Source spans (optional — xml-act provides; native sets undefined)
// =============================================================================

export interface SourcePos {
  readonly offset: number
  readonly line: number
  readonly col: number
}

export interface SourceSpan {
  readonly start: SourcePos
  readonly end: SourcePos
}

// =============================================================================
// Tool registration
// =============================================================================

/**
 * RegisteredTool — a tool wrapped with the metadata the runtime needs to
 * dispatch it.
 *
 * Generic over `R` — the services the tool's execute requires from the
 * effect context. The `layerProvider` returns a Layer that provides
 * exactly those services, so the dispatcher can fully resolve the tool
 * effect's R channel without losing or fabricating type information.
 *
 * Default `R = never` keeps every existing call site (tools with no
 * service requirements, callers passing `Layer.empty`) typing unchanged.
 */
export interface RegisteredTool<R = never> {
  readonly tool: ToolDefinition
  /** Single canonical identifier for the tool. xml-act uses this as the tag name; native uses it as the function name. */
  readonly toolName: string
  readonly groupName: string
  readonly meta?: unknown
  readonly layerProvider?: () => Effect.Effect<Layer.Layer<R, never, never>, unknown>
}

// =============================================================================
// Reasoning / messaging events
// =============================================================================

/**
 * ThoughtStart — opens a reasoning block.
 *
 * `kind` is opaque metadata. xml-act sets it from the `about=` attribute
 * (e.g. `"alignment"`, `"tasks"`); native codec sets a default like
 * `"reasoning"`. Engine forwards the value but does not interpret it.
 */
export interface ThoughtStart {
  readonly _tag: 'ThoughtStart'
  readonly kind: string
}

export interface ThoughtChunk {
  readonly _tag: 'ThoughtChunk'
  readonly text: string
}

export interface ThoughtEnd {
  readonly _tag: 'ThoughtEnd'
}

export interface MessageStart {
  readonly _tag: 'MessageStart'
  readonly id: string
  readonly to: string
}

export interface MessageChunk {
  readonly _tag: 'MessageChunk'
  readonly id: string
  readonly text: string
}

export interface MessageEnd {
  readonly _tag: 'MessageEnd'
  readonly id: string
}

// =============================================================================
// Tool input lifecycle
// =============================================================================

export interface ToolCallContext {
  readonly toolCallId: string
  readonly toolName: string
  readonly group: string
}

export interface ToolInputStarted {
  readonly _tag: 'ToolInputStarted'
  readonly toolCallId: string
  readonly toolName: string
  readonly group: string
  readonly openSpan?: SourceSpan
}

/**
 * Streaming chunk for a parameter field.
 * `path` identifies the field (top-level or nested). `delta` is raw incremental text;
 * consumers apply this to their own StreamingPartial via `applyFieldChunk`.
 */
export interface ToolInputFieldChunk<TInput = unknown> {
  readonly _tag: 'ToolInputFieldChunk'
  readonly toolCallId: string
  readonly field: string & keyof TInput
  readonly path: DeepPaths<TInput>
  readonly delta: string
}

/**
 * Final value for a completed parameter field.
 */
export interface ToolInputFieldComplete<TInput = unknown> {
  readonly _tag: 'ToolInputFieldComplete'
  readonly toolCallId: string
  readonly field: string & keyof TInput
  readonly path: DeepPaths<TInput>
  readonly value: unknown
}

export interface ToolInputReady<TInput = unknown> {
  readonly _tag: 'ToolInputReady'
  readonly toolCallId: string
  readonly input: TInput
}

// =============================================================================
// Tool execution lifecycle
// =============================================================================

export interface ToolExecutionStarted<TInput = unknown> {
  readonly _tag: 'ToolExecutionStarted'
  readonly toolCallId: string
  readonly toolName: string
  readonly group: string
  readonly input: TInput
  readonly cached: boolean
}

export interface ToolExecutionEnded<TOutput = unknown> {
  readonly _tag: 'ToolExecutionEnded'
  readonly toolCallId: string
  readonly toolName: string
  readonly group: string
  readonly result: ToolResult<TOutput>
}

export interface ToolEmission<TEmission = unknown> {
  readonly _tag: 'ToolEmission'
  readonly toolCallId: string
  readonly value: TEmission
}

// =============================================================================
// Decode failures
//
// Format-layer (xml-act translator, codec adapter) reports something it
// could not decode. The engine treats these as routable failures:
//   - ToolInputDecodeFailure: scoped to a specific tool call. Engine may
//     emit a synthetic ToolExecutionEnded { result: Error } so consumers
//     see a uniform tool-failed lifecycle, then terminate the turn with
//     ToolInputDecodeFailure outcome.
//   - TurnStructureDecodeFailure: not attributable to any tool. Engine
//     terminates the turn with TurnStructureDecodeFailure outcome.
//
// `detail` is opaque to the engine — formats put their structured error
// payload here.
// =============================================================================

export interface ToolInputDecodeFailure {
  readonly _tag: 'ToolInputDecodeFailure'
  readonly toolCallId: string
  readonly toolName: string
  readonly group: string
  readonly detail: unknown
}

export interface TurnStructureDecodeFailure {
  readonly _tag: 'TurnStructureDecodeFailure'
  readonly detail: unknown
}

// =============================================================================
// Turn end
// =============================================================================

export interface TurnControl {
  readonly _tag: 'TurnControl'
  readonly target: 'user' | 'invoke' | 'worker' | 'parent'
  readonly termination: 'natural' | 'runaway'
}

export interface TurnEnd {
  readonly _tag: 'TurnEnd'
  readonly outcome: TurnEngineOutcome
  readonly usage: ResponseUsage
}

export type SafetyStopReason =
  | { readonly _tag: 'IdenticalResponseCircuitBreaker'; readonly threshold: number }
  | { readonly _tag: 'Other'; readonly message: string }

export type TurnEngineOutcome =
  | { readonly _tag: 'Completed'; readonly toolCallsCount: number }
  | { readonly _tag: 'OutputTruncated' }
  | { readonly _tag: 'ContentFiltered' }
  | { readonly _tag: 'SafetyStop'; readonly reason: SafetyStopReason }
  | { readonly _tag: 'ToolInputDecodeFailure'; readonly toolCallId: string; readonly toolName: string; readonly detail: unknown }
  | { readonly _tag: 'TurnStructureDecodeFailure'; readonly detail: unknown }
  | { readonly _tag: 'GateRejected'; readonly toolCallId: string; readonly toolName: string }
  | { readonly _tag: 'EngineDefect'; readonly message: string }

// =============================================================================
// Result types
// =============================================================================

export type ToolResult<TOutput = unknown> =
  | { readonly _tag: 'Success'; readonly output: TOutput }
  | { readonly _tag: 'Error'; readonly error: string }
  | { readonly _tag: 'Rejected'; readonly rejection: unknown }
  | { readonly _tag: 'Interrupted' }

export type ToolOutcome =
  | { readonly _tag: 'Completed'; readonly result: ToolResult }
  | { readonly _tag: 'DecodeFailure' }

// =============================================================================
// Engine state
// =============================================================================

/** Owned by the turn engine; folded over the event stream. */
export interface EngineState {
  readonly toolCallMap: ReadonlyMap<string, string>
  readonly toolOutcomes: ReadonlyMap<string, ToolOutcome>
  readonly deadToolCalls: ReadonlySet<string>
  readonly stopped: boolean
}

// =============================================================================
// Errors
// =============================================================================

export class TurnEngineCrash {
  readonly _tag = 'TurnEngineCrash'
  constructor(readonly message: string, readonly cause?: unknown) {}
}

// =============================================================================
// Interceptor
// =============================================================================

export interface InterceptorContext {
  readonly toolCallId: string
  readonly toolName: string
  readonly group: string
  readonly input: unknown
  readonly meta: unknown
}

export type InterceptorDecision =
  | { readonly _tag: 'Proceed'; readonly modifiedInput?: unknown }
  | { readonly _tag: 'Reject'; readonly rejection: unknown }

export interface ToolInterceptor {
  readonly beforeExecute: (ctx: InterceptorContext) => Effect.Effect<InterceptorDecision>
  readonly afterExecute?: (ctx: InterceptorContext & { readonly result: unknown }) => Effect.Effect<InterceptorDecision>
}

export class ToolInterceptorTag extends Context.Tag('ToolInterceptor')<
  ToolInterceptorTag, ToolInterceptor
>() {}

// =============================================================================
// ToolLifecycleEvent — narrowed view for state-model consumers
// =============================================================================

export type ToolLifecycleEvent<TInput = unknown, TOutput = unknown, TEmission = unknown> =
  | ToolInputStarted
  | ToolInputFieldChunk<TInput>
  | ToolInputFieldComplete<TInput>
  | ToolInputReady<TInput>
  | ToolInputDecodeFailure
  | ToolExecutionStarted<TInput>
  | ToolExecutionEnded<TOutput>
  | ToolEmission<TEmission>

/**
 * Apply a ToolInputFieldChunk to an existing StreamingPartial, returning the updated partial.
 * Implementation lives in input-builder.ts (Phase 2) — exported from index.ts.
 */
export type ApplyFieldChunk = <TInput>(
  partial: StreamingPartial<TInput>,
  path: DeepPaths<TInput>,
  text: string
) => StreamingPartial<TInput>

// =============================================================================
// TurnEngineEvent — top-level output event union
// =============================================================================

export type TurnEngineEvent<TInput = unknown, TOutput = unknown, TEmission = unknown> =
  | ThoughtStart
  | ThoughtChunk
  | ThoughtEnd
  | MessageStart
  | MessageChunk
  | MessageEnd
  | ToolInputStarted
  | ToolInputFieldChunk<TInput>
  | ToolInputFieldComplete<TInput>
  | ToolInputReady<TInput>
  | ToolInputDecodeFailure
  | TurnStructureDecodeFailure
  | ToolExecutionStarted<TInput>
  | ToolExecutionEnded<TOutput>
  | ToolEmission<TEmission>
  | TurnEnd

// =============================================================================
// Runtime config
// =============================================================================

export interface RuntimeConfig {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly messageDestination: string
  readonly thoughtKind?: string
}
