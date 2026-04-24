/**
 * Core Types for the format runtime.
 */

import { Context, Effect, Layer } from "effect"
import type { ToolDefinition, ContentPart, DeepPaths, StreamingPartial } from "@magnitudedev/tools"

export type { DeepPaths, StreamingPartial }

// =============================================================================
// Token Types
// =============================================================================

/**
 * Token types for the streaming tokenizer.
 * Uses asymmetric delimiters: <|tag> to open, <tag|> to close.
 */
export type Token =
  | { readonly _tag: 'Open';      readonly tagName: string; readonly attrs: ReadonlyMap<string, string>; readonly afterNewline: boolean; readonly raw?: string }
  | { readonly _tag: 'Close';     readonly tagName: string; readonly afterNewline: boolean; readonly raw?: string }
  | { readonly _tag: 'SelfClose'; readonly tagName: string; readonly attrs: ReadonlyMap<string, string>; readonly afterNewline: boolean; readonly raw?: string }
  | { readonly _tag: 'Content';   readonly text: string }

// =============================================================================
// Parse Event Types (from parser)
// =============================================================================

export interface ParameterStarted {
  readonly _tag: 'ParameterStarted'
  readonly toolCallId: string
  readonly parameterName: string
}

export interface ParameterChunk {
  readonly _tag: 'ParameterChunk'
  readonly toolCallId: string
  readonly parameterName: string
  readonly text: string
}

export interface ParameterComplete {
  readonly _tag: 'ParameterComplete'
  readonly toolCallId: string
  readonly parameterName: string
  readonly value: string
}

export interface FilterStarted {
  readonly _tag: 'FilterStarted'
  readonly toolCallId: string
  readonly filterType: string
}

export interface FilterChunk {
  readonly _tag: 'FilterChunk'
  readonly toolCallId: string
  readonly text: string
}

export interface FilterComplete {
  readonly _tag: 'FilterComplete'
  readonly toolCallId: string
  readonly query: string
}

export interface InvokeStarted {
  readonly _tag: 'InvokeStarted'
  readonly toolCallId: string
  readonly toolTag: string
  readonly toolName: string
  readonly group: string
}

export interface InvokeComplete {
  readonly _tag: 'InvokeComplete'
  readonly toolCallId: string
  readonly hasFilter: boolean
}

/**
 * Union of parse events from the parser (format-specific lifecycle).
 */
export type ParseEvent =
  | ParameterStarted
  | ParameterChunk
  | ParameterComplete
  | FilterStarted
  | FilterChunk
  | FilterComplete
  | InvokeStarted
  | InvokeComplete

// =============================================================================
// Structural Event Types (from parser)
// =============================================================================

export interface ProseChunk {
  readonly _tag: 'ProseChunk'
  readonly text: string
}

export interface ProseEnd {
  readonly _tag: 'ProseEnd'
  readonly content: string
}

export interface LensStart { readonly _tag: 'LensStart'; readonly name: string }
export interface LensChunk { readonly _tag: 'LensChunk'; readonly text: string }
export interface LensEnd { readonly _tag: 'LensEnd'; readonly name: string; readonly content: string }

export interface MessageStart {
  readonly _tag: 'MessageStart'
  readonly id: string
  readonly to: string | null
}
export interface MessageChunk { readonly _tag: 'MessageChunk'; readonly id: string; readonly text: string }
export interface MessageEnd { readonly _tag: 'MessageEnd'; readonly id: string }

export interface TurnControl {
  readonly _tag: 'TurnControl'
  readonly target: 'user' | 'invoke' | 'worker' | 'parent'
  readonly termination: 'natural' | 'runaway'
}

export type StructuralEvent =
  | ProseChunk
  | ProseEnd
  | LensStart
  | LensChunk
  | LensEnd
  | MessageStart
  | MessageChunk
  | MessageEnd
  | TurnControl

// =============================================================================
// Tool Registration
// =============================================================================

export interface RegisteredTool {
  readonly tool: ToolDefinition
  readonly tagName: string
  readonly groupName: string
  readonly meta?: unknown
  readonly layerProvider?: () => Effect.Effect<Layer.Layer<never>, unknown>
}

// =============================================================================
// Runtime Event Types (format-agnostic)
// =============================================================================

export interface ToolCallContext {
  readonly toolCallId: string
  readonly tagName: string
  readonly toolName: string
  readonly group: string
}

export interface ToolInputStarted {
  readonly _tag: 'ToolInputStarted'
  readonly toolCallId: string
  readonly tagName: string
  readonly toolName: string
  readonly group: string
}



/**
 * Streaming chunk for a parameter field.
 * path: type-safe path into TInput (e.g. ["config", "tls", "cert"] for nested JSON)
 * delta: raw incremental text — consumer applies this to their own StreamingPartial via applyFieldChunk
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
 * value: coerced typed value for this specific field
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

/** Tool-scoped parse error event — used in ToolLifecycleEvent */
export interface ToolParseErrorEvent {
  readonly _tag: 'ToolParseError'
  readonly toolCallId: string
  readonly tagName: string
  readonly toolName: string
  readonly group: string
  readonly error: ToolParseError
  readonly correctToolShape?: string
}

export interface ToolExecutionStarted<TInput = unknown> {
  readonly _tag: 'ToolExecutionStarted'
  readonly toolCallId: string
  readonly tagName: string
  readonly group: string
  readonly toolName: string
  readonly input: TInput
  readonly cached: boolean
}

export interface ToolExecutionEnded<TOutput = unknown> {
  readonly _tag: 'ToolExecutionEnded'
  readonly toolCallId: string
  readonly tagName: string
  readonly group: string
  readonly toolName: string
  readonly result: ToolResult<TOutput>
}

export interface ToolEmission<TEmission = unknown> {
  readonly _tag: 'ToolEmission'
  readonly toolCallId: string
  readonly value: TEmission
}

export interface ToolObservation {
  readonly _tag: 'ToolObservation'
  readonly toolCallId: string
  readonly tagName: string
  readonly query: string | null
  readonly content: ContentPart[]
}

export interface TurnEnd {
  readonly _tag: 'TurnEnd'
  readonly result: ExecutionResult
}

// =============================================================================
// Result Types
// =============================================================================

export type ToolResult<TOutput = unknown> =
  | { readonly _tag: 'Success'; readonly output: TOutput; readonly query: string | null }
  | { readonly _tag: 'Error'; readonly error: string }
  | { readonly _tag: 'Rejected'; readonly rejection: unknown }
  | { readonly _tag: 'Interrupted' }

export type ExecutionResult =
  | { readonly _tag: 'Success'; readonly turnControl: { readonly target: 'user' | 'invoke' | 'worker' | 'parent' } | null; readonly termination: 'natural' | 'runaway' }
  | { readonly _tag: 'Failure'; readonly error: string }
  | { readonly _tag: 'Interrupted' }
  | { readonly _tag: 'GateRejected'; readonly rejection: unknown }

// =============================================================================
// Error Types
// =============================================================================

export class TurnEngineCrash {
  readonly _tag = 'TurnEngineCrash'
  constructor(readonly message: string, readonly cause?: unknown) {}
}

// =============================================================================
// Parse Error Detail Variants
// =============================================================================

// --- Tool-scoped (routable to a tool state model) ---

export interface UnknownParameterError {
  readonly _tag: 'UnknownParameter'
  readonly toolCallId: string
  readonly tagName: string
  readonly parameterName: string
  readonly detail: string
}

export interface IncompleteToolError {
  readonly _tag: 'IncompleteTool'
  readonly toolCallId: string
  readonly tagName: string
  readonly detail: string
}

export interface JsonStructuralError {
  readonly _tag: 'JsonStructuralError'
  readonly toolCallId: string
  readonly tagName: string
  readonly parameterName: string
  readonly detail: string
}

export interface SchemaCoercionError {
  readonly _tag: 'SchemaCoercionError'
  readonly toolCallId: string
  readonly tagName: string
  readonly parameterName: string
  readonly detail: string
}

export interface MissingRequiredFieldError {
  readonly _tag: 'MissingRequiredField'
  readonly toolCallId: string
  readonly tagName: string
  readonly parameterName: string
  readonly detail: string
}

/** Tool-scoped parse errors — routable to a tool state model */
export interface DuplicateParameterError {
  readonly _tag: 'DuplicateParameter'
  readonly toolCallId: string
  readonly tagName: string
  readonly parameterName: string
  readonly detail: string
}

export type ToolParseError =
  | UnknownParameterError
  | DuplicateParameterError
  | IncompleteToolError
  | JsonStructuralError
  | SchemaCoercionError
  | MissingRequiredFieldError

// --- Structural (not routable to any tool) ---

export interface UnknownToolError {
  readonly _tag: 'UnknownTool'
  readonly tagName: string
  readonly detail: string
}

export interface MalformedTagError {
  readonly _tag: 'MalformedTag'
  readonly tagName: string
  readonly detail: string
}

export interface UnexpectedContentError {
  readonly _tag: 'UnexpectedContent'
  readonly context: string
  readonly detail: string
}

export interface UnclosedThinkError {
  readonly _tag: 'UnclosedThink'
  readonly message: string
}

export interface StrayCloseTagError {
  readonly _tag: 'StrayCloseTag'
  readonly tagName: string
  readonly detail: string
}

export interface MissingToolNameError {
  readonly _tag: 'MissingToolName'
  readonly detail: string
}

export interface InvalidMagnitudeOpenError {
  readonly _tag: 'InvalidMagnitudeOpen'
  readonly tagName: string
  readonly parentTagName: string | undefined
  readonly raw: string
  readonly detail: string
}

export interface AmbiguousMagnitudeCloseError {
  readonly _tag: 'AmbiguousMagnitudeClose'
  readonly tagName: string
  readonly expectedTagName: string | undefined
  readonly raw: string
  readonly detail: string
}

/** Structural parse errors — not routable to any tool */
export type StructuralParseError =
  | UnknownToolError
  | MalformedTagError
  | UnexpectedContentError
  | UnclosedThinkError
  | StrayCloseTagError
  | MissingToolNameError
  | InvalidMagnitudeOpenError
  | AmbiguousMagnitudeCloseError

/** All parse error details */
export type ParseErrorDetail =
  | ToolParseError
  | StructuralParseError

// =============================================================================
// ParseError Events
// =============================================================================

/** Structural parse error event */
export interface StructuralParseErrorEvent {
  readonly _tag: 'StructuralParseError'
  readonly error: StructuralParseError
}



// =============================================================================
// FilterReady (internal parser event, not in TurnEngineEvent)
// =============================================================================

export interface FilterReady {
  readonly _tag: 'FilterReady'
  readonly toolCallId: string
  readonly query: string
}

export type InternalParserEvent = FilterReady

// =============================================================================
// Interceptor
// =============================================================================

export interface ToolInterceptor {
  readonly beforeExecute: (ctx: InterceptorContext) => Effect.Effect<InterceptorDecision>
  readonly afterExecute?: (ctx: InterceptorContext & { readonly result: unknown }) => Effect.Effect<InterceptorDecision>
}

export interface InterceptorContext {
  readonly toolCallId: string
  readonly tagName: string
  readonly group: string
  readonly toolName: string
  readonly input: unknown
  readonly meta: unknown
}

export type InterceptorDecision =
  | { readonly _tag: 'Proceed'; readonly modifiedInput?: unknown }
  | { readonly _tag: 'Reject'; readonly rejection: unknown }

export class ToolInterceptorTag extends Context.Tag('ToolInterceptor')<
  ToolInterceptorTag, ToolInterceptor
>() {}

// =============================================================================
// Reactor State / Engine State
// =============================================================================

export type ToolOutcome =
  | { readonly _tag: 'Completed'; readonly result: ToolResult }
  | { readonly _tag: 'ParseError' }



/** Replaces ReactorState. Owned by the turn engine. */
export interface EngineState {
  readonly toolCallMap: ReadonlyMap<string, string>
  readonly toolOutcomes: ReadonlyMap<string, ToolOutcome>
  readonly deadToolCalls: ReadonlySet<string>
  readonly stopped: boolean
}

// =============================================================================
// Tool Lifecycle Event (for state model consumers)
// =============================================================================

/**
 * Subset of TurnEngineEvent relevant to tool state.
 * Used by StateModel.reduce() — consumers get full type safety via TInput/TOutput/TEmission.
 * No information loss — just type narrowing (structural events excluded).
 */
export type ToolLifecycleEvent<TInput = unknown, TOutput = unknown, TEmission = unknown> =
  | ToolInputStarted
  | ToolInputFieldChunk<TInput>
  | ToolInputFieldComplete<TInput>
  | ToolInputReady<TInput>
  | ToolParseErrorEvent
  | ToolExecutionStarted<TInput>
  | ToolExecutionEnded<TOutput>
  | ToolEmission<TEmission>

/**
 * Apply a ToolInputFieldChunk to an existing StreamingPartial, returning the updated partial.
 * Consumers (state models, projections) call this in their reduce() to build streaming input state.
 * Implementation lives in engine/input-builder.ts — exported from index.ts.
 */
export type ApplyFieldChunk = <TInput>(
  partial: StreamingPartial<TInput>,
  path: DeepPaths<TInput>,
  text: string
) => StreamingPartial<TInput>

// =============================================================================
// Runtime Event Union
// =============================================================================

export type TurnEngineEvent<TInput = unknown, TOutput = unknown, TEmission = unknown> =
  | ToolInputStarted
  | ToolInputFieldChunk<TInput>
  | ToolInputFieldComplete<TInput>
  | ToolInputReady<TInput>
  | ToolParseErrorEvent
  | StructuralParseErrorEvent
  | ToolExecutionStarted<TInput>
  | ToolExecutionEnded<TOutput>
  | ToolEmission<TEmission>
  | ToolObservation
  | ProseChunk | ProseEnd
  | LensStart | LensChunk | LensEnd
  | MessageStart | MessageChunk | MessageEnd
  | TurnEnd



// =============================================================================
// Runtime Config
// =============================================================================

export interface RuntimeConfig {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly defaultProseDest?: string
  readonly resultsDir: string
}
