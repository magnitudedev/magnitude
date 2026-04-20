/**
 * Core Types for the format runtime.
 */

import { Context, Effect, Layer } from "effect"
import type { ToolDefinition, ContentPart } from "@magnitudedev/tools"

// =============================================================================
// Token Types
// =============================================================================

/**
 * Token types for the streaming tokenizer.
 * Uses asymmetric delimiters: <|tag> to open, <tag|> to close.
 */
export type Token =
  | { readonly _tag: 'Open'; readonly name: string; readonly variant?: string }
  | { readonly _tag: 'Close'; readonly name: string; readonly pipe?: string }
  | { readonly _tag: 'SelfClose'; readonly name: string; readonly variant?: string }
  | { readonly _tag: 'Parameter'; readonly name: string }
  | { readonly _tag: 'ParameterClose' }
  | { readonly _tag: 'Content'; readonly text: string }

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
  | ParseError

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
  readonly target: 'user' | 'tool' | 'worker' | 'parent'
  readonly termination: 'natural' | 'runaway'
}

export interface ParseError {
  readonly _tag: 'ParseError'
  readonly error: StructuralParseErrorDetail
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

export interface ToolInputFieldValue {
  readonly _tag: 'ToolInputFieldValue'
  readonly toolCallId: string
  readonly field: string
  readonly value: string | number | boolean
}

export interface ToolInputReady<TInput = unknown> {
  readonly _tag: 'ToolInputReady'
  readonly toolCallId: string
  readonly input: TInput
}

export interface ToolInputParseError {
  readonly _tag: 'ToolInputParseError'
  readonly toolCallId: string
  readonly tagName: string
  readonly toolName: string
  readonly group: string
  readonly error: ParseErrorDetail
}

export interface ToolExecutionStarted<TInput = unknown> {
  readonly _tag: 'ToolExecutionStarted'
  readonly toolCallId: string
  readonly group: string
  readonly toolName: string
  readonly input: TInput
  readonly cached: boolean
}

export interface ToolExecutionEnded<TOutput = unknown> {
  readonly _tag: 'ToolExecutionEnded'
  readonly toolCallId: string
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

export interface StructuralParseError {
  readonly _tag: 'StructuralParseError'
  readonly error: StructuralParseErrorDetail
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
  | { readonly _tag: 'Success'; readonly turnControl: { readonly target: 'user' | 'tool' | 'worker' | 'parent' } | null; readonly termination: 'natural' | 'runaway' }
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

export interface ParseErrorDetail {
  readonly _tag: string
  readonly id: string
  readonly tagName: string
  readonly detail: string
}

export interface StructuralParseErrorDetail {
  readonly _tag: string
  readonly message: string
}

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
// Reactor State
// =============================================================================

export type ToolOutcome =
  | { readonly _tag: 'Completed'; readonly result: ToolResult }
  | { readonly _tag: 'ParseError' }

export interface ReactorState {
  readonly toolCallMap: ReadonlyMap<string, string>
  readonly deadToolCalls: ReadonlySet<string>
  readonly stopped: boolean
  readonly toolOutcomes: ReadonlyMap<string, ToolOutcome>
}

// =============================================================================
// Runtime Event Union
// =============================================================================

export type RuntimeEvent<TInput = unknown, TOutput = unknown, TEmission = unknown> =
  | ToolInputStarted
  | ToolInputFieldValue
  | ToolInputReady<TInput>
  | ToolInputParseError
  | ToolExecutionStarted<TInput>
  | ToolExecutionEnded<TOutput>
  | ToolEmission<TEmission>
  | ToolObservation
  | ProseChunk | ProseEnd
  | LensStart | LensChunk | LensEnd
  | MessageStart | MessageChunk | MessageEnd
  | StructuralParseError
  | TurnEnd

// =============================================================================
// Runtime Config
// =============================================================================

export interface RuntimeConfig {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly defaultProseDest?: string
}
