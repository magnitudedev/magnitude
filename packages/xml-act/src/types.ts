/**
 * xml-act Core Types
 *
 * Event types, result types, configuration, and service tags for the
 * standalone XML tool execution runtime.
 *
 * Event interfaces are parameterized with defaulted generics (TInput, TOutput, B).
 * When unparameterized, fields use fallback types (string, unknown) — identical to
 * pre-parameterization behavior. When parameterized with a concrete tool's types,
 * field names narrow to literal keys, values narrow to actual types, and event
 * variants that can't fire for the given binding resolve to never.
 */

import { Context, Effect, Layer } from "effect"
import type { Tool, XmlBinding, XmlChildBinding, ContentPart } from "@magnitudedev/tools"
import type { OutputNode } from './output-tree'
import type {
  BaseToolParseErrorDetail,
  UnclosedThinkDetail,
  FinishWithoutEvidenceDetail,
  UnclosedActionsDetail,
  TurnControlConflictDetail,
} from './parser/types'

// =============================================================================
// XML Tag Binding (erased tag variant of XmlBinding<T>)
// =============================================================================

/** Runtime-erased tag variant of XmlBinding<T>, without the discriminator. */
export type XmlTagBinding = Omit<Extract<XmlBinding<unknown>, { type: 'tag' }>, 'type'>

export type { XmlChildBinding }

// =============================================================================
// Binding Type-Level Helpers
// =============================================================================

/** Resolve a dotted field path to the value type at that path */
export type ResolvePath<T, P extends string> =
  P extends `${infer H}.${infer R}`
    ? H extends keyof T ? ResolvePath<NonNullable<T[H]>, R> : never
    : P extends keyof T ? T[P] : never

/** Extract attribute binding objects from a binding */
export type BindingAttrs<B> = B extends { readonly attributes: readonly (infer A)[] } ? A : never
/** Extract XML attribute name literals from a binding */
export type BindingAttrNames<B> = BindingAttrs<B> extends { readonly attr: infer A extends string } ? A : never
/** Extract bound field path literals from a binding */
export type BindingAttrFields<B> = BindingAttrs<B> extends { readonly field: infer F extends string } ? F : never
/** Extract body field name literal from a binding */
export type BindingBody<B> = B extends { readonly body: infer F extends string } ? F : never
/** Extract children binding objects from a binding */
export type BindingChildren<B> = B extends { readonly children: readonly (infer C)[] } ? C : never
/** Extract childTags binding objects from a binding */
export type BindingChildTags<B> = B extends { readonly childTags: readonly (infer CT)[] } ? CT : never
/** Extract childRecord binding object from a binding */
export type BindingChildRecord<B> = B extends { readonly childRecord: infer CR } ? CR : never

/** Extract field name from a child binding */
export type ChildBindingField<C> = C extends { readonly field: infer F extends string } ? F : never
/** Extract attribute binding objects from a child binding */
export type ChildBindingAttrs<C> = C extends { readonly attributes: readonly (infer A)[] } ? A : never
/** Extract child XML attribute names */
export type ChildBindingAttrNames<C> = ChildBindingAttrs<C> extends { readonly attr: infer A extends string } ? A : never
/** Extract child bound field names */
export type ChildBindingAttrFields<C> = ChildBindingAttrs<C> extends { readonly field: infer F extends string } ? F : never
/** Extract body field name from a child binding */
export type ChildBindingBody<C> = C extends { readonly body: infer F extends string } ? F : never
/** Extract field name from a childRecord binding */
export type ChildRecordField<CR> = CR extends { readonly field: infer F extends string } ? F : never

/** Resolve the array element type for a child binding's field */
export type ChildElem<TInput, C> =
  ChildBindingField<C> extends infer CF extends string ?
    CF extends keyof TInput ?
      TInput[CF] extends ReadonlyArray<infer E> ? E : never
    : never
  : never

/** Pick only the attribute fields from a child element type */
export type ChildAttrsPick<Elem, C> =
  [ChildBindingAttrFields<C>] extends [never] ? {} : Pick<Elem, ChildBindingAttrFields<C> & keyof Elem>

// =============================================================================
// Tool Call Context + Error (consumer-facing, self-contained)
// =============================================================================

export interface ToolCallContext {
  readonly toolCallId: string
  readonly tagName: string
  readonly toolName: string
  readonly group: string
}

export type ToolCallError = BaseToolParseErrorDetail & { readonly call: ToolCallContext }

// =============================================================================
// Registered Tool
// =============================================================================

/**
 * A tool registered with the runtime, including its XML binding and metadata.
 */
export interface RegisteredTool {
  readonly tool: Tool.Any
  readonly tagName: string
  readonly groupName: string
  readonly binding: XmlTagBinding
  readonly meta?: unknown
  readonly layerProvider?: () => Effect.Effect<Layer.Layer<never>, unknown>
}

// =============================================================================
// Tool Input Events (streaming, before execution)
// =============================================================================

/** Tool call beginning — tag opened, name and group known */
export interface ToolInputStarted {
  readonly _tag: 'ToolInputStarted'
  readonly toolCallId: string
  readonly tagName: string
  readonly toolName: string
  readonly group: string
}

/** Scalar field got its final value (attribute parsed and coerced) */
export interface ToolInputFieldValue<TInput = unknown, B = unknown> {
  readonly _tag: 'ToolInputFieldValue'
  readonly toolCallId: string
  readonly field: [BindingAttrNames<B>] extends [never] ? string : BindingAttrNames<B>
  readonly value: string | number | boolean
}

/** All body field names derivable from a binding — parent body + child body fields */
export type AllBodyFields<B> = BindingBody<B> | ChildBindingBody<BindingChildren<B>>

/** Text chunk for a body field — parent body or child body.
 *  path disambiguates: ['content'] for parent, ['edits', '0', 'content'] for child. */
export interface ToolInputBodyChunk<_TInput = unknown, B = unknown> {
  readonly _tag: 'ToolInputBodyChunk'
  readonly toolCallId: string
  readonly path: readonly string[]
  readonly field: [AllBodyFields<B>] extends [never] ? string : AllBodyFields<B>
  readonly text: string
}

/** Child element opened — its attributes are available.
 *  Fires BEFORE the child's body starts streaming. */
export interface ToolInputChildStarted<TInput = unknown, B = unknown> {
  readonly _tag: 'ToolInputChildStarted'
  readonly toolCallId: string
  readonly field: [BindingChildren<B>] extends [never] ? string : ChildBindingField<BindingChildren<B>>
  readonly index: number
  readonly attributes: [BindingChildren<B>] extends [never]
    ? Readonly<Record<string, string | number | boolean>>
    : ChildAttrsPick<ChildElem<TInput, BindingChildren<B>>, BindingChildren<B>>
}

/** Child element closed — full value available */
export interface ToolInputChildComplete<TInput = unknown, B = unknown> {
  readonly _tag: 'ToolInputChildComplete'
  readonly toolCallId: string
  readonly field: [BindingChildren<B>] extends [never] ? string : ChildBindingField<BindingChildren<B>>
  readonly index: number
  readonly value: [BindingChildren<B>] extends [never]
    ? Readonly<Record<string, unknown>>
    : ChildElem<TInput, BindingChildren<B>>
}

/** Full input built, tool will be dispatched */
export interface ToolInputReady<TInput = unknown> {
  readonly _tag: 'ToolInputReady'
  readonly toolCallId: string
  readonly input: TInput
}

/** Tool input was invalid — terminates the tool call (no execution) */
export interface ToolInputParseError {
  readonly _tag: 'ToolInputParseError'
  readonly toolCallId: string
  readonly tagName: string
  readonly toolName: string
  readonly group: string
  readonly error: ToolCallError
}

// =============================================================================
// Prose Events
// =============================================================================

export interface ProseChunk {
  readonly _tag: 'ProseChunk'
  readonly patternId: string  // 'prose' | 'think'
  readonly text: string
}

export interface ProseEnd {
  readonly _tag: 'ProseEnd'
  readonly patternId: string
  readonly content: string
  readonly about: string | null
}

export interface LensStart {
  readonly _tag: 'LensStart'
  readonly name: string
}

export interface LensChunk {
  readonly _tag: 'LensChunk'
  readonly text: string
}

export interface LensEnd {
  readonly _tag: 'LensEnd'
  readonly name: string
  readonly content: string
}


export interface MessageStart {
  readonly _tag: 'MessageStart'
  readonly id: string
  readonly dest: string
  readonly artifactsRaw: string | null
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
// Execution Events (during and after tool execution)
// =============================================================================

/** Tool execution is about to begin */
export interface ToolExecutionStarted<TInput = unknown> {
  readonly _tag: 'ToolExecutionStarted'
  readonly toolCallId: string
  readonly group: string
  readonly toolName: string
  readonly input: TInput
  readonly cached: boolean
}

/** Tool execution completed */
export interface ToolExecutionEnded<TOutput = unknown> {
  readonly _tag: 'ToolExecutionEnded'
  readonly toolCallId: string
  readonly group: string
  readonly toolName: string
  readonly result: XmlToolResult<TOutput>
}

/** Entire XML stream processed */
export interface TurnEnd {
  readonly _tag: 'TurnEnd'
  readonly result: XmlExecutionResult
}

// =============================================================================
// Tool Call Event (tool-scoped subset — what a visual reducer receives)
// =============================================================================

/** Events scoped to a specific tool call. Excludes prose and terminal events. */
export type ToolCallEvent<TInput = unknown, TOutput = unknown, B = unknown> =
  | ToolInputStarted
  | ToolInputFieldValue<TInput, B>
  | ToolInputBodyChunk<TInput, B>
  | ToolInputChildStarted<TInput, B>
  | ToolInputChildComplete<TInput, B>
  | ToolInputReady<TInput>
  | ToolInputParseError
  | ToolExecutionStarted<TInput>
  | ToolExecutionEnded<TOutput>
  | ToolObservation

// =============================================================================
// Combined Event Union
// =============================================================================

export interface ToolObservation {
  readonly _tag: 'ToolObservation'
  readonly toolCallId: string
  readonly tagName: string
  readonly query: string
  readonly content: ContentPart[]
}

export interface StructuralParseError {
  readonly _tag: 'StructuralParseError'
  readonly error: UnclosedThinkDetail | UnclosedActionsDetail | TurnControlConflictDetail | FinishWithoutEvidenceDetail
}


/** Full event stream — composes ToolCallEvent with prose and terminal events. */
export type XmlRuntimeEvent<TInput = unknown, TOutput = unknown, B = unknown> =
  | ToolCallEvent<TInput, TOutput, B>
  | ProseChunk | ProseEnd | LensStart | LensChunk | LensEnd
  | MessageStart | MessageChunk | MessageEnd
  | StructuralParseError
  | TurnEnd

// =============================================================================
// Result Types
// =============================================================================

export type XmlToolResult<TOutput = unknown> =
  | { readonly _tag: 'Success'; readonly output: TOutput; readonly outputTree: { readonly tag: string; readonly tree: OutputNode }; readonly query: string }
  | { readonly _tag: 'Error'; readonly error: string }
  | { readonly _tag: 'Rejected'; readonly rejection: unknown }
  | { readonly _tag: 'Interrupted' }

export type XmlExecutionResult =
  | { readonly _tag: 'Success'; readonly turnControl: 'continue' | 'yield' | null }
  | { readonly _tag: 'Success'; readonly turnControl: 'finish'; readonly evidence: string }
  | { readonly _tag: 'Failure'; readonly error: string }
  | { readonly _tag: 'Interrupted' }
  | { readonly _tag: 'GateRejected'; readonly rejection: unknown }

// =============================================================================
// Runtime Crash Error
// =============================================================================

export class XmlRuntimeCrash {
  readonly _tag = 'XmlRuntimeCrash'
  constructor(readonly message: string, readonly cause?: unknown) {}
}

// =============================================================================
// Service Tags (Effect DI)
// =============================================================================

/** Interceptor for permission/approval/validation before and after tool execution. */
export interface ToolInterceptor {
  /** Called before tool execution. Can proceed (optionally modifying input) or reject. */
  readonly beforeExecute: (
    ctx: InterceptorContext
  ) => Effect.Effect<InterceptorDecision>

  /** Called after successful execution. Can inspect result or reject. */
  readonly afterExecute?: (
    ctx: InterceptorContext & { readonly result: unknown }
  ) => Effect.Effect<InterceptorDecision>
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

export class ToolInterceptorTag extends Context.Tag('xml-act/ToolInterceptor')<
  ToolInterceptorTag, ToolInterceptor
>() {}

// =============================================================================
// Reactor State (for replay support)
// =============================================================================

export type ToolOutcome =
  | { readonly _tag: 'Completed'; readonly result: XmlToolResult }
  | { readonly _tag: 'ParseError' }

export interface ReactorState {
  readonly toolCallMap: ReadonlyMap<string, string>       // toolCallId → tagName
  readonly deadToolCalls: ReadonlySet<string>              // toolCallIds with parse errors
  readonly outputTrees: ReadonlyMap<string, readonly OutputNode[]> // retained successful output trees from older replay state
  readonly stopped: boolean                                // processing halted
  readonly toolOutcomes: ReadonlyMap<string, ToolOutcome>  // known outcomes for replay
}

// =============================================================================
// Runtime Configuration
// =============================================================================

export interface XmlRuntimeConfig {
  /** Registered tools keyed by XML tag name */
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly defaultProseDest?: string
}
