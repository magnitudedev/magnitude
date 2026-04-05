/**
 * xml-act Core Types
 */

import { Context, Effect, Layer } from "effect"
import type { ToolDefinition, XmlBinding, XmlChildBinding, ContentPart } from "@magnitudedev/tools"
import type { OutputNode } from './output-tree'
import type { TagParseErrorDetail, StructuralParseErrorDetail } from './format/types'

export type XmlTagBinding = Omit<Extract<XmlBinding<unknown>, { type: 'tag' }>, 'type'> & { readonly tag: string }
export type { XmlChildBinding }

export type ResolvePath<T, P extends string> =
  P extends `${infer H}.${infer R}`
    ? H extends keyof T ? ResolvePath<NonNullable<T[H]>, R> : never
    : P extends keyof T ? T[P] : never

export type BindingAttrs<B> = B extends { readonly attributes: readonly (infer A)[] } ? A : never
export type BindingAttrNames<B> = BindingAttrs<B> extends { readonly attr: infer A extends string } ? A : never
export type BindingAttrFields<B> = BindingAttrs<B> extends { readonly field: infer F extends string } ? F : never
export type BindingBody<B> = B extends { readonly body: infer F extends string } ? F : never
export type BindingChildren<B> = B extends { readonly children: readonly (infer C)[] } ? C : never
export type BindingChildTags<B> = B extends { readonly childTags: readonly (infer CT)[] } ? CT : never
export type BindingChildRecord<B> = B extends { readonly childRecord: infer CR } ? CR : never
export type ChildBindingField<C> = C extends { readonly field: infer F extends string } ? F : never
export type ChildBindingAttrs<C> = C extends { readonly attributes: readonly (infer A)[] } ? A : never
export type ChildBindingAttrNames<C> = ChildBindingAttrs<C> extends { readonly attr: infer A extends string } ? A : never
export type ChildBindingAttrFields<C> = ChildBindingAttrs<C> extends { readonly field: infer F extends string } ? F : never
export type ChildBindingBody<C> = C extends { readonly body: infer F extends string } ? F : never
export type ChildRecordField<CR> = CR extends { readonly field: infer F extends string } ? F : never

export type ChildElem<TInput, C> =
  ChildBindingField<C> extends infer CF extends string ?
    CF extends keyof TInput ?
      TInput[CF] extends ReadonlyArray<infer E> ? E : never
    : never
  : never

export type ChildAttrsPick<Elem, C> =
  [ChildBindingAttrFields<C>] extends [never] ? {} : Pick<Elem, ChildBindingAttrFields<C> & keyof Elem>

export interface ToolCallContext {
  readonly toolCallId: string
  readonly tagName: string
  readonly toolName: string
  readonly group: string
}

export interface RegisteredTool {
  readonly tool: ToolDefinition
  readonly tagName: string
  readonly groupName: string
  readonly binding: XmlTagBinding
  readonly outputBinding?: import('@magnitudedev/tools').XmlBinding<unknown>
  readonly meta?: unknown
  readonly layerProvider?: () => Effect.Effect<Layer.Layer<never>, unknown>
}

export interface ToolInputStarted {
  readonly _tag: 'ToolInputStarted'
  readonly toolCallId: string
  readonly tagName: string
  readonly toolName: string
  readonly group: string
  readonly taskId: string | null
}

export interface ToolInputFieldValue<TInput = unknown, B = unknown> {
  readonly _tag: 'ToolInputFieldValue'
  readonly toolCallId: string
  readonly field: [BindingAttrNames<B>] extends [never] ? string : BindingAttrNames<B>
  readonly value: string | number | boolean
}

export type AllBodyFields<B> = BindingBody<B> | ChildBindingBody<BindingChildren<B>>

export interface ToolInputBodyChunk<_TInput = unknown, B = unknown> {
  readonly _tag: 'ToolInputBodyChunk'
  readonly toolCallId: string
  readonly path: readonly string[]
  readonly field: [AllBodyFields<B>] extends [never] ? string : AllBodyFields<B>
  readonly text: string
}

export interface ToolInputChildStarted<TInput = unknown, B = unknown> {
  readonly _tag: 'ToolInputChildStarted'
  readonly toolCallId: string
  readonly field: [BindingChildren<B>] extends [never] ? string : ChildBindingField<BindingChildren<B>>
  readonly index: number
  readonly attributes: [BindingChildren<B>] extends [never]
    ? Readonly<Record<string, string | number | boolean>>
    : ChildAttrsPick<ChildElem<TInput, BindingChildren<B>>, BindingChildren<B>>
}

export interface ToolInputChildComplete<TInput = unknown, B = unknown> {
  readonly _tag: 'ToolInputChildComplete'
  readonly toolCallId: string
  readonly field: [BindingChildren<B>] extends [never] ? string : ChildBindingField<BindingChildren<B>>
  readonly index: number
  readonly value: [BindingChildren<B>] extends [never]
    ? Readonly<Record<string, unknown>>
    : ChildElem<TInput, BindingChildren<B>>
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
  readonly error: TagParseErrorDetail
}

export interface ProseChunk {
  readonly _tag: 'ProseChunk'
  readonly patternId: string
  readonly text: string
}

export interface ProseEnd {
  readonly _tag: 'ProseEnd'
  readonly patternId: string
  readonly content: string
  readonly about: string | null
}

export interface LensStart { readonly _tag: 'LensStart'; readonly name: string }
export interface LensChunk { readonly _tag: 'LensChunk'; readonly text: string }
export interface LensEnd { readonly _tag: 'LensEnd'; readonly name: string; readonly content: string }

export interface TaskStarted {
  readonly _tag: 'TaskStarted'
  readonly id: string
  readonly taskType: string | null
  readonly title: string | null
  readonly parent: string | null
  readonly explicitParent: string | null
  readonly after: string | null
  readonly status: string | null
}
export interface TaskFinished { readonly _tag: 'TaskFinished'; readonly id: string }
export interface TaskPatched {
  readonly _tag: 'TaskPatched'
  readonly id: string
  readonly taskType: string | null
  readonly title: string | null
  readonly parent: string | null
  readonly explicitParent: string | null
  readonly after: string | null
  readonly status: string | null
}
export interface TaskDelegated { readonly _tag: 'TaskDelegated'; readonly taskId: string; readonly role: string; readonly body: string }

export interface MessageStart {
  readonly _tag: 'MessageStart'
  readonly id: string
  readonly scope: 'top-level' | 'task'
  readonly taskId: string | null
  readonly to: string | null
}
export interface MessageChunk { readonly _tag: 'MessageChunk'; readonly id: string; readonly text: string }
export interface MessageEnd { readonly _tag: 'MessageEnd'; readonly id: string }

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
  readonly result: XmlToolResult<TOutput>
}
export interface ToolEmission<TEmission = unknown> { readonly _tag: 'ToolEmission'; readonly toolCallId: string; readonly value: TEmission }
export interface TurnEnd { readonly _tag: 'TurnEnd'; readonly result: XmlExecutionResult }

export type ToolCallEvent<TInput = unknown, TOutput = unknown, B = unknown, TEmission = unknown> =
  | ToolInputStarted
  | ToolInputFieldValue<TInput, B>
  | ToolInputBodyChunk<TInput, B>
  | ToolInputChildStarted<TInput, B>
  | ToolInputChildComplete<TInput, B>
  | ToolInputReady<TInput>
  | ToolInputParseError
  | ToolExecutionStarted<TInput>
  | ToolExecutionEnded<TOutput>
  | ToolEmission<TEmission>
  | ToolObservation

export interface ToolObservation {
  readonly _tag: 'ToolObservation'
  readonly toolCallId: string
  readonly tagName: string
  readonly query: string
  readonly content: ContentPart[]
}
export interface StructuralParseError { readonly _tag: 'StructuralParseError'; readonly error: StructuralParseErrorDetail }

export type XmlRuntimeEvent<TInput = unknown, TOutput = unknown, B = unknown, TEmission = unknown> =
  | ToolCallEvent<TInput, TOutput, B, TEmission>
  | ProseChunk | ProseEnd | LensStart | LensChunk | LensEnd
  | TaskStarted | TaskFinished | TaskPatched | TaskDelegated
  | MessageStart | MessageChunk | MessageEnd
  | StructuralParseError
  | TurnEnd

export type XmlToolResult<TOutput = unknown> =
  | { readonly _tag: 'Success'; readonly output: TOutput; readonly outputTree: { readonly tag: string; readonly tree: OutputNode }; readonly query: string }
  | { readonly _tag: 'Error'; readonly error: string }
  | { readonly _tag: 'Rejected'; readonly rejection: unknown }
  | { readonly _tag: 'Interrupted' }

export type XmlExecutionResult =
  | { readonly _tag: 'Success'; readonly turnControl: 'continue' | 'idle' | null }
  | { readonly _tag: 'Success'; readonly turnControl: 'finish'; readonly evidence: string }
  | { readonly _tag: 'Failure'; readonly error: string }
  | { readonly _tag: 'Interrupted' }
  | { readonly _tag: 'GateRejected'; readonly rejection: unknown }

export class XmlRuntimeCrash {
  readonly _tag = 'XmlRuntimeCrash'
  constructor(readonly message: string, readonly cause?: unknown) {}
}

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

export class ToolInterceptorTag extends Context.Tag('xml-act/ToolInterceptor')<
  ToolInterceptorTag, ToolInterceptor
>() {}

export type ToolOutcome =
  | { readonly _tag: 'Completed'; readonly result: XmlToolResult }
  | { readonly _tag: 'ParseError' }

export interface ReactorState {
  readonly toolCallMap: ReadonlyMap<string, string>
  readonly toolTaskMap: ReadonlyMap<string, string | null>
  readonly taskStack: readonly string[]
  readonly deadToolCalls: ReadonlySet<string>
  readonly outputTrees: ReadonlyMap<string, readonly OutputNode[]>
  readonly stopped: boolean
  readonly toolOutcomes: ReadonlyMap<string, ToolOutcome>
}

export interface XmlRuntimeConfig {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly defaultProseDest?: string
}
