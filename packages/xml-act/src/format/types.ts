import type { Op } from '../machine'
import type { TagSchema } from '../execution/binding-validator'

/** Scalar value type for parsed attributes (coerced from string during parsing) */
export type AttributeValue = string | number | boolean

export interface ParsedChild {
  readonly tagName: string
  readonly attributes: ReadonlyMap<string, AttributeValue>
  readonly body: string
}

export interface ParsedElement {
  readonly tagName: string
  readonly toolCallId: string
  readonly attributes: ReadonlyMap<string, AttributeValue>
  readonly body: string
  readonly children: readonly ParsedChild[]
}

export type TagParseErrorDetail =
  | { readonly _tag: 'IncompleteTag'; readonly id: string; readonly tagName: string; readonly detail: string }
  | { readonly _tag: 'UnexpectedBody'; readonly id: string; readonly tagName: string; readonly detail: string }
  | { readonly _tag: 'UnclosedChild'; readonly id: string; readonly tagName: string; readonly childTagName: string; readonly detail: string }
  | { readonly _tag: 'UnknownAttribute'; readonly id: string; readonly tagName: string; readonly attribute: string; readonly detail: string }
  | { readonly _tag: 'InvalidAttributeValue'; readonly id: string; readonly tagName: string; readonly attribute: string; readonly expected: string; readonly received: string; readonly detail: string }
  | { readonly _tag: 'MissingRequiredFields'; readonly id: string; readonly tagName: string; readonly fields: readonly string[]; readonly detail: string }
  | { readonly _tag: 'ToolValidationFailed'; readonly id: string; readonly tagName: string; readonly detail: string }

export type ActiveLens = {
  readonly name: string
  readonly body: string
  readonly depth: number
}

export type CompletedLens = {
  readonly name: string
  readonly body: string
}

export type ResolveResult<F, E> =
  | { readonly _tag: 'handle'; readonly handler: TagHandler<F, E> }
  | { readonly _tag: 'passthrough' }

export function HANDLE<F, E>(handler: TagHandler<F, E>): ResolveResult<F, E> {
  return { _tag: 'handle', handler }
}

export const PASS: ResolveResult<never, never> = { _tag: 'passthrough' }

export type TagMap = ReadonlyMap<string, TagHandler<XmlActFrame, XmlActEvent>>

export type XmlActFrame =
  | { readonly type: 'prose'; readonly body: string; readonly pendingNewlines: number; readonly tags: TagMap }

  | {
      readonly type: 'think'
      readonly tag: string
      readonly body: string
      readonly depth: number
      readonly about: string | null
      readonly isLenses: boolean
      readonly activeLens: ActiveLens | null
      readonly lenses: readonly CompletedLens[]
      readonly tags: TagMap
    }
  | {
      readonly type: 'message'
      readonly id: string
      readonly body: string
      readonly depth: number
      readonly pendingNewlines: number
      readonly tags: TagMap
    }

  | {
      readonly type: 'tool-body'
      readonly tag: string
      readonly id: string
      readonly attrs: ReadonlyMap<string, AttributeValue>
      readonly body: string
      readonly children: readonly ParsedChild[]
      readonly childCounts: ReadonlyMap<string, number>
      readonly childTags: ReadonlySet<string>
      readonly schema: TagSchema | undefined
      readonly tags: TagMap
    }
  | {
      readonly type: 'child-body'
      readonly childTagName: string
      readonly childAttrs: ReadonlyMap<string, AttributeValue>
      readonly body: string
      readonly parentToolId: string
      readonly parentTag: string
      readonly childIndex: number
      readonly tags: TagMap
    }
  | { readonly type: 'body-capture'; readonly tag: string; readonly body: string; readonly tags: TagMap }

export type StructuralParseErrorDetail =
  | { readonly _tag: 'UnclosedThink' }

  | { readonly _tag: 'TurnControlConflict' }
  | { readonly _tag: 'FinishWithoutEvidence' }

export type UnclosedThinkDetail = Extract<StructuralParseErrorDetail, { _tag: 'UnclosedThink' }>

export type FinishWithoutEvidenceDetail = Extract<StructuralParseErrorDetail, { _tag: 'FinishWithoutEvidence' }>
export type TurnControlConflictDetail = Extract<StructuralParseErrorDetail, { _tag: 'TurnControlConflict' }>

export type ParseErrorDetail = TagParseErrorDetail | StructuralParseErrorDetail

export type XmlActEvent =
  | {
      readonly _tag: 'TagOpened'
      readonly tagName: string
      readonly toolCallId: string
      readonly attributes: ReadonlyMap<string, AttributeValue>
    }
  | { readonly _tag: 'BodyChunk'; readonly toolCallId: string; readonly text: string }
  | {
      readonly _tag: 'ChildOpened'
      readonly parentToolCallId: string
      readonly childTagName: string
      readonly childIndex: number
      readonly attributes: ReadonlyMap<string, AttributeValue>
    }
  | {
      readonly _tag: 'ChildBodyChunk'
      readonly parentToolCallId: string
      readonly childTagName: string
      readonly childIndex: number
      readonly text: string
    }
  | {
      readonly _tag: 'ChildComplete'
      readonly parentToolCallId: string
      readonly childTagName: string
      readonly childIndex: number
      readonly attributes: ReadonlyMap<string, AttributeValue>
      readonly body: string
    }
  | {
      readonly _tag: 'TagClosed'
      readonly toolCallId: string
      readonly tagName: string
      readonly element: ParsedElement
    }
  | { readonly _tag: 'ProseChunk'; readonly patternId: string; readonly text: string }
  | { readonly _tag: 'ProseEnd'; readonly patternId: string; readonly content: string; readonly about: string | null }
  | { readonly _tag: 'LensStart'; readonly name: string }
  | { readonly _tag: 'LensChunk'; readonly text: string }
  | { readonly _tag: 'LensEnd'; readonly name: string; readonly content: string }

  | {
      readonly _tag: 'MessageStart'
      readonly id: string
      readonly to: string | null
    }
  | { readonly _tag: 'MessageChunk'; readonly id: string; readonly text: string }
  | { readonly _tag: 'MessageEnd'; readonly id: string }
  | { readonly _tag: 'TurnControl'; readonly decision: 'observe' | 'idle' }
  | { readonly _tag: 'TurnControl'; readonly decision: 'finish'; readonly evidence: string }
  | { readonly _tag: 'ParseError'; readonly error: ParseErrorDetail }

export type ParseEvent = XmlActEvent

export type ToolDef = {
  readonly tag: string
  readonly childTags: ReadonlySet<string>
  readonly schema?: TagSchema
}

export interface OpenContext<F> {
  readonly tagName: string
  readonly attrs: ReadonlyMap<string, string>
  readonly afterNewline: boolean
  readonly stack: ReadonlyArray<F>
  readonly generateId: () => string
}

export interface CloseContext<F> {
  readonly tagName: string
  readonly afterNewline: boolean
  readonly stack: ReadonlyArray<F>
}

export interface SelfCloseContext<F> {
  readonly tagName: string
  readonly attrs: ReadonlyMap<string, string>
  readonly afterNewline: boolean
  readonly stack: ReadonlyArray<F>
  readonly generateId: () => string
}

export interface TagHandler<F, E> {
  open(ctx: OpenContext<F>): Op<F, E>[]
  close(ctx: CloseContext<F>): Op<F, E>[]
  selfClose(ctx: SelfCloseContext<F>): Op<F, E>[]
}

export interface Format<F, E> {
  resolve(tagName: string, stack: ReadonlyArray<F>): ResolveResult<F, E>
  onContent(frame: F, text: string): Op<F, E>[]
  onFlush(stack: ReadonlyArray<F>): Op<F, E>[]
  onUnknownOpen(tagName: string, attrs: ReadonlyMap<string, string>, afterNewline: boolean, stack: ReadonlyArray<F>, raw: string): Op<F, E>[]
  onUnknownClose(tagName: string, stack: ReadonlyArray<F>, raw: string): Op<F, E>[]
}

function isFrameType<T extends XmlActFrame['type']>(
  frame: XmlActFrame,
  type: T,
): frame is Extract<XmlActFrame, { type: T }> {
  return frame.type === type
}

export function findFrame<T extends XmlActFrame['type']>(
  stack: ReadonlyArray<XmlActFrame>,
  type: T,
): Extract<XmlActFrame, { type: T }> | undefined {
  for (let i = stack.length - 1; i >= 0; i--) {
    const frame = stack[i]
    if (isFrameType(frame, type)) return frame
  }
  return undefined
}