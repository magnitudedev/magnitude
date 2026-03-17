/**
 * Parser Types
 *
 * Internal types for the streaming XML parser.
 * These are NOT exposed to consumers — the runtime maps them to tool-aware events.
 */

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

export type ParseEvent =
  | { readonly _tag: 'TagOpened'; readonly tagName: string; readonly toolCallId: string; readonly attributes: ReadonlyMap<string, AttributeValue> }
  | { readonly _tag: 'BodyChunk'; readonly toolCallId: string; readonly text: string }
  | { readonly _tag: 'ChildOpened'; readonly parentToolCallId: string; readonly childTagName: string; readonly childIndex: number; readonly attributes: ReadonlyMap<string, AttributeValue> }
  | { readonly _tag: 'ChildBodyChunk'; readonly parentToolCallId: string; readonly childTagName: string; readonly childIndex: number; readonly text: string }
  | { readonly _tag: 'ChildComplete'; readonly parentToolCallId: string; readonly childTagName: string; readonly childIndex: number; readonly attributes: ReadonlyMap<string, AttributeValue>; readonly body: string }
  | { readonly _tag: 'TagClosed'; readonly toolCallId: string; readonly tagName: string; readonly element: ParsedElement }
  | { readonly _tag: 'ProseChunk'; readonly patternId: 'prose' | 'think' | (string & {}); readonly text: string }
  | { readonly _tag: 'ProseEnd'; readonly patternId: 'prose' | 'think' | (string & {}); readonly content: string; readonly about: string | null }
  | { readonly _tag: 'LensStart'; readonly name: string }
  | { readonly _tag: 'LensChunk'; readonly text: string }
  | { readonly _tag: 'LensEnd'; readonly name: string; readonly content: string }
  | { readonly _tag: 'ActionsOpen' }
  | { readonly _tag: 'ActionsClose' }
  | { readonly _tag: 'CommsOpen' }
  | { readonly _tag: 'CommsClose' }
  | { readonly _tag: 'MessageTagOpen'; readonly id: string; readonly dest: string; readonly artifactsRaw: string | null }
  | { readonly _tag: 'MessageBodyChunk'; readonly id: string; readonly text: string }
  | { readonly _tag: 'MessageTagClose'; readonly id: string }
  | { readonly _tag: 'TurnControl'; readonly decision: 'continue' | 'yield' }
  | { readonly _tag: 'TurnControl'; readonly decision: 'finish'; readonly evidence: string }
  | { readonly _tag: 'ParseError'; readonly error: ParseErrorDetail }

export type StepResult =
  | { _tag: 'Emit'; events: ParseEvent[] }
  | { _tag: 'EmitAndReprocess'; events: ParseEvent[] }
  | { _tag: 'Reprocess' }
  | { _tag: 'Noop' }

export const NOOP: StepResult = { _tag: 'Noop' }

export function emit(...events: ParseEvent[]): StepResult {
  return { _tag: 'Emit', events }
}

export function emitAndReprocess(...events: ParseEvent[]): StepResult {
  return { _tag: 'EmitAndReprocess', events }
}

export function reprocess(): StepResult {
  return { _tag: 'Reprocess' }
}

export type BaseToolParseErrorDetail =
  | { readonly _tag: 'IncompleteToolTag'; readonly detail: string }
  | { readonly _tag: 'UnexpectedBody'; readonly detail: string }
  | { readonly _tag: 'UnclosedChildTag'; readonly childTagName: string; readonly detail: string }
  | { readonly _tag: 'UnknownAttribute'; readonly attribute: string; readonly detail: string }
  | { readonly _tag: 'InvalidAttributeValue'; readonly attribute: string; readonly expected: string; readonly received: string; readonly detail: string }
  | { readonly _tag: 'MissingRequiredFields'; readonly fields: readonly string[]; readonly detail: string }

export type ToolParseErrorDetail = BaseToolParseErrorDetail & {
  readonly toolCallId: string
  readonly tagName: string
}

export type UnclosedThinkDetail = {
  readonly _tag: 'UnclosedThink'
  readonly detail: string
}

export type UnclosedActionsDetail = {
  readonly _tag: 'UnclosedActions'
  readonly detail: string
}

export type TurnControlConflictDetail = {
  readonly _tag: 'TurnControlConflict'
  readonly detail: string
}

export type FinishWithoutEvidenceDetail = {
  readonly _tag: 'FinishWithoutEvidence'
  readonly detail: string
}

export type ParseErrorDetail =
  | ToolParseErrorDetail
  | UnclosedThinkDetail
  | UnclosedActionsDetail
  | TurnControlConflictDetail
  | FinishWithoutEvidenceDetail

export const enum FencePhase {
  LeadingWs,
  Tick1,
  Tick2,
  Tick3,
  X,
  XM,
  XML,
  TrailingWs,
  Broken,
}

export interface FenceState {
  phase: FencePhase
  buffer: string
  deferred: string
  pendingWhitespace: string
}

export type AttrPhase =
  | { readonly _tag: 'Idle' }
  | { readonly _tag: 'PendingEquals'; readonly key: string }
  | { readonly _tag: 'PendingSlash' }

export interface AttrState {
  phase: AttrPhase
  key: string
  value: string
  attrs: Map<string, AttributeValue>
  hasError: boolean
}

export interface PrefixMatch {
  candidates: readonly string[]
  matched: string
  raw: string
}

export interface ThinkState {
  tagName: string
  body: string
  depth: number
  openPrefix: PrefixMatch | null
  openAfterNewline: boolean
  lastCharNewline: boolean
  about: string | null
  lenses: { name: string; content: string | null }[]
  activeLens: { name: string; content: string; depth: number } | null
}

export interface CloseTagBuf {
  name: string
  raw: string
}

export type CdataPhase =
  | { readonly _tag: 'Prefix'; index: number; buffer: string }
  | { readonly _tag: 'Body'; buffer: string; closeBrackets: number }

export interface ParserConfig {
  knownTags: ReadonlySet<string>
  childTagMap: ReadonlyMap<string, ReadonlySet<string>>
  tagSchemas?: ReadonlyMap<string, TagSchema>
  generateId: () => string
  defaultMessageDest: string
  keywords: {
    actions: string
    think: string
    thinking: string
    lenses: string
    comms: string
    next: string
    yield: string
    finish: string
  }
  structuralTags: ReadonlySet<string>
  actionsTags: ReadonlySet<string>
  topLevelTags: ReadonlySet<string>
  messageTags: ReadonlySet<string>
}

export type ProseFrame = { readonly _tag: 'Prose'; fence: FenceState; proseAccum: string; lastCharNewline: boolean; justClosedStructural: boolean }
export type ActionsFrame = { readonly _tag: 'Actions' }
export type CommsFrame = { readonly _tag: 'Comms' }
export type OpenPrefixMatchFrame = { readonly _tag: 'OpenPrefixMatch'; prefix: PrefixMatch; afterNewline: boolean }
export type ClosePrefixMatchFrame = { readonly _tag: 'ClosePrefixMatch'; prefix: PrefixMatch; afterNewline: boolean }
export type TagAttrsFrame = { readonly _tag: 'TagAttrs'; tagName: string; toolCallId: string; attr: AttrState; raw: string }
export type TagAttrValueFrame = { readonly _tag: 'TagAttrValue'; tagName: string; toolCallId: string; attr: AttrState; raw: string }
export type TagUnquotedAttrValueFrame = { readonly _tag: 'TagUnquotedAttrValue'; tagName: string; toolCallId: string; attr: AttrState; raw: string }
export type PendingStructuralOpenFrame = { readonly _tag: 'PendingStructuralOpen'; tagName: string; raw: string }
export type PendingTopLevelCloseFrame = { readonly _tag: 'PendingTopLevelClose'; tagName: string; closeRaw: string }
export type ThinkFrame = { readonly _tag: 'Think'; think: ThinkState; pendingLt: boolean }
export type ThinkClosePrefixMatchFrame = { readonly _tag: 'ThinkClosePrefixMatch'; think: ThinkState; afterNewline: boolean; prefix: PrefixMatch }
export type PendingThinkCloseFrame = { readonly _tag: 'PendingThinkClose'; think: ThinkState; closeRaw: string }
export type LensOpenPrefixMatchFrame = { readonly _tag: 'LensOpenPrefixMatch'; think: ThinkState; prefix: PrefixMatch }
export type LensTagAttrsFrame = { readonly _tag: 'LensTagAttrs'; think: ThinkState; attrKey: string; attrValue: string; phase: 'key' | 'equals' | 'value'; nameAttr: string | null; pendingSlash: boolean }
export type FinishBodyFrame = { _tag: 'FinishBody'; body: string; pendingLt: boolean }
export type FinishClosePrefixMatchFrame = { _tag: 'FinishClosePrefixMatch'; body: string; prefix: { candidates: readonly string[]; matched: string; raw: string } }
export type MessageBodyFrame = { readonly _tag: 'MessageBody'; id: string; dest: string; artifactsRaw: string | null; body: string; pendingLt: boolean; depth: number; pendingNewline: boolean }
export type MessageOpenPrefixMatchFrame = { readonly _tag: 'MessageOpenPrefixMatch'; id: string; dest: string; artifactsRaw: string | null; body: string; depth: number; pendingNewline: boolean; prefix: PrefixMatch }
export type MessageClosePrefixMatchFrame = { readonly _tag: 'MessageClosePrefixMatch'; id: string; dest: string; artifactsRaw: string | null; body: string; depth: number; pendingNewline: boolean; prefix: PrefixMatch }
export type MessageOpenTagTailFrame = { readonly _tag: 'MessageOpenTagTail'; id: string; dest: string; artifactsRaw: string | null; body: string; depth: number; pendingNewline: boolean; raw: string; selfClosing: boolean }
export type ToolBodyFrame = { readonly _tag: 'ToolBody'; tagName: string; toolCallId: string; attrs: Map<string, AttributeValue>; body: string; children: ParsedChild[]; childCounts: Map<string, number>; pendingLt: boolean }
export type ToolClosePrefixMatchFrame = { readonly _tag: 'ToolClosePrefixMatch'; tool: ToolBodyFrame; prefix: PrefixMatch }
export type ChildOpenPrefixMatchFrame = { readonly _tag: 'ChildOpenPrefixMatch'; tool: ToolBodyFrame; prefix: PrefixMatch }
export type ChildAttrsFrame = { readonly _tag: 'ChildAttrs'; childTagName: string; attr: AttrState; tool: ToolBodyFrame }
export type ChildAttrValueFrame = { readonly _tag: 'ChildAttrValue'; childTagName: string; attr: AttrState; tool: ToolBodyFrame }
export type ChildUnquotedAttrValueFrame = { readonly _tag: 'ChildUnquotedAttrValue'; childTagName: string; attr: AttrState; tool: ToolBodyFrame }
export type ChildBodyFrame = { readonly _tag: 'ChildBody'; childTagName: string; childAttrs: Map<string, AttributeValue>; childBody: string; pendingLt: boolean; tool: ToolBodyFrame }
export type ChildClosePrefixMatchFrame = { readonly _tag: 'ChildClosePrefixMatch'; childTagName: string; childAttrs: Map<string, AttributeValue>; childBody: string; tool: ToolBodyFrame; prefix: PrefixMatch }
export type CdataFrame = { readonly _tag: 'Cdata'; cdata: CdataPhase; origin: ToolBodyFrame | ChildBodyFrame | ProseFrame }
export type DoneFrame = { readonly _tag: 'Done' }

export type StackFrame =
  | ProseFrame
  | FinishBodyFrame
  | FinishClosePrefixMatchFrame
  | ActionsFrame
  | CommsFrame
  | OpenPrefixMatchFrame
  | ClosePrefixMatchFrame
  | TagAttrsFrame
  | TagAttrValueFrame
  | TagUnquotedAttrValueFrame
  | PendingStructuralOpenFrame
  | PendingTopLevelCloseFrame
  | ThinkFrame
  | ThinkClosePrefixMatchFrame
  | PendingThinkCloseFrame
  | LensOpenPrefixMatchFrame
  | LensTagAttrsFrame
  | MessageBodyFrame
  | MessageOpenPrefixMatchFrame
  | MessageClosePrefixMatchFrame
  | MessageOpenTagTailFrame
  | ToolBodyFrame
  | ToolClosePrefixMatchFrame
  | ChildOpenPrefixMatchFrame
  | ChildAttrsFrame
  | ChildAttrValueFrame
  | ChildUnquotedAttrValueFrame
  | ChildBodyFrame
  | ChildClosePrefixMatchFrame
  | CdataFrame
  | DoneFrame

export type ParseStack = [ProseFrame, ...StackFrame[]]

export function mkFence(): FenceState {
  return { phase: FencePhase.LeadingWs, buffer: '', deferred: '', pendingWhitespace: '' }
}

export function mkAttrState(): AttrState {
  return { phase: { _tag: 'Idle' }, key: '', value: '', attrs: new Map(), hasError: false }
}

export function mkCloseTag(): CloseTagBuf {
  return { name: '', raw: '</' }
}

export function mkRootProse(): ProseFrame {
  return { _tag: 'Prose', fence: mkFence(), proseAccum: '', lastCharNewline: true, justClosedStructural: false }
}