/**
 * Parser State Types
 *
 * Strongly-typed discriminated union state for the streaming XML parser.
 * Each state variant carries only the data relevant to that state.
 * Impossible states are unrepresentable.
 */

import type { AttributeValue, ParsedChild } from './types'

// =============================================================================
// Sub-state types
// =============================================================================

/** Structural context — persists across all states */
export interface StructuralCtx {
  inActions: boolean
  inInspect: boolean
  inComms: boolean
  justClosedStructural: boolean
  /** Whether the last character processed at prose level was a newline. Used to gate structural tag matching. */
  lastCharNewline: boolean
}

/** Code fence detection phase */
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

/** Code fence detection state (only meaningful in Prose) */
export interface FenceState {
  phase: FencePhase
  buffer: string
  deferred: string
  pendingWhitespace: string
}

/** Attribute parsing phase — unified for tool and child tags */
export type AttrPhase =
  | { readonly _tag: 'Idle' }
  | { readonly _tag: 'PendingEquals'; readonly key: string }
  | { readonly _tag: 'PendingSlash' }

/** Attribute accumulation state */
export interface AttrState {
  phase: AttrPhase
  key: string
  value: string
  attrs: Map<string, AttributeValue>
  hasError: boolean
}

/** Think block state */
export interface ThinkState {
  tagName: string
  body: string
  depth: number
  openTagBuf: string
  /** Whether the pending '<' that started openTagBuf was after a newline */
  openAfterNewline: boolean
  /** Whether the last body char emitted was a newline */
  lastCharNewline: boolean
  /** The about="..." attribute if provided */
  about: string | null
  /** Parsed lenses when tagName is lenses */
  lenses: { name: string; content: string | null }[]
  /** Active open <lens ...> content accumulator */
  activeLens: { name: string; content: string } | null
}

/** Parent context — snapshot of parent tool tag, persists across child lifecycle */
export interface ParentCtx {
  readonly tagName: string
  readonly toolCallId: string
  readonly attrs: ReadonlyMap<string, AttributeValue>
  readonly bodyBefore: string
  readonly children: ParsedChild[]
  readonly childCounts: Map<string, number>
}

/** Close tag accumulator */
export interface CloseTagBuf {
  name: string
  raw: string
}

/** CDATA parsing phase */
export type CdataPhase =
  | { readonly _tag: 'Prefix'; index: number; buffer: string }
  | { readonly _tag: 'Body'; buffer: string; closeBrackets: number }

// =============================================================================
// Main parser state — discriminated union
// =============================================================================

export type ParserState =
  /** Outside any tag */
  | {
      readonly _tag: 'Prose'
      fence: FenceState
      proseAccum: string
    }
  /** Saw '<', accumulating tag name. Carries prose context for endProseBlock/emitProseChunk on resolve. */
  | {
      readonly _tag: 'TagName'
      name: string
      raw: string
      fence: FenceState
      proseAccum: string
      /** Whether the '<' was preceded by a newline (or at start of input). Structural tags only match with this. */
      afterNewline: boolean
    }
  /** Top-level close tag from prose level (e.g. </actions>, </inspect>). Carries prose context. */
  | {
      readonly _tag: 'TopLevelCloseTag'
      close: CloseTagBuf
      fence: FenceState
      proseAccum: string
      /** Whether the '</' was preceded by a newline. Structural close tags only match with this. */
      afterNewline: boolean
    }
  /** Parsing attributes on a top-level tag. Carries prose context. */
  | {
      readonly _tag: 'TagAttrs'
      tagName: string
      toolCallId: string
      attr: AttrState
      raw: string
      fence: FenceState
      proseAccum: string
    }
  /** Inside a quoted attribute value on a top-level tag. Carries prose context. */
  | {
      readonly _tag: 'TagAttrValue'
      tagName: string
      toolCallId: string
      attr: AttrState
      raw: string
      fence: FenceState
      proseAccum: string
    }
  /** Inside an unquoted attribute value on a top-level tag. Carries prose context. */
  | {
      readonly _tag: 'TagUnquotedAttrValue'
      tagName: string
      toolCallId: string
      attr: AttrState
      raw: string
      fence: FenceState
      proseAccum: string
    }
  /** Inside a think/thinking block body */
  | {
      readonly _tag: 'Think'
      think: ThinkState
      pendingLt: boolean
    }
  /** Close tag encountered inside think block */
  | {
      readonly _tag: 'ThinkCloseTag'
      think: ThinkState
      close: CloseTagBuf
      /** Whether the '</' was preceded by a newline */
      afterNewline: boolean
    }
  /** Inside a tool tag body (flat — no child context yet) */
  | {
      readonly _tag: 'ToolBody'
      tagName: string
      toolCallId: string
      attrs: Map<string, AttributeValue>
      body: string
      pendingLt: boolean
    }
  /** Close tag inside flat tool body */
  | {
      readonly _tag: 'ToolCloseTag'
      tagName: string
      toolCallId: string
      attrs: Map<string, AttributeValue>
      body: string
      close: CloseTagBuf
    }
  /** Inside a tool tag body with parent context (children detected) */
  | {
      readonly _tag: 'ParentBody'
      parent: ParentCtx
      body: string
      pendingLt: boolean
    }
  /** Close tag inside parent body (with children) */
  | {
      readonly _tag: 'ParentCloseTag'
      parent: ParentCtx
      body: string
      close: CloseTagBuf
    }
  /** Parsing child tag name */
  | {
      readonly _tag: 'ChildTagName'
      parent: ParentCtx
      parentBody: string
      childName: string
    }
  /** Parsing child tag attributes */
  | {
      readonly _tag: 'ChildAttrs'
      parent: ParentCtx
      parentBody: string
      childTagName: string
      attr: AttrState
    }
  /** Inside a quoted child attribute value */
  | {
      readonly _tag: 'ChildAttrValue'
      parent: ParentCtx
      parentBody: string
      childTagName: string
      attr: AttrState
    }
  /** Inside an unquoted child attribute value */
  | {
      readonly _tag: 'ChildUnquotedAttrValue'
      parent: ParentCtx
      parentBody: string
      childTagName: string
      attr: AttrState
    }
  /** Inside child tag body */
  | {
      readonly _tag: 'ChildBody'
      parent: ParentCtx
      parentBody: string
      childTagName: string
      childAttrs: Map<string, AttributeValue>
      childBody: string
      pendingLt: boolean
    }
  /** Close tag inside child body */
  | {
      readonly _tag: 'ChildCloseTag'
      parent: ParentCtx
      parentBody: string
      childTagName: string
      childAttrs: Map<string, AttributeValue>
      childBody: string
      close: CloseTagBuf
    }
  /** CDATA section (stores origin state tag for return) */
  | {
      readonly _tag: 'Cdata'
      cdata: CdataPhase
      origin: CdataOrigin
    }
  /** Accumulating tag name inside lenses block (potential <lens ...>) */
  | {
      readonly _tag: 'LensTagName'
      think: ThinkState
      name: string
    }
  /** Parsing attributes on a <lens> tag inside lenses block */
  | {
      readonly _tag: 'LensTagAttrs'
      think: ThinkState
      attrKey: string
      attrValue: string
      phase: 'key' | 'equals' | 'value'
      nameAttr: string | null
      pendingSlash: boolean
    }
  /** Saw a structural open tag (think/actions) without preceding newline — waiting to see if next char is \n */
  | {
      readonly _tag: 'PendingStructuralOpen'
      tagName: string
      fence: FenceState
      proseAccum: string
      raw: string
    }
  /** Saw a think close tag without preceding newline — waiting to see if next char is \n */
  | {
      readonly _tag: 'PendingThinkClose'
      think: ThinkState
      closeRaw: string
    }
  /** Saw an actions/inspect/comms close tag without preceding newline — waiting to see if next char is \n */
  | {
      readonly _tag: 'PendingTopLevelClose'
      tagName: string
      fence: FenceState
      proseAccum: string
      closeRaw: string
    }
  /** Inside a comms message body */
  | {
      readonly _tag: 'MessageBody'
      id: string
      dest: string
      artifactsRaw: string | null
      body: string
      pendingLt: boolean
    }
  /** Close tag inside message body */
  | {
      readonly _tag: 'MessageCloseTag'
      id: string
      dest: string
      artifactsRaw: string | null
      body: string
      close: CloseTagBuf
    }
  /** Terminal state — turn control tag emitted, no further parsing */
  | { readonly _tag: 'Done' }

/** Where CDATA was entered from — determines where to return and how to emit content */
export type CdataOrigin =
  | { readonly _tag: 'FromProse'; fence: FenceState; proseAccum: string }
  | { readonly _tag: 'FromToolBody'; tagName: string; toolCallId: string; attrs: Map<string, AttributeValue>; body: string }
  | { readonly _tag: 'FromParentBody'; parent: ParentCtx; body: string }
  | { readonly _tag: 'FromChildBody'; parent: ParentCtx; parentBody: string; childTagName: string; childAttrs: Map<string, AttributeValue>; childBody: string }

// =============================================================================
// Factory helpers
// =============================================================================

export function mkFence(): FenceState {
  return { phase: FencePhase.LeadingWs, buffer: '', deferred: '', pendingWhitespace: '' }
}

export function mkProse(): Extract<ParserState, { _tag: 'Prose' }> {
  return { _tag: 'Prose', fence: mkFence(), proseAccum: '' }
}

export function mkStructural(): StructuralCtx {
  return { inActions: false, inInspect: false, inComms: false, justClosedStructural: false, lastCharNewline: true }
}

export function mkAttrState(): AttrState {
  return { phase: { _tag: 'Idle' }, key: '', value: '', attrs: new Map(), hasError: false }
}

export function mkCloseTag(): CloseTagBuf {
  return { name: '', raw: '</' }
}
