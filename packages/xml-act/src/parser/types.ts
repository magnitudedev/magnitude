/**
 * Parser Types
 *
 * Internal types for the streaming XML parser.
 * These are NOT exposed to consumers — the runtime maps them to tool-aware events.
 */

/** Scalar value type for parsed attributes (coerced from string during parsing) */
export type AttributeValue = string | number | boolean

// =============================================================================
// Parsed Element (complete tag for dispatcher)
// =============================================================================

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

// =============================================================================
// Parse Events (internal)
// =============================================================================


export type ParseEvent =
  /** Tool tag opened — attributes available */
  | { readonly _tag: 'TagOpened'; readonly tagName: string; readonly toolCallId: string; readonly attributes: ReadonlyMap<string, AttributeValue> }
  /** Body text chunk for the current tag */
  | { readonly _tag: 'BodyChunk'; readonly toolCallId: string; readonly text: string }
  /** Child tag opened inside a tool tag — attributes available */
  | { readonly _tag: 'ChildOpened'; readonly parentToolCallId: string; readonly childTagName: string; readonly childIndex: number; readonly attributes: ReadonlyMap<string, AttributeValue> }
  /** Child tag body text chunk */
  | { readonly _tag: 'ChildBodyChunk'; readonly parentToolCallId: string; readonly childTagName: string; readonly childIndex: number; readonly text: string }
  /** Child tag closed — full child data available */
  | { readonly _tag: 'ChildComplete'; readonly parentToolCallId: string; readonly childTagName: string; readonly childIndex: number; readonly attributes: ReadonlyMap<string, AttributeValue>; readonly body: string }
  /** Tool tag closed — complete element available for dispatch */
  | { readonly _tag: 'TagClosed'; readonly toolCallId: string; readonly tagName: string; readonly element: ParsedElement }
  /** Prose text chunk (message or think) */
  | { readonly _tag: 'ProseChunk'; readonly patternId: 'prose' | 'think' | (string & {}); readonly text: string }
  /** Prose block complete */
  | { readonly _tag: 'ProseEnd'; readonly patternId: 'prose' | 'think' | (string & {}); readonly content: string; readonly about: string | null }
  /** Lens started */
  | { readonly _tag: 'LensStart'; readonly name: string }
  /** Lens body text chunk */
  | { readonly _tag: 'LensChunk'; readonly text: string }
  /** Lens closed */
  | { readonly _tag: 'LensEnd'; readonly name: string; readonly content: string }

  /** Actions block opened */
  | { readonly _tag: 'ActionsOpen' }
  /** Actions block closed */
  | { readonly _tag: 'ActionsClose' }
  /** Inspect block opened */
  | { readonly _tag: 'InspectOpen' }
  /** Inspect block closed */
  | { readonly _tag: 'InspectClose' }
  /** Comms block opened */
  | { readonly _tag: 'CommsOpen' }
  /** Comms block closed */
  | { readonly _tag: 'CommsClose' }
  /** Message tag opened inside comms */
  | { readonly _tag: 'MessageTagOpen'; readonly id: string; readonly dest: string; readonly artifactsRaw: string | null }
  /** Message body text chunk */
  | { readonly _tag: 'MessageBodyChunk'; readonly id: string; readonly text: string }
  /** Message tag closed */
  | { readonly _tag: 'MessageTagClose'; readonly id: string }
  /** Ref resolved inside inspect block — carries the resolved content */
  | { readonly _tag: 'InspectResult'; readonly toolRef: string; readonly query?: string; readonly content: string }
  /** Explicit turn control */
  | { readonly _tag: 'TurnControl'; readonly decision: 'continue' | 'yield' }
  /** Parse error — tool-scoped or structural */
  | { readonly _tag: 'ParseError'; readonly error: ParseErrorDetail }

// =============================================================================
// Parse Error Details (parser-internal, no call context)
// =============================================================================

/** Base error detail — no tool call context. Returned by validators. */
export type BaseToolParseErrorDetail =
  | {
      readonly _tag: 'IncompleteToolTag'
      readonly detail: string
    }
  | {
      readonly _tag: 'UnexpectedBody'
      readonly detail: string
    }
  | {
      readonly _tag: 'UnclosedChildTag'
      readonly childTagName: string
      readonly detail: string
    }
  | {
      readonly _tag: 'UnknownAttribute'
      readonly attribute: string
      readonly detail: string
    }
  | {
      readonly _tag: 'InvalidAttributeValue'
      readonly attribute: string
      readonly expected: string
      readonly received: string
      readonly detail: string
    }
  | {
      readonly _tag: 'MissingRequiredFields'
      readonly fields: readonly string[]
      readonly detail: string
    }

/** Tool-scoped error detail — base detail + tool call context. Added by the parser at emit time. */
export type ToolParseErrorDetail = BaseToolParseErrorDetail & {
  readonly toolCallId: string
  readonly tagName: string
}

/** Non-tool error detail — structural errors like invalid refs. */
export type InvalidRefDetail = {
  readonly _tag: 'InvalidRef'
  readonly toolRef: string
  readonly detail: string
}

export type UnclosedThinkDetail = {
  readonly _tag: 'UnclosedThink'
  readonly detail: string
}

export type UnclosedActionsDetail = {
  readonly _tag: 'UnclosedActions'
  readonly detail: string
}

export type UnclosedInspectDetail = {
  readonly _tag: 'UnclosedInspect'
  readonly detail: string
}

/** Full parse error detail union — tool-scoped or structural. */
export type ParseErrorDetail =
  | ToolParseErrorDetail
  | InvalidRefDetail
  | UnclosedThinkDetail
  | UnclosedActionsDetail
  | UnclosedInspectDetail
  | TurnControlConflictDetail
