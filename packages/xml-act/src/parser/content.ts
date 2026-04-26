/**
 * Per-frame content handlers — append text to the current frame and emit
 * the appropriate streaming chunk events.
 *
 * ContentHandlers is a mapped type — exhaustive by construction. Adding a new frame type
 * without a corresponding entry here is a compile-time error.
 *
 * onContent routes content to the correct per-frame handler via the ContentHandlers map.
 * One cast at the call site is unavoidable (TypeScript cannot correlate a runtime string key
 * with the specific function overload in the map), but the mapped type guarantees every
 * case is handled correctly.
 */

import type { TurnEngineEvent, DeepPaths, SourceSpan } from '../types'
import type { Op } from '../machine'
import type { Frame, ProseFrame, MessageFrame, ThinkFrame, ParameterFrame, FilterFrame, InvokeFrame } from './types'
import type { ParserOp } from './ops'
import { emitEvent, emitStructuralError } from './ops'

// =============================================================================
// Character helpers
// =============================================================================

export function isWhitespace(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r'
}

export function isAllWhitespace(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    if (!isWhitespace(text[i])) return false
  }
  return true
}

export function countNewlines(text: string): number {
  let count = 0
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n') count++
  }
  return count
}

export function stripLeadingWhitespace(text: string): string {
  let i = 0
  while (i < text.length && isWhitespace(text[i])) i++
  return i === 0 ? text : text.slice(i)
}

export function stripTrailingWhitespace(text: string): string {
  let i = text.length - 1
  while (i >= 0 && isWhitespace(text[i])) i--
  return i < 0 ? '' : text.slice(0, i + 1)
}

// =============================================================================
// ContentHandlers — exhaustive mapped type
// =============================================================================

/**
 * ContentHandlers — maps every frame type to its content accumulation function.
 *
 * This mapped type is exhaustive by construction: TypeScript errors if a new frame type
 * is added to the Frame union without a corresponding entry here.
 * "Property 'newFrame' is missing in type" — impossible to miss.
 */
type ContentHandlers = {
  readonly [K in Frame['type']]: (frame: Extract<Frame, { type: K }>, text: string, tokenSpan: SourceSpan) => ParserOp[]
}

const contentHandlers: ContentHandlers = {
  prose:     (frame, text) => appendProse(frame, text) as ParserOp[],
  think:    (frame, text) => appendThink(frame, text) as ParserOp[],
  message:   (frame, text) => appendMessage(frame, text) as ParserOp[],
  parameter: parameterContent,
  filter:    filterContent,
  invoke:    invokeContent,
}

/**
 * onContent — route content text to the current frame's handler.
 *
 * CONTRACT:
 * - prose, think, message: returns Op[] only, no mutation
 * - parameter: mutates frame.rawValue and frame.jsonishParser, returns Op[] for ToolInputFieldChunk
 * - filter: mutates frame.query, returns []
 * - invoke: returns error op for unexpected non-whitespace content
 *
 * Mutation for parameter and filter is intentional — see types.ts for justification.
 */
export function onContent(top: Frame, text: string, tokenSpan: SourceSpan): ParserOp[] {
  // One cast at the call site. The ContentHandlers mapped type guarantees every case is handled
  // and each handler is typed to its specific frame type. TypeScript cannot correlate top.type
  // (a runtime string) with the handler's parameter type through a map lookup — the cast is
  // unavoidable but safe by construction.
  return (contentHandlers[top.type] as (frame: Frame, text: string, tokenSpan: SourceSpan) => ParserOp[])(top, text, tokenSpan)
}

// =============================================================================
// Prose
// =============================================================================

export function appendProse(top: ProseFrame, text: string): Op<ProseFrame, TurnEngineEvent>[] {
  const ops: Op<ProseFrame, TurnEngineEvent>[] = []

  if (!top.hasContent) {
    const stripped = stripLeadingWhitespace(text)
    if (stripped.length === 0) {
      ops.push({ type: 'replace', frame: { ...top, pendingNewlines: top.pendingNewlines + countNewlines(text) } })
      return ops
    }
    ops.push({ type: 'replace', frame: { ...top, body: stripped, hasContent: true, pendingNewlines: 0 } })
    ops.push({ type: 'emit', event: { _tag: 'ProseChunk', text: stripped } })
    return ops
  }

  if (top.pendingNewlines > 0) {
    const prefix = '\n'.repeat(top.pendingNewlines)
    ops.push({ type: 'replace', frame: { ...top, body: top.body + prefix + text, pendingNewlines: 0 } })
    ops.push({ type: 'emit', event: { _tag: 'ProseChunk', text: prefix } })
    ops.push({ type: 'emit', event: { _tag: 'ProseChunk', text } })
  } else {
    ops.push({ type: 'replace', frame: { ...top, body: top.body + text } })
    ops.push({ type: 'emit', event: { _tag: 'ProseChunk', text } })
  }
  return ops
}

export function endTopProse(top: ProseFrame): Op<ProseFrame, TurnEngineEvent>[] {
  const trimmed = stripTrailingWhitespace(top.body)
  const ops: Op<ProseFrame, TurnEngineEvent>[] = []
  if (trimmed.length > 0 || top.hasContent) {
    ops.push({ type: 'emit', event: { _tag: 'ProseEnd', content: trimmed } })
  }
  ops.push({
    type: 'replace',
    frame: { type: 'prose', body: '', pendingNewlines: 0, hasContent: false },
  })
  return ops
}

// =============================================================================
// Message
// =============================================================================

export function appendMessage(top: MessageFrame, text: string): Op<MessageFrame, TurnEngineEvent>[] {
  const ops: Op<MessageFrame, TurnEngineEvent>[] = []

  // Pure newline runs → buffer
  if (isAllWhitespace(text) && text.split('').every(c => c === '\n')) {
    ops.push({ type: 'replace', frame: { ...top, pendingNewlines: top.pendingNewlines + text.length } })
    return ops
  }

  let segment = text

  if (top.content.length === 0) {
    segment = stripLeadingWhitespace(segment)
  }

  // Count and strip trailing newlines
  let trailingNewlines = 0
  for (let i = segment.length - 1; i >= 0; i--) {
    if (segment[i] === '\n') trailingNewlines++
    else break
  }
  if (trailingNewlines > 0) {
    segment = segment.slice(0, segment.length - trailingNewlines)
  }

  const prefix = (top.pendingNewlines > 0 && top.content.length > 0)
    ? '\n'.repeat(top.pendingNewlines)
    : ''

  const full = prefix + segment
  const nextContent = top.content + full

  ops.push({ type: 'replace', frame: { ...top, content: nextContent, pendingNewlines: trailingNewlines } })
  if (full.length > 0) {
    ops.push({ type: 'emit', event: { _tag: 'MessageChunk', id: top.id, text: full } })
  }
  return ops
}

// =============================================================================
// Think / Lens
// =============================================================================

export function appendThink(top: ThinkFrame, text: string): Op<ThinkFrame, TurnEngineEvent>[] {
  const ops: Op<ThinkFrame, TurnEngineEvent>[] = []

  if (!top.hasContent) {
    const stripped = stripLeadingWhitespace(text)
    if (stripped.length === 0) {
      ops.push({ type: 'replace', frame: { ...top, pendingNewlines: top.pendingNewlines + countNewlines(text) } })
      return ops
    }
    ops.push({ type: 'replace', frame: { ...top, content: stripped, hasContent: true, pendingNewlines: 0 } })
    ops.push({ type: 'emit', event: { _tag: 'LensChunk', text: stripped } })
    return ops
  }

  if (top.pendingNewlines > 0) {
    const prefix = '\n'.repeat(top.pendingNewlines)
    const full = prefix + text
    ops.push({ type: 'replace', frame: { ...top, content: top.content + full, pendingNewlines: 0 } })
    ops.push({ type: 'emit', event: { _tag: 'LensChunk', text: full } })
  } else {
    ops.push({ type: 'replace', frame: { ...top, content: top.content + text } })
    ops.push({ type: 'emit', event: { _tag: 'LensChunk', text } })
  }
  return ops
}

// =============================================================================
// Parameter — mutable frame accumulation
// =============================================================================

function parameterContent(frame: ParameterFrame, text: string): ParserOp[] {
  if (frame.dead) return []
  frame.rawValue += text  // mutation — see types.ts for justification
  if (frame.jsonishParser !== null) frame.jsonishParser.push(text)
  const jsonPath = frame.jsonishParser !== null ? frame.jsonishParser.currentPath : []
  const path = [frame.paramName, ...jsonPath] as unknown as DeepPaths<unknown>
  return [
    emitEvent({
      _tag: 'ToolInputFieldChunk',
      toolCallId: frame.toolCallId,
      field: frame.paramName as string & keyof unknown,
      path,
      delta: text,
    }),
  ]
}

// =============================================================================
// Filter — mutable frame accumulation
// =============================================================================

function filterContent(frame: FilterFrame, text: string): ParserOp[] {
  frame.query += text  // mutation — see types.ts for justification
  return []
}

// =============================================================================
// Invoke — unexpected content between parameters
// =============================================================================

function invokeContent(frame: InvokeFrame, text: string, tokenSpan: SourceSpan): ParserOp[] {
  if (isAllWhitespace(text)) return []
  return [
    emitStructuralError({
      _tag: 'UnexpectedContent',
      context: 'invoke:' + frame.toolTag,
      detail: `Unexpected content between parameters: "${text.slice(0, 40)}"`,
      primarySpan: tokenSpan,
    }),
  ]
}
