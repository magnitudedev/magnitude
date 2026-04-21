/**
 * Per-frame content handlers — append text to the current frame and emit
 * the appropriate streaming chunk events.
 *
 * All functions are pure: they take a frame + text and return Op[].
 * The caller (index.ts) applies the ops via the stack machine.
 */

import type { TurnEngineEvent } from '../types'
import type { Op } from '../machine'
import type { ProseFrame, MessageFrame, ThinkFrame } from './types'
import { PROSE_VALID_TAGS } from './types'

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
    frame: { type: 'prose', body: '', pendingNewlines: 0, hasContent: false, validTags: PROSE_VALID_TAGS },
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
