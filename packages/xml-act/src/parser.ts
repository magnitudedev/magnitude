/**
 * Streaming parser.
 *
 * Consumes Token events from the tokenizer and produces:
 * - ParseEvent (parameter, filter, invoke lifecycle events)
 * - Structural events (ProseChunk, ProseEnd, LensStart/LensChunk/LensEnd, MessageStart/MessageChunk/MessageEnd, TurnControl)
 */

import type { Token, ParseEvent, StructuralEvent, ParameterStarted, ParameterChunk, ParameterComplete, FilterStarted, FilterChunk, FilterComplete, InvokeStarted, InvokeComplete } from './types'
import { createStackMachine, type Op } from './machine'

// =============================================================================
// Character helpers (no regex)
// =============================================================================

function isWhitespace(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r'
}

function isNewline(ch: string): boolean {
  return ch === '\n'
}

function countLeadingWhitespace(text: string): number {
  for (let i = 0; i < text.length; i++) {
    if (!isWhitespace(text[i])) return i
  }
  return text.length
}

function countTrailingWhitespace(text: string): number {
  for (let i = text.length - 1; i >= 0; i--) {
    if (!isWhitespace(text[i])) return text.length - 1 - i
  }
  return text.length
}

function countNewlines(text: string): number {
  let count = 0
  for (let i = 0; i < text.length; i++) {
    if (isNewline(text[i])) count++
  }
  return count
}

function isAllWhitespace(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    if (!isWhitespace(text[i])) return false
  }
  return true
}

function stripLeadingWhitespace(text: string): string {
  const count = countLeadingWhitespace(text)
  return count === 0 ? text : text.slice(count)
}

function stripTrailingWhitespace(text: string): string {
  const count = countTrailingWhitespace(text)
  return count === 0 ? text : text.slice(0, text.length - count)
}

// =============================================================================
// Frame Types
// =============================================================================

export type Frame =
  | { readonly type: 'prose'; readonly body: string; readonly pendingNewlines: number; readonly hasContent: boolean }
  | { readonly type: 'think'; readonly name: string; readonly content: string; readonly hasContent: boolean; readonly pendingNewlines: number }
  | { readonly type: 'message'; readonly id: string; readonly to: string | null; readonly content: string; readonly pendingNewlines: number }
  | { readonly type: 'invoke'; readonly toolCallId: string; readonly toolTag: string; readonly toolName: string; readonly group: string; readonly parameters: Map<string, string>; readonly hasFilter: boolean }
  | { readonly type: 'parameter'; readonly toolCallId: string; readonly parameterName: string; readonly value: string }
  | { readonly type: 'filter'; readonly toolCallId: string; readonly filterType: string; readonly query: string }

// =============================================================================
// Structural Event Types
// =============================================================================

export type { StructuralEvent } from './types'
export type ParserEvent = ParseEvent | StructuralEvent

// =============================================================================
// Parser State
// =============================================================================

export interface Parser {
  pushToken(token: Token): void
  end(): void
  /** Consume and return all pending events, clearing the internal buffer */
  drain(): readonly ParserEvent[]
}

// (ID generator moved inside createParser)

// =============================================================================
// Prose helpers
// =============================================================================

type ProseFrame = Extract<Frame, { type: 'prose' }>

/**
 * Append text to the top prose frame with whitespace suppression.
 * - Leading whitespace at prose start is stripped
 * - Pure whitespace is tracked as pendingNewlines and NOT emitted
 * - Only non-whitespace content triggers ProseChunk emission
 */
function appendProse(top: ProseFrame, text: string): Op<Frame, ParserEvent>[] {
  const ops: Op<Frame, ParserEvent>[] = []

  // Case: prose has no real content yet
  if (!top.hasContent) {
    const stripped = stripLeadingWhitespace(text)
    
    if (stripped.length === 0) {
      // All whitespace — track pending newlines, emit nothing
      const newPending = top.pendingNewlines + countNewlines(text)
      ops.push({ type: 'replace', frame: { ...top, pendingNewlines: newPending } })
      return ops
    }
    
    // Has real content after stripping leading whitespace
    ops.push({ type: 'replace', frame: { ...top, body: stripped, hasContent: true, pendingNewlines: 0 } })
    ops.push({ type: 'emit', event: { _tag: 'ProseChunk', text: stripped } })
    return ops
  }

  // Case: prose already has content — emit everything
  // Flush any pending newlines first
  if (top.pendingNewlines > 0) {
    const prefix = '\n'.repeat(top.pendingNewlines)
    ops.push({ type: 'replace', frame: { ...top, body: top.body + prefix, pendingNewlines: 0 } })
    ops.push({ type: 'emit', event: { _tag: 'ProseChunk', text: prefix } })
  }

  ops.push({ type: 'replace', frame: { ...top, body: top.body + text, pendingNewlines: 0 } })
  ops.push({ type: 'emit', event: { _tag: 'ProseChunk', text } })
  return ops
}

/**
 * End the top prose section. Trailing whitespace is trimmed.
 * If the prose body is empty after trimming, NO ProseEnd is emitted.
 */
function endTopProse(top: ProseFrame): Op<Frame, ParserEvent>[] {
  const trimmed = stripTrailingWhitespace(top.body)
  
  if (trimmed.length === 0 && !top.hasContent) {
    // Entire prose section was whitespace — suppress completely
    return [{ type: 'replace', frame: { type: 'prose', body: '', pendingNewlines: 0, hasContent: false } }]
  }
  
  return [
    { type: 'emit', event: { _tag: 'ProseEnd', content: trimmed } },
    { type: 'replace', frame: { type: 'prose', body: '', pendingNewlines: 0, hasContent: false } }
  ]
}

// =============================================================================
// Message helpers
// =============================================================================

type MessageFrame = Extract<Frame, { type: 'message' }>

/**
 * Append text to a message frame with pending newline handling.
 * - Leading newlines at message start are stripped
 * - Pure newlines are tracked as pendingNewlines and NOT emitted immediately
 * - Trailing newlines on content are tracked for potential suppression
 */
function appendMessage(top: MessageFrame, text: string): Op<Frame, ParserEvent>[] {
  const ops: Op<Frame, ParserEvent>[] = []

  // Pure newlines — defer
  if (isAllWhitespace(text) && countNewlines(text) === text.length) {
    ops.push({ type: 'replace', frame: { ...top, pendingNewlines: top.pendingNewlines + text.length } })
    return ops
  }

  // Content starts — handle leading newlines
  let segment = text
  let trailingNewlines = 0

  // Strip leading newlines if body is empty
  if (top.content.length === 0) {
    segment = stripLeadingWhitespace(segment)
  }

  // Count trailing newlines
  for (let i = segment.length - 1; i >= 0; i--) {
    if (isNewline(segment[i])) trailingNewlines++
    else break
  }
  if (trailingNewlines > 0) {
    segment = segment.slice(0, segment.length - trailingNewlines)
  }

  // Emit pending newlines prefix if body has content
  const prefix = top.pendingNewlines > 0 && top.content.length > 0
    ? '\n'.repeat(top.pendingNewlines)
    : ''
  
  const full = prefix + segment
  const nextBody = full.length > 0 ? top.content + full : top.content

  ops.push({ type: 'replace', frame: { ...top, content: nextBody, pendingNewlines: trailingNewlines } })
  if (full.length > 0) {
    ops.push({ type: 'emit', event: { _tag: 'MessageChunk', id: top.id, text: full } })
  }
  return ops
}

// =============================================================================
// Coalescing Layer
// =============================================================================

type Writable<T> = { -readonly [K in keyof T]: T[K] }

interface CoalescingBuffer {
  key: string
  event: Writable<ParserEvent>
}

function classifyEvent(event: ParserEvent): string | null {
  switch (event._tag) {
    case 'LensChunk': return 'lens'
    case 'MessageChunk': return `message:${event.id}`
    case 'ProseChunk': return 'prose'
    case 'ParameterChunk': return `param:${event.toolCallId}:${event.parameterName}`
    case 'FilterChunk': return `filter:${event.toolCallId}`
    default: return null
  }
}

function mergeEvent(target: Writable<ParserEvent>, source: ParserEvent): void {
  if ('text' in target && 'text' in source) {
    target.text += source.text
  }
}

// =============================================================================
// Parser Implementation
// =============================================================================

export function createParser(customGenerateId?: () => string): Parser {
  let idCounter = 0
  const generateId = customGenerateId ?? (() => `mact-${++idCounter}-${Date.now().toString(36)}`)

  const events: ParserEvent[] = []
  let coalescingBuffer: CoalescingBuffer | null = null
  let deferredYieldTarget: 'user' | 'tool' | 'worker' | 'parent' | null = null
  let postYieldHasContent = false

  function flushCoalescing(): void {
    if (coalescingBuffer === null) return
    events.push(coalescingBuffer.event as ParserEvent)
    coalescingBuffer = null
  }

  function emit(event: ParserEvent): void {
    const key = classifyEvent(event)
    
    if (key === null) {
      // Non-coalescable event — flush buffer, emit immediately
      flushCoalescing()
      events.push(event)
      return
    }

    if (coalescingBuffer !== null && coalescingBuffer.key === key) {
      // Same classification — merge
      mergeEvent(coalescingBuffer.event, event)
      return
    }

    // Different classification or first event — flush buffer, start new
    flushCoalescing()
    coalescingBuffer = { key, event: { ...event } }
  }

  const machine = createStackMachine<Frame, ParserEvent>(
    { type: 'prose', body: '', pendingNewlines: 0, hasContent: false },
    emit
  )

  // Helper to get current frame of a given type (searches from top of stack)
  function getCurrentFrame<T extends Frame['type']>(type: T): Extract<Frame, { type: T }> | undefined {
    for (let i = machine.stack.length - 1; i >= 0; i--) {
      const frame = machine.stack[i]
      if (frame.type === type) return frame as Extract<Frame, { type: T }>
    }
    return undefined
  }

  function pushToken(token: Token): void {
    if (machine.mode === 'observing') {
      if (token._tag === 'Content' && !isAllWhitespace(token.text)) {
        postYieldHasContent = true
      } else if (token._tag !== 'Content') {
        postYieldHasContent = true
      }
      return
    }
    if (machine.mode !== 'active') return

    switch (token._tag) {
      case 'Open': handleOpen(token.name, token.variant); break
      case 'Close': handleClose(token.name, token.pipe); break
      case 'SelfClose': handleSelfClose(token.name, token.variant); break
      case 'Parameter': handleParameterOpen(token.name); break
      case 'ParameterClose': handleParameterClose(); break
      case 'Content': handleContent(token.text); break
    }
  }

  // ===========================================================================
  // Structural element open — first end any active prose section
  // ===========================================================================

  function endCurrentProse(): void {
    const top = machine.peek()
    if (top?.type === 'prose') {
      machine.apply(endTopProse(top))
    }
  }

  function handleOpen(name: string, variant: string | undefined): void {
    const top = machine.peek()

    switch (name) {
      case 'think': {
        // <|think:NAME> - variant is the lens name
        const lensName = variant ?? 'analyze'
        
        // Check if we're already inside a think — treat nested open as content
        const currentThink = getCurrentFrame('think')
        if (currentThink) {
          // Nested think — treat open tag as content within current think
          const raw = `<|think:${lensName}>`
          machine.apply([
            { type: 'replace', frame: { ...currentThink, content: currentThink.content + raw, hasContent: true, pendingNewlines: 0 } }
          ])
          // Only emit LensChunk if we've already started content
          if (currentThink.hasContent || currentThink.pendingNewlines > 0) {
            machine.apply([{ type: 'emit', event: { _tag: 'LensChunk', text: raw } }])
          }
          return
        }

        // End current prose, then push think frame
        endCurrentProse()
        machine.apply([
          { type: 'push', frame: { type: 'think', name: lensName, content: '', hasContent: false, pendingNewlines: 0 } },
          { type: 'emit', event: { _tag: 'LensStart', name: lensName } }
        ])
        break
      }

      case 'message': {
        // <|message> or <|message:recipient>
        const id = generateId()
        const to = variant ?? null

        // End current prose, then push message frame
        endCurrentProse()
        machine.apply([
          { type: 'push', frame: { type: 'message', id, to, content: '', pendingNewlines: 0 } },
          { type: 'emit', event: { _tag: 'MessageStart', id, to } }
        ])
        break
      }

      case 'invoke': {
        // <|invoke:NAME> - variant is the tool tag/name
        if (!variant) {
          // Invalid: invoke without name - treat as content in current context
          const raw = '<|invoke>'
          if (top?.type === 'prose') {
            machine.apply(appendProse(top, raw))
          }
          return
        }

        const toolCallId = generateId()
        const toolTag = variant
        const parts = toolTag.split(':')
        const group = parts.length > 1 ? parts[0] : 'default'
        const toolName = parts.length > 1 ? parts.slice(1).join(':') : toolTag

        // End current prose, then push invoke frame
        endCurrentProse()
        machine.apply([
          { type: 'push', frame: { type: 'invoke', toolCallId, toolTag, toolName, group, parameters: new Map(), hasFilter: false } },
          { type: 'emit', event: { _tag: 'InvokeStarted', toolCallId, toolTag, toolName, group } }
        ])
        break
      }

      default: {
        // Unknown open tag - treat as content in current context
        const variantStr = variant ? `:${variant}` : ''
        const text = `<|${name}${variantStr}>`
        if (top?.type === 'prose') {
          machine.apply(appendProse(top, text))
        } else if (top?.type === 'message') {
          machine.apply(appendMessage(top, text))
        } else if (top?.type === 'think') {
          machine.apply([
            { type: 'replace', frame: { ...top, content: top.content + text } },
            { type: 'emit', event: { _tag: 'LensChunk', text } }
          ])
        } else if (top?.type === 'parameter') {
          machine.apply([
            { type: 'replace', frame: { ...top, value: top.value + text } },
            { type: 'emit', event: { _tag: 'ParameterChunk', toolCallId: top.toolCallId, parameterName: top.parameterName, text } }
          ])
        } else if (top?.type === 'filter') {
          machine.apply([
            { type: 'replace', frame: { ...top, query: top.query + text } },
            { type: 'emit', event: { _tag: 'FilterChunk', toolCallId: top.toolCallId, text } }
          ])
        }
        break
      }
    }
  }

  function handleClose(name: string, pipe: string | undefined): void {
    const top = machine.peek()

    // Check if this is a piped close (filter)
    if (pipe) {
      if (name === 'invoke' && top?.type === 'invoke') {
        const toolCallId = top.toolCallId
        machine.apply([
          { type: 'replace', frame: { ...top, hasFilter: true } },
          { type: 'push', frame: { type: 'filter', toolCallId, filterType: pipe, query: '' } },
          { type: 'emit', event: { _tag: 'FilterStarted', toolCallId, filterType: pipe } }
        ])
      }
      return
    }

    // Regular close
    switch (name) {
      case 'think': {
        const thinkFrame = getCurrentFrame('think')
        if (thinkFrame) {
          // Strip trailing whitespace from think content
          const trimmed = stripTrailingWhitespace(thinkFrame.content)
          machine.apply([
            { type: 'emit', event: { _tag: 'LensEnd', name: thinkFrame.name, content: trimmed } },
            { type: 'pop' }
          ])
        }
        break
      }

      case 'message': {
        const msgFrame = getCurrentFrame('message')
        if (msgFrame) {
          machine.apply([
            { type: 'emit', event: { _tag: 'MessageEnd', id: msgFrame.id } },
            { type: 'pop' }
          ])
        }
        break
      }

      case 'invoke': {
        const invokeFrame = getCurrentFrame('invoke')
        if (invokeFrame) {
          machine.apply([
            { type: 'emit', event: { _tag: 'InvokeComplete', toolCallId: invokeFrame.toolCallId, hasFilter: invokeFrame.hasFilter } },
            { type: 'pop' }
          ])
        }
        break
      }

      case 'parameter': {
        const paramFrame = getCurrentFrame('parameter')
        if (paramFrame) {
          machine.apply([
            { type: 'emit', event: { _tag: 'ParameterComplete', toolCallId: paramFrame.toolCallId, parameterName: paramFrame.parameterName, value: paramFrame.value } },
            { type: 'pop' }
          ])
        } else {
          // Parameter close without being inside one - treat as content
          if (top?.type === 'prose') {
            machine.apply(appendProse(top, '<parameter|>'))
          }
        }
        break
      }

      case 'filter': {
        const filterFrame = getCurrentFrame('filter')
        const invokeFrame = getCurrentFrame('invoke')
        if (filterFrame) {
          machine.apply([
            { type: 'emit', event: { _tag: 'FilterComplete', toolCallId: filterFrame.toolCallId, query: filterFrame.query } },
            { type: 'pop' }
          ])
        }
        if (invokeFrame && invokeFrame.hasFilter) {
          machine.apply([
            { type: 'emit', event: { _tag: 'InvokeComplete', toolCallId: invokeFrame.toolCallId, hasFilter: true } },
            { type: 'pop' }
          ])
        }
        break
      }

      default: {
        // Unknown close tag - treat as content in current context
        const text = `<${name}|>`
        if (top?.type === 'prose') {
          machine.apply(appendProse(top, text))
        } else if (top?.type === 'message') {
          machine.apply(appendMessage(top, text))
        } else if (top?.type === 'think') {
          machine.apply([
            { type: 'replace', frame: { ...top, content: top.content + text } },
            { type: 'emit', event: { _tag: 'LensChunk', text } }
          ])
        } else if (top?.type === 'parameter') {
          machine.apply([
            { type: 'replace', frame: { ...top, value: top.value + text } },
            { type: 'emit', event: { _tag: 'ParameterChunk', toolCallId: top.toolCallId, parameterName: top.parameterName, text } }
          ])
        } else if (top?.type === 'filter') {
          machine.apply([
            { type: 'replace', frame: { ...top, query: top.query + text } },
            { type: 'emit', event: { _tag: 'FilterChunk', toolCallId: top.toolCallId, text } }
          ])
        }
        break
      }
    }
  }

  function handleSelfClose(name: string, variant: string | undefined): void {
    const top = machine.peek()

    switch (name) {
      case 'yield': {
        // <|yield:target|>
        const target = (variant as 'user' | 'tool' | 'worker' | 'parent') ?? 'user'
        // End current prose before yield
        endCurrentProse()
        deferredYieldTarget = target
        postYieldHasContent = false
        machine.apply([{ type: 'observe' }])
        break
      }

      default: {
        // Unknown self-close tag - treat as content in current context
        const variantStr = variant ? `:${variant}` : ''
        const text = `<|${name}${variantStr}|>`
        if (top?.type === 'prose') {
          machine.apply(appendProse(top, text))
        } else if (top?.type === 'message') {
          machine.apply(appendMessage(top, text))
        } else if (top?.type === 'think') {
          machine.apply([
            { type: 'replace', frame: { ...top, content: top.content + text } },
            { type: 'emit', event: { _tag: 'LensChunk', text } }
          ])
        } else if (top?.type === 'parameter') {
          machine.apply([
            { type: 'replace', frame: { ...top, value: top.value + text } },
            { type: 'emit', event: { _tag: 'ParameterChunk', toolCallId: top.toolCallId, parameterName: top.parameterName, text } }
          ])
        } else if (top?.type === 'filter') {
          machine.apply([
            { type: 'replace', frame: { ...top, query: top.query + text } },
            { type: 'emit', event: { _tag: 'FilterChunk', toolCallId: top.toolCallId, text } }
          ])
        }
        break
      }
    }
  }

  function handleParameterOpen(name: string): void {
    const invokeFrame = getCurrentFrame('invoke')
    if (!invokeFrame) {
      // Parameter outside invoke - treat as content
      const top = machine.peek()
      const text = `<|parameter:${name}>`
      if (top?.type === 'prose') {
        machine.apply(appendProse(top, text))
      } else if (top?.type === 'message') {
        machine.apply(appendMessage(top, text))
      } else if (top?.type === 'think') {
        machine.apply([
          { type: 'replace', frame: { ...top, content: top.content + text } }
        ])
      }
      return
    }

    machine.apply([
      { type: 'push', frame: { type: 'parameter', toolCallId: invokeFrame.toolCallId, parameterName: name, value: '' } },
      { type: 'emit', event: { _tag: 'ParameterStarted', toolCallId: invokeFrame.toolCallId, parameterName: name } }
    ])
  }

  function handleParameterClose(): void {
    const paramFrame = getCurrentFrame('parameter')
    if (!paramFrame) {
      const top = machine.peek()
      if (top?.type === 'prose') {
        machine.apply(appendProse(top, '<parameter|>'))
      }
      return
    }

    machine.apply([
      { type: 'emit', event: { _tag: 'ParameterComplete', toolCallId: paramFrame.toolCallId, parameterName: paramFrame.parameterName, value: paramFrame.value } },
      { type: 'pop' }
    ])
  }

  function handleContent(text: string): void {
    const top = machine.peek()
    if (!top) return

    switch (top.type) {
      case 'prose': {
        machine.apply(appendProse(top, text))
        break
      }

      case 'think': {
        // Leading whitespace stripping at think start, like prose
        if (!top.hasContent) {
          const stripped = stripLeadingWhitespace(text)
          if (stripped.length === 0) {
            // All whitespace — track pending newlines, emit nothing
            machine.apply([
              { type: 'replace', frame: { ...top, pendingNewlines: top.pendingNewlines + countNewlines(text) } }
            ])
          } else {
            // Has real content after stripping
            machine.apply([
              { type: 'replace', frame: { ...top, content: stripped, hasContent: true, pendingNewlines: 0 } },
              { type: 'emit', event: { _tag: 'LensChunk', text: stripped } }
            ])
          }
        } else {
          // Flush pending newlines before new content
          if (top.pendingNewlines > 0) {
            const prefix = '\n'.repeat(top.pendingNewlines)
            const full = prefix + text
            machine.apply([
              { type: 'replace', frame: { ...top, content: top.content + full, pendingNewlines: 0 } },
              { type: 'emit', event: { _tag: 'LensChunk', text: full } }
            ])
          } else {
            machine.apply([
              { type: 'replace', frame: { ...top, content: top.content + text } },
              { type: 'emit', event: { _tag: 'LensChunk', text } }
            ])
          }
        }
        break
      }

      case 'message': {
        machine.apply(appendMessage(top, text))
        break
      }

      case 'parameter': {
        machine.apply([
          { type: 'replace', frame: { ...top, value: top.value + text } },
          { type: 'emit', event: { _tag: 'ParameterChunk', toolCallId: top.toolCallId, parameterName: top.parameterName, text } }
        ])
        break
      }

      case 'filter': {
        machine.apply([
          { type: 'replace', frame: { ...top, query: top.query + text } },
          { type: 'emit', event: { _tag: 'FilterChunk', toolCallId: top.toolCallId, text } }
        ])
        break
      }

      case 'invoke': {
        // Content directly inside invoke (between parameters) - ignore
        break
      }
    }
  }

  function end(): void {
    if (machine.mode === 'done') return

    // Flush coalescing buffer
    flushCoalescing()

    // Emit deferred TurnControl from yield (with runaway detection result)
    if (deferredYieldTarget !== null) {
      const termination = postYieldHasContent ? 'runaway' : 'natural'
      events.push({ _tag: 'TurnControl', target: deferredYieldTarget, termination })
      deferredYieldTarget = null
      machine.apply([{ type: 'done' }])
      return
    }

    // Close any open frames in reverse order
    const paramFrame = getCurrentFrame('parameter')
    if (paramFrame) {
      machine.apply([
        { type: 'emit', event: { _tag: 'ParameterComplete', toolCallId: paramFrame.toolCallId, parameterName: paramFrame.parameterName, value: paramFrame.value } },
        { type: 'pop' }
      ])
    }

    const filterFrame = getCurrentFrame('filter')
    if (filterFrame) {
      machine.apply([
        { type: 'emit', event: { _tag: 'FilterComplete', toolCallId: filterFrame.toolCallId, query: filterFrame.query } },
        { type: 'pop' }
      ])
    }

    const invokeFrame = getCurrentFrame('invoke')
    if (invokeFrame) {
      machine.apply([
        { type: 'emit', event: { _tag: 'InvokeComplete', toolCallId: invokeFrame.toolCallId, hasFilter: invokeFrame.hasFilter } },
        { type: 'pop' }
      ])
    }

    const msgFrame = getCurrentFrame('message')
    if (msgFrame) {
      machine.apply([
        { type: 'emit', event: { _tag: 'MessageEnd', id: msgFrame.id } },
        { type: 'pop' }
      ])
    }

    const thinkFrame = getCurrentFrame('think')
    if (thinkFrame) {
      machine.apply([
        {
          type: 'emit',
          event: {
            _tag: 'ParseError',
            error: { _tag: 'UnclosedThink', message: `Unclosed think tag: ${thinkFrame.name}` }
          }
        },
        { type: 'pop' }
      ])
    }

    // End prose with whitespace trimming
    const proseFrame = machine.peek()
    if (proseFrame?.type === 'prose') {
      const trimmed = stripTrailingWhitespace(proseFrame.body)
      if (trimmed.length > 0 || proseFrame.hasContent) {
        machine.apply([
          { type: 'emit', event: { _tag: 'ProseEnd', content: trimmed } },
          { type: 'done' }
        ])
      } else {
        // All whitespace — suppress entirely
        machine.apply([{ type: 'done' }])
      }
    } else {
      machine.apply([{ type: 'done' }])
    }
  }

  return {
    pushToken,
    end,
    drain(): readonly ParserEvent[] {
      flushCoalescing()
      const pending = [...events]
      events.length = 0
      return pending
    }
  }
}
