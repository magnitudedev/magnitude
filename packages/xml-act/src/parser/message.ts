import type { MessageBodyFrame, MessageClosePrefixMatchFrame, MessageOpenPrefixMatchFrame, MessageOpenTagTailFrame, ParseEvent, ParseStack, ParserConfig, StepResult } from './types'
import { emit, NOOP } from './types'
import { advancePrefixMatch } from './prefix-match'

export function flushPendingMessageNewline(frame: MessageBodyFrame): ParseEvent[] {
  if (!frame.pendingNewline) return []
  frame.pendingNewline = false
  frame.body += '\n'
  return [{ _tag: 'MessageBodyChunk', id: frame.id, text: '\n' }]
}

function toMessageBody(frame: { id: string; dest: string; artifactsRaw: string | null; body: string; depth: number; pendingNewline: boolean }): MessageBodyFrame {
  return { _tag: 'MessageBody', id: frame.id, dest: frame.dest, artifactsRaw: frame.artifactsRaw, body: frame.body, pendingLt: false, depth: frame.depth, pendingNewline: frame.pendingNewline }
}

function flushLiteralToBody(state: ParseStack, frame: { id: string; dest: string; artifactsRaw: string | null; body: string; depth: number; pendingNewline: boolean }, literal: string): StepResult {
  const bodyFrame = toMessageBody(frame)
  state[state.length - 1] = bodyFrame
  const events: ParseEvent[] = []
  events.push(...flushPendingMessageNewline(bodyFrame))
  bodyFrame.body += literal
  events.push({ _tag: 'MessageBodyChunk', id: bodyFrame.id, text: literal })
  return emit(...events)
}

export function stepMessageBody({ frame, state, ch }: { frame: MessageBodyFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (frame.pendingLt) {
    frame.pendingLt = false
    if (ch === '/') {
      state[state.length - 1] = {
        _tag: 'MessageClosePrefixMatch',
        id: frame.id, dest: frame.dest, artifactsRaw: frame.artifactsRaw,
        body: frame.body, depth: frame.depth, pendingNewline: frame.pendingNewline,
        prefix: { candidates: ['message'], matched: '', raw: '</' },
      }
      return NOOP
    }
    if (ch === 'm') {
      state[state.length - 1] = {
        _tag: 'MessageOpenPrefixMatch',
        id: frame.id, dest: frame.dest, artifactsRaw: frame.artifactsRaw,
        body: frame.body, depth: frame.depth, pendingNewline: frame.pendingNewline,
        prefix: { candidates: ['message'], matched: 'm', raw: '<m' },
      }
      return NOOP
    }
    // Any other char after < — emit literal
    events.push(...flushPendingMessageNewline(frame))
    const text = '<' + ch
    frame.body += text
    events.push({ _tag: 'MessageBodyChunk', id: frame.id, text })
    return emit(...events)
  }
  if (ch === '<') { frame.pendingLt = true; return NOOP }
  if (ch === '\n') {
    if (frame.body.length === 0 && frame.depth === 0) return NOOP
    frame.pendingNewline = true
    return NOOP
  }
  events.push(...flushPendingMessageNewline(frame))
  frame.body += ch
  events.push({ _tag: 'MessageBodyChunk', id: frame.id, text: ch })
  return emit(...events)
}

export function stepMessageOpenPrefixMatch({ frame, state, ch }: { frame: MessageOpenPrefixMatchFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const advanced = advancePrefixMatch(frame.prefix, ch)

  if (advanced._tag === 'Continue') {
    frame.prefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    return NOOP
  }

  if (advanced._tag === 'NoMatch') {
    return flushLiteralToBody(state, frame, advanced.literal)
  }

  // Matched 'message' — check delimiter
  if (advanced.delimiter === '>') {
    // Nested <message> — emit as literal, increment depth
    const bodyFrame = toMessageBody({ ...frame, depth: frame.depth + 1 })
    state[state.length - 1] = bodyFrame
    const events: ParseEvent[] = []
    events.push(...flushPendingMessageNewline(bodyFrame))
    bodyFrame.body += advanced.raw
    events.push({ _tag: 'MessageBodyChunk', id: bodyFrame.id, text: advanced.raw })
    return emit(...events)
  }

  // Delimiter is whitespace or / — enter tail mode to consume attrs until >
  state[state.length - 1] = {
    _tag: 'MessageOpenTagTail',
    id: frame.id, dest: frame.dest, artifactsRaw: frame.artifactsRaw,
    body: frame.body, depth: frame.depth, pendingNewline: frame.pendingNewline,
    raw: advanced.raw,
    selfClosing: advanced.delimiter === '/',
  }
  return NOOP
}

export function stepMessageOpenTagTail({ frame, state, ch }: { frame: MessageOpenTagTailFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  frame.raw += ch
  if (ch === '/') frame.selfClosing = true
  if (ch !== '>') return NOOP

  // Got > — emit the whole tag as literal, adjust depth
  const newDepth = frame.selfClosing ? frame.depth : frame.depth + 1
  const bodyFrame = toMessageBody({ ...frame, depth: newDepth })
  state[state.length - 1] = bodyFrame
  const events: ParseEvent[] = []
  events.push(...flushPendingMessageNewline(bodyFrame))
  bodyFrame.body += frame.raw
  events.push({ _tag: 'MessageBodyChunk', id: bodyFrame.id, text: frame.raw })
  return emit(...events)
}

export function stepMessageClosePrefixMatch({ frame, state, ch }: { frame: MessageClosePrefixMatchFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const advanced = advancePrefixMatch(frame.prefix, ch)

  if (advanced._tag === 'Continue') {
    frame.prefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    return NOOP
  }

  if (advanced._tag === 'NoMatch') {
    return flushLiteralToBody(state, frame, advanced.literal)
  }

  // Matched 'message' — only accept > as delimiter
  if (advanced.delimiter !== '>') {
    return flushLiteralToBody(state, frame, advanced.raw)
  }

  // Real </message> close
  if (frame.depth === 0) {
    const events: ParseEvent[] = [{ _tag: 'MessageTagClose', id: frame.id }]
    state.pop()
    if (state[state.length - 1]?._tag === 'Comms' && state[state.length - 2]?._tag === 'Prose') state.pop()
    return emit(...events)
  }

  // Nested </message> — emit as literal, decrement depth
  const bodyFrame = toMessageBody({ ...frame, depth: frame.depth - 1 })
  state[state.length - 1] = bodyFrame
  const events: ParseEvent[] = []
  events.push(...flushPendingMessageNewline(bodyFrame))
  bodyFrame.body += advanced.raw
  events.push({ _tag: 'MessageBodyChunk', id: bodyFrame.id, text: advanced.raw })
  return emit(...events)
}