import type { MessageBodyFrame, MessageBodyOpenTagFrame, MessageCloseTagFrame, ParseEvent, ParseStack, ParserConfig, StepResult } from './types'
import { emit, mkCloseTag, NOOP } from './types'

export function flushPendingMessageNewline(frame: MessageBodyFrame): ParseEvent[] {
  if (!frame.pendingNewline) return []
  frame.pendingNewline = false
  frame.body += '\n'
  return [{ _tag: 'MessageBodyChunk', id: frame.id, text: '\n' }]
}

export function stepMessageBody({ frame, state, ch }: { frame: MessageBodyFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (frame.pendingLt) {
    frame.pendingLt = false
    if (ch === '/') {
      state[state.length - 1] = { _tag: 'MessageCloseTag', id: frame.id, dest: frame.dest, artifactsRaw: frame.artifactsRaw, body: frame.body, close: mkCloseTag(), depth: frame.depth, pendingNewline: frame.pendingNewline }
      return NOOP
    }
    if (/[a-zA-Z0-9_-]/.test(ch)) {
      state[state.length - 1] = { _tag: 'MessageBodyOpenTag', id: frame.id, dest: frame.dest, artifactsRaw: frame.artifactsRaw, body: frame.body, depth: frame.depth, pendingNewline: frame.pendingNewline, raw: '<' + ch, name: ch, matchingName: 'message'.startsWith(ch), inName: true, selfClosing: false }
      return NOOP
    }
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

export function stepMessageBodyOpenTag({ frame, state, ch }: { frame: MessageBodyOpenTagFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.raw += ch
  if (frame.inName && /[a-zA-Z0-9_-]/.test(ch)) {
    frame.name += ch
    frame.matchingName = frame.matchingName && 'message'.startsWith(frame.name)
    return NOOP
  }
  if (frame.inName) {
    frame.inName = false
    if (frame.name !== 'message') frame.matchingName = false
  }
  if (ch === '>') {
    if (frame.pendingNewline) {
      frame.body += '\n'
      events.push({ _tag: 'MessageBodyChunk', id: frame.id, text: '\n' })
    }
    const text = frame.raw
    frame.body += text
    events.push({ _tag: 'MessageBodyChunk', id: frame.id, text })
    state[state.length - 1] = { _tag: 'MessageBody', id: frame.id, dest: frame.dest, artifactsRaw: frame.artifactsRaw, body: frame.body, pendingLt: false, depth: frame.matchingName && frame.name === 'message' && !frame.selfClosing ? frame.depth + 1 : frame.depth, pendingNewline: false }
    return emit(...events)
  }
  if (ch === '/') frame.selfClosing = true
  return NOOP
}

export function stepMessageCloseTag({ frame, state, ch }: { frame: MessageCloseTagFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.close.raw += ch
  if (ch !== '>') { frame.close.name += ch; return NOOP }
  if (frame.close.name === 'message' && frame.depth === 0) {
    events.push({ _tag: 'MessageTagClose', id: frame.id })
    state.pop()
    if (state[state.length - 1]?._tag === 'Comms' && state[state.length - 2]?._tag === 'Prose') {
      state.pop()
    }
    return emit(...events)
  }
  if (frame.pendingNewline) {
    frame.body += '\n'
    events.push({ _tag: 'MessageBodyChunk', id: frame.id, text: '\n' })
  }
  const text = frame.close.raw
  frame.body += text
  events.push({ _tag: 'MessageBodyChunk', id: frame.id, text })
  state[state.length - 1] = { _tag: 'MessageBody', id: frame.id, dest: frame.dest, artifactsRaw: frame.artifactsRaw, body: frame.body, pendingLt: false, depth: frame.close.name === 'message' ? frame.depth - 1 : frame.depth, pendingNewline: false }
  return emit(...events)
}