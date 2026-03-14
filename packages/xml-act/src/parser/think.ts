import type {
  LensTagAttrsFrame,
  LensTagNameFrame,
  ParseEvent,
  ParseStack,
  ParserConfig,
  PendingThinkCloseFrame,
  StepResult,
  ThinkCloseTagFrame,
  ThinkFrame,
  ThinkState,
} from './types'
import { emit, NOOP } from './types'

export function emitThinkChar(think: ThinkState, ch: string, config: ParserConfig): ParseEvent[] {
  if (think.tagName === config.keywords.lenses && think.activeLens) {
    think.activeLens.content += ch
    think.lastCharNewline = ch === '\n'
    return [{ _tag: 'LensChunk', text: ch }]
  }
  think.body += ch
  const events: ParseEvent[] = []
  if (think.tagName !== config.keywords.lenses) {
    events.push({ _tag: 'ProseChunk', patternId: 'think', text: ch })
  }
  if (think.openTagBuf) {
    if (ch === '>') {
      if (think.openAfterNewline && think.openTagBuf === think.tagName) think.depth++
      think.openTagBuf = ''
    } else if (/[a-zA-Z]/.test(ch)) {
      think.openTagBuf += ch
    } else {
      think.openTagBuf = ''
    }
  }
  think.lastCharNewline = ch === '\n'
  return events
}

export function stepThink({ frame, state, ch, config }: { frame: ThinkFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (frame.pendingLt) {
    frame.pendingLt = false
    const wasAfterNewline = frame.think.lastCharNewline
    if (ch === '/') {
      state[state.length - 1] = { _tag: 'ThinkCloseTag', think: frame.think, close: { name: '', raw: '</' }, afterNewline: wasAfterNewline }
    } else if (ch === '!') {
      events.push(...emitThinkChar(frame.think, '<', config))
      events.push(...emitThinkChar(frame.think, '!', config))
    } else if (/[a-zA-Z0-9_-]/.test(ch)) {
      if (frame.think.tagName === config.keywords.lenses) {
        state[state.length - 1] = { _tag: 'LensTagName', think: frame.think, name: ch }
      } else {
        events.push(...emitThinkChar(frame.think, '<', config))
        events.push(...emitThinkChar(frame.think, ch, config))
        frame.think.openTagBuf = ch
        frame.think.openAfterNewline = wasAfterNewline
      }
    } else {
      events.push(...emitThinkChar(frame.think, '<', config))
      frame.think.openTagBuf = ''
      events.push(...emitThinkChar(frame.think, ch, config))
    }
  } else if (ch === '<') {
    frame.pendingLt = true
  } else {
    events.push(...emitThinkChar(frame.think, ch, config))
  }
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepThinkCloseTag({ frame, state, ch, config }: { frame: ThinkCloseTagFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.close.raw += ch
  if (ch !== '>') {
    frame.close.name += ch
    return NOOP
  }
  if (frame.think.tagName === config.keywords.lenses && frame.close.name === 'lens') {
    if (frame.think.activeLens) {
      if (frame.think.activeLens.depth > 0) {
        frame.think.activeLens.depth--
        for (const c of frame.close.raw) events.push(...emitThinkChar(frame.think, c, config))
      } else {
        const content = frame.think.activeLens.content.trim()
        events.push({ _tag: 'LensEnd', name: frame.think.activeLens.name, content })
        frame.think.lenses.push({ name: frame.think.activeLens.name, content })
        frame.think.activeLens = null
      }
    } else {
      for (const c of frame.close.raw) events.push(...emitThinkChar(frame.think, c, config))
    }
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    return events.length > 0 ? emit(...events) : NOOP
  }
  if (frame.close.name === frame.think.tagName && frame.afterNewline) {
    if (frame.think.depth > 0) {
      frame.think.depth--
      for (const c of frame.close.raw) events.push(...emitThinkChar(frame.think, c, config))
      state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    } else {
      if (frame.think.tagName !== config.keywords.lenses) {
        events.push({ _tag: 'ProseEnd', patternId: 'think', content: frame.think.body, about: frame.think.about })
      }
      state.pop()
    }
  } else if (frame.close.name === frame.think.tagName && !frame.afterNewline) {
    state[state.length - 1] = { _tag: 'PendingThinkClose', think: frame.think, closeRaw: frame.close.raw }
  } else {
    for (const c of frame.close.raw) events.push(...emitThinkChar(frame.think, c, config))
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
  }
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepPendingThinkClose({ frame, state, ch, config }: { frame: PendingThinkCloseFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (ch === '\n') {
    if (frame.think.depth > 0) {
      frame.think.depth--
      for (const c of frame.closeRaw) events.push(...emitThinkChar(frame.think, c, config))
      events.push(...emitThinkChar(frame.think, '\n', config))
      state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    } else {
      if (frame.think.tagName !== config.keywords.lenses) {
        events.push({ _tag: 'ProseEnd', patternId: 'think', content: frame.think.body, about: frame.think.about })
      }
      state.pop()
      const prose = state[0]
      prose.lastCharNewline = true
    }
    return events.length > 0 ? emit(...events) : NOOP
  }
  for (const c of frame.closeRaw) events.push(...emitThinkChar(frame.think, c, config))
  state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
  events.push(...emitThinkChar(frame.think, ch, config))
  return emit(...events)
}

export function stepLensTagName({ frame, state, ch, config }: { frame: LensTagNameFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (/[a-zA-Z]/.test(ch)) {
    const next = frame.name + ch
    if ('lens'.startsWith(next)) frame.name = next
    else {
      for (const c of '<' + next) events.push(...emitThinkChar(frame.think, c, config))
      state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    }
    return events.length > 0 ? emit(...events) : NOOP
  }
  if (frame.name !== 'lens') {
    for (const c of '<' + frame.name + ch) events.push(...emitThinkChar(frame.think, c, config))
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    return emit(...events)
  }
  if (/\s/.test(ch)) {
    state[state.length - 1] = { _tag: 'LensTagAttrs', think: frame.think, attrKey: '', attrValue: '', phase: 'key', nameAttr: null, pendingSlash: false }
    return NOOP
  }
  if (ch === '/') {
    state[state.length - 1] = { _tag: 'LensTagAttrs', think: frame.think, attrKey: '', attrValue: '', phase: 'key', nameAttr: null, pendingSlash: true }
    return NOOP
  }
  if (ch === '>') {
    frame.think.activeLens = { name: '', content: '', depth: 0 }
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    return NOOP
  }
  for (const c of '<' + frame.name + ch) events.push(...emitThinkChar(frame.think, c, config))
  state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
  return emit(...events)
}

export function stepLensTagAttrs({ frame, state, ch, config }: { frame: LensTagAttrsFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (frame.pendingSlash) {
    if (/\s/.test(ch)) return NOOP
    if (ch === '>') {
      const name = frame.nameAttr ?? ''
      if (frame.think.activeLens) {
        for (const c of `<lens${name ? ` name="${name}"` : ''} />`) events.push(...emitThinkChar(frame.think, c, config))
      } else {
        events.push({ _tag: 'LensStart', name })
        events.push({ _tag: 'LensEnd', name, content: '' })
        frame.think.lenses.push({ name, content: null })
      }
      state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
      return emit(...events)
    }
    frame.pendingSlash = false
  }
  if (frame.phase === 'equals') {
    if (/\s/.test(ch)) return NOOP
    if (ch === '"') {
      frame.attrValue = ''
      frame.phase = 'value'
      return NOOP
    }
    for (const c of `<lens ${frame.attrKey}=${ch}`) events.push(...emitThinkChar(frame.think, c, config))
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    return emit(...events)
  }
  if (frame.phase === 'value') {
    if (ch === '"') {
      if (frame.attrKey === 'name') frame.nameAttr = frame.attrValue
      frame.attrKey = ''
      frame.attrValue = ''
      frame.phase = 'key'
    } else frame.attrValue += ch
    return NOOP
  }
  if (ch === '>') {
    const name = frame.nameAttr ?? ''
    if (frame.think.activeLens) {
      frame.think.activeLens.depth++
      for (const c of `<lens${name ? ` name="${name}"` : ''}>`) events.push(...emitThinkChar(frame.think, c, config))
    } else {
      frame.think.activeLens = { name, content: '', depth: 0 }
      events.push({ _tag: 'LensStart', name })
    }
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    return events.length > 0 ? emit(...events) : NOOP
  }
  if (ch === '/') { frame.pendingSlash = true; return NOOP }
  if (/\s/.test(ch)) { if (frame.attrKey.length > 0) frame.attrKey = ''; return NOOP }
  if (ch === '=') { frame.phase = 'equals'; return NOOP }
  frame.attrKey += ch
  return NOOP
}