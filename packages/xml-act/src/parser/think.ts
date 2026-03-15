import type {
  LensTagAttrsFrame,
  LensOpenPrefixMatchFrame,
  ParseEvent,
  ParseStack,
  ParserConfig,
  PendingThinkCloseFrame,
  StepResult,
  ThinkClosePrefixMatchFrame,
  ThinkFrame,
  ThinkState,
} from './types'
import { emit, NOOP } from './types'
import { advancePrefixMatch } from './prefix-match'

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
  // Non-lenses nested open-tag depth tracking via prefix matching
  if (think.openPrefix) {
    const advanced = advancePrefixMatch(think.openPrefix, ch)
    if (advanced._tag === 'Continue') {
      think.openPrefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    } else if (advanced._tag === 'Matched' && advanced.delimiter === '>' && think.openAfterNewline) {
      think.depth++
      think.openPrefix = null
    } else {
      // NoMatch or non-> delimiter — stop tracking
      think.openPrefix = null
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
      // Close tag — build candidates
      const closeCandidates: string[] = [frame.think.tagName]
      if (frame.think.tagName === config.keywords.lenses) closeCandidates.push('lens')
      state[state.length - 1] = {
        _tag: 'ThinkClosePrefixMatch',
        think: frame.think,
        afterNewline: wasAfterNewline,
        prefix: { candidates: closeCandidates, matched: '', raw: '</' },
      }
    } else if (ch === '!') {
      events.push(...emitThinkChar(frame.think, '<', config))
      events.push(...emitThinkChar(frame.think, '!', config))
    } else if (frame.think.tagName === config.keywords.lenses && ch === 'l') {
      // Only match <lens in lenses context
      state[state.length - 1] = {
        _tag: 'LensOpenPrefixMatch',
        think: frame.think,
        prefix: { candidates: ['lens'], matched: 'l', raw: '<l' },
      }
    } else if (frame.think.tagName !== config.keywords.lenses && ch === frame.think.tagName[0]) {
      // Non-lenses: start tracking potential nested open tag for depth
      events.push(...emitThinkChar(frame.think, '<', config))
      events.push(...emitThinkChar(frame.think, ch, config))
      frame.think.openPrefix = { candidates: [frame.think.tagName], matched: ch, raw: '<' + ch }
      frame.think.openAfterNewline = wasAfterNewline
    } else {
      events.push(...emitThinkChar(frame.think, '<', config))
      frame.think.openPrefix = null
      events.push(...emitThinkChar(frame.think, ch, config))
    }
  } else if (ch === '<') {
    frame.pendingLt = true
  } else {
    events.push(...emitThinkChar(frame.think, ch, config))
  }
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepThinkClosePrefixMatch({ frame, state, ch, config }: { frame: ThinkClosePrefixMatchFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const advanced = advancePrefixMatch(frame.prefix, ch)

  if (advanced._tag === 'Continue') {
    frame.prefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    return NOOP
  }

  if (advanced._tag === 'NoMatch') {
    // Flush as literal think content
    for (const c of advanced.literal) events.push(...emitThinkChar(frame.think, c, config))
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    return emit(...events)
  }

  // Only accept > for close tags
  if (advanced.delimiter !== '>') {
    for (const c of advanced.raw) events.push(...emitThinkChar(frame.think, c, config))
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    return emit(...events)
  }

  // Handle </lens> inside lenses
  if (frame.think.tagName === config.keywords.lenses && advanced.tagName === 'lens') {
    if (frame.think.activeLens) {
      if (frame.think.activeLens.depth > 0) {
        frame.think.activeLens.depth--
        for (const c of advanced.raw) events.push(...emitThinkChar(frame.think, c, config))
      } else {
        const content = frame.think.activeLens.content.trim()
        events.push({ _tag: 'LensEnd', name: frame.think.activeLens.name, content })
        frame.think.lenses.push({ name: frame.think.activeLens.name, content })
        frame.think.activeLens = null
      }
    } else {
      for (const c of advanced.raw) events.push(...emitThinkChar(frame.think, c, config))
    }
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    return events.length > 0 ? emit(...events) : NOOP
  }

  // Handle </think> or </thinking> or </lenses>
  if (advanced.tagName === frame.think.tagName) {
    if (frame.afterNewline) {
      if (frame.think.depth > 0) {
        frame.think.depth--
        for (const c of advanced.raw) events.push(...emitThinkChar(frame.think, c, config))
        state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
      } else {
        if (frame.think.tagName !== config.keywords.lenses) {
          events.push({ _tag: 'ProseEnd', patternId: 'think', content: frame.think.body, about: frame.think.about })
        }
        state.pop()
      }
    } else {
      state[state.length - 1] = { _tag: 'PendingThinkClose', think: frame.think, closeRaw: advanced.raw }
    }
  } else {
    for (const c of advanced.raw) events.push(...emitThinkChar(frame.think, c, config))
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

export function stepLensOpenPrefixMatch({ frame, state, ch, config }: { frame: LensOpenPrefixMatchFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const advanced = advancePrefixMatch(frame.prefix, ch)

  if (advanced._tag === 'Continue') {
    frame.prefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    return NOOP
  }

  if (advanced._tag === 'NoMatch') {
    for (const c of advanced.literal) events.push(...emitThinkChar(frame.think, c, config))
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    return emit(...events)
  }

  // Matched 'lens'
  if (advanced.delimiter === '>') {
    frame.think.activeLens = { name: '', content: '', depth: 0 }
    events.push({ _tag: 'LensStart', name: '' })
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: false }
    return emit(...events)
  }

  if (/\s/.test(advanced.delimiter)) {
    state[state.length - 1] = { _tag: 'LensTagAttrs', think: frame.think, attrKey: '', attrValue: '', phase: 'key', nameAttr: null, pendingSlash: false }
    return NOOP
  }

  if (advanced.delimiter === '/') {
    state[state.length - 1] = { _tag: 'LensTagAttrs', think: frame.think, attrKey: '', attrValue: '', phase: 'key', nameAttr: null, pendingSlash: true }
    return NOOP
  }

  // Unexpected delimiter
  for (const c of advanced.raw) events.push(...emitThinkChar(frame.think, c, config))
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
    } else if (ch === '<') {
      // Bail out: unclosed attr value. Flush as literal think content, reprocess '<'
      let reconstructed = '<lens'
      if (frame.nameAttr) reconstructed += ` name="${frame.nameAttr}"`
      reconstructed += ` ${frame.attrKey}="${frame.attrValue}`
      for (const c of reconstructed) events.push(...emitThinkChar(frame.think, c, config))
      state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: true }
      return emit(...events)
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
  if (ch === '<') {
    // Bail out: tag was never closed. Flush as literal think content, reprocess '<'
    let reconstructed = '<lens'
    if (frame.nameAttr) reconstructed += ` name="${frame.nameAttr}"`
    if (frame.attrKey) reconstructed += ` ${frame.attrKey}`
    for (const c of reconstructed) events.push(...emitThinkChar(frame.think, c, config))
    state[state.length - 1] = { _tag: 'Think', think: frame.think, pendingLt: true }
    return events.length > 0 ? emit(...events) : NOOP
  }
  if (ch === '/') { frame.pendingSlash = true; return NOOP }
  if (/\s/.test(ch)) { if (frame.attrKey.length > 0) frame.attrKey = ''; return NOOP }
  if (ch === '=') { frame.phase = 'equals'; return NOOP }
  frame.attrKey += ch
  return NOOP
}