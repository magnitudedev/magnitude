import type { FenceState, ParseEvent, ParseStack, ParserConfig, ProseFrame, StepResult } from './types'
import { emit, FencePhase, NOOP } from './types'

export function isFenceComplete(phase: FencePhase): boolean {
  return phase === FencePhase.Tick3 || phase === FencePhase.XML || phase === FencePhase.TrailingWs
}

export function rawEmitProse(state: ParseStack, text: string): ParseEvent[] {
  const prose = state[0]
  if (text.length === 0) return []
  const events: ParseEvent[] = []
  events.push(...flushDeferredFence(state))
  events.push({ _tag: 'ProseChunk', patternId: 'prose', text })
  prose.proseAccum += text
  return events
}

export function flushDeferredFence(state: ParseStack): ParseEvent[] {
  const prose = state[0]
  if (prose.fence.deferred.length === 0) return []
  const text = prose.fence.deferred
  prose.fence.deferred = ''
  prose.proseAccum += text
  return [{ _tag: 'ProseChunk', patternId: 'prose', text }]
}

export function flushPendingWhitespace(state: ParseStack): ParseEvent[] {
  const prose = state[0]
  if (prose.fence.pendingWhitespace.length === 0) return []
  const ws = prose.fence.pendingWhitespace
  prose.fence.pendingWhitespace = ''
  return rawEmitProse(state, ws)
}

export function flushFenceBuffer(state: ParseStack): ParseEvent[] {
  const prose = state[0]
  if (prose.fence.buffer.length === 0) return []
  const events: ParseEvent[] = []
  events.push(...flushPendingWhitespace(state))
  const text = prose.fence.buffer
  prose.fence.buffer = ''
  events.push(...rawEmitProse(state, text))
  return events
}

export function resetFence(fence: FenceState): void {
  fence.phase = FencePhase.LeadingWs
  fence.buffer = ''
}

export function emitProseChunk(state: ParseStack, text: string): ParseEvent[] {
  if (text.length === 0) return []
  const events: ParseEvent[] = []
  events.push(...flushFenceBuffer(state))
  events.push(...flushPendingWhitespace(state))
  events.push(...rawEmitProse(state, text))
  return events
}

export function endProseBlock(state: ParseStack): ParseEvent[] {
  const prose = state[0]
  const fence = prose.fence
  const events: ParseEvent[] = []
  if (fence.phase !== FencePhase.Broken && isFenceComplete(fence.phase)) {
    fence.buffer = ''
    fence.pendingWhitespace = ''
  } else {
    events.push(...flushFenceBuffer(state))
  }
  fence.deferred = ''
  fence.pendingWhitespace = ''
  if (prose.proseAccum.length > 0) {
    const content = prose.proseAccum.trim()
    if (content.length > 0) events.push({ _tag: 'ProseEnd', patternId: 'prose', content, about: null })
    prose.proseAccum = ''
  }
  resetFence(fence)
  return events
}

function breakFence(state: ParseStack, ch: string): ParseEvent[] {
  const prose = state[0]
  prose.justClosedStructural = false
  prose.fence.buffer += ch
  prose.fence.phase = FencePhase.Broken
  return flushFenceBuffer(state)
}

export function appendProseChar(state: ParseStack, ch: string): ParseEvent[] {
  const prose = state[0]
  const fence = prose.fence
  const events: ParseEvent[] = []

  if (ch === '\n') {
    if (fence.phase !== FencePhase.Broken && isFenceComplete(fence.phase)) {
      if (prose.justClosedStructural) {
        fence.buffer = ''
        fence.pendingWhitespace = ''
        prose.justClosedStructural = false
        resetFence(fence)
      } else {
        fence.deferred = fence.pendingWhitespace + fence.buffer + '\n'
        fence.pendingWhitespace = ''
        fence.buffer = ''
        resetFence(fence)
      }
    } else {
      events.push(...flushFenceBuffer(state))
      fence.pendingWhitespace += '\n'
      resetFence(fence)
    }
    return events
  }

  if (fence.pendingWhitespace.length > 0 && (ch === ' ' || ch === '\t' || ch === '\r')) {
    fence.pendingWhitespace += ch
    return events
  }

  if (fence.phase === FencePhase.Broken) {
    events.push(...flushPendingWhitespace(state))
    events.push(...rawEmitProse(state, ch))
    return events
  }

  switch (fence.phase) {
    case FencePhase.LeadingWs:
      if (ch === ' ' || ch === '\t') fence.buffer += ch
      else if (ch === '`') { fence.buffer += ch; fence.phase = FencePhase.Tick1 }
      else events.push(...breakFence(state, ch))
      break
    case FencePhase.Tick1:
      if (ch === '`') { fence.buffer += ch; fence.phase = FencePhase.Tick2 }
      else events.push(...breakFence(state, ch))
      break
    case FencePhase.Tick2:
      if (ch === '`') { fence.buffer += ch; fence.phase = FencePhase.Tick3 }
      else events.push(...breakFence(state, ch))
      break
    case FencePhase.Tick3:
      if (ch === 'x' || ch === 'X') { fence.buffer += ch; fence.phase = FencePhase.X }
      else if (ch === ' ' || ch === '\t') { fence.buffer += ch; fence.phase = FencePhase.TrailingWs }
      else events.push(...breakFence(state, ch))
      break
    case FencePhase.X:
      if (ch === 'm' || ch === 'M') { fence.buffer += ch; fence.phase = FencePhase.XM }
      else events.push(...breakFence(state, ch))
      break
    case FencePhase.XM:
      if (ch === 'l' || ch === 'L') { fence.buffer += ch; fence.phase = FencePhase.XML }
      else events.push(...breakFence(state, ch))
      break
    case FencePhase.XML:
    case FencePhase.TrailingWs:
      if (ch === ' ' || ch === '\t') { fence.buffer += ch; fence.phase = FencePhase.TrailingWs }
      else events.push(...breakFence(state, ch))
      break
  }

  return events
}

export function stepProse({ frame, state, ch }: { frame: ProseFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (ch === '<') {
    events.push(...flushFenceBuffer(state))
    state.push({ _tag: 'TagName', name: '', raw: '<', afterNewline: frame.lastCharNewline })
    frame.lastCharNewline = false
  } else {
    frame.lastCharNewline = ch === '\n'
    events.push(...appendProseChar(state, ch))
  }
  return events.length > 0 ? emit(...events) : NOOP
}