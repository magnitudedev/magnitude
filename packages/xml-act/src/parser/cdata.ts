import type { CdataFrame, ParseEvent, ParseStack, ParserConfig, StepResult } from './types'
import { emit, NOOP } from './types'
import { appendProseChar } from './prose'

const CDATA_PREFIX = '![CDATA['

export function stepCdata({ frame, state, ch, config }: { frame: CdataFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const cdata = frame.cdata
  if (cdata._tag === 'Prefix') {
    if (ch === CDATA_PREFIX[cdata.index]) {
      cdata.buffer += ch
      cdata.index++
      if (cdata.index === CDATA_PREFIX.length) frame.cdata = { _tag: 'Body', buffer: '', closeBrackets: 0 }
    } else events.push(...returnFromCdata(frame, state, cdata.buffer + ch))
    return events.length > 0 ? emit(...events) : NOOP
  }
  if (ch === ']') cdata.closeBrackets++
  else if (ch === '>' && cdata.closeBrackets >= 2) events.push(...returnFromCdata(frame, state, cdata.buffer))
  else {
    if (cdata.closeBrackets > 0) {
      cdata.buffer += ']'.repeat(cdata.closeBrackets)
      cdata.closeBrackets = 0
    }
    cdata.buffer += ch
  }
  return events.length > 0 ? emit(...events) : NOOP
}

export function returnFromCdata(frame: CdataFrame, state: ParseStack, text: string): ParseEvent[] {
  const events: ParseEvent[] = []
  state.pop()
  switch (frame.origin._tag) {
    case 'Prose':
      for (const c of text) events.push(...appendProseChar(state, c))
      return events
    case 'ToolBody':
      frame.origin.body += text
      events.push({ _tag: 'BodyChunk', toolCallId: frame.origin.toolCallId, text })
      return events
    case 'ChildBody': {
      const idx = frame.origin.tool.childCounts.get(frame.origin.childTagName) ?? 0
      frame.origin.childBody += text
      events.push({ _tag: 'ChildBodyChunk', parentToolCallId: frame.origin.tool.toolCallId, childTagName: frame.origin.childTagName, childIndex: idx, text })
      return events
    }
  }
}