import type {
  AttributeValue,
  ChildAttrsFrame,
  ChildAttrValueFrame,
  ChildBodyFrame,
  ChildClosePrefixMatchFrame,
  ChildOpenPrefixMatchFrame,
  ChildUnquotedAttrValueFrame,
  ParseEvent,
  ParseStack,
  ParsedElement,
  ParserConfig,
  StepResult,
  ToolBodyFrame,
  ToolClosePrefixMatchFrame,
} from './types'
import { emit, mkAttrState, NOOP } from './types'
import { validateChildAttr } from './validate-attrs'
import { advancePrefixMatch, candidatesStartingWith } from './prefix-match'

function isValidChildTag(tool: ToolBodyFrame, childTag: string, config: ParserConfig): boolean {
  const validSet = config.childTagMap.get(tool.tagName)
  return validSet ? validSet.has(childTag) : false
}

function childOpenCandidates(tool: ToolBodyFrame, config: ParserConfig): string[] {
  const candidates: string[] = []
  const validSet = config.childTagMap.get(tool.tagName)
  if (validSet) candidates.push(...validSet)
  return candidates
}

function getChildIndex(tool: ToolBodyFrame, childTag: string): number {
  return tool.childCounts.get(childTag) ?? 0
}

function incrementChildIndex(tool: ToolBodyFrame, childTag: string): void {
  tool.childCounts.set(childTag, (tool.childCounts.get(childTag) ?? 0) + 1)
}

function finalizeChildAttrVal(tool: ToolBodyFrame, config: ParserConfig, childTagName: string, attr: { attrs: Map<string, AttributeValue>; hasError: boolean }, key: string, raw: string): ParseEvent[] {
  if (attr.hasError) return []
  const childSchema = config.tagSchemas?.get(tool.tagName)?.children.get(childTagName)
  if (!childSchema) {
    attr.attrs.set(key, raw)
    return []
  }
  const result = validateChildAttr(tool.tagName, childTagName, childSchema, key, raw)
  if (result.ok) {
    attr.attrs.set(key, result.value)
    return []
  }
  attr.hasError = true
  return [{ _tag: 'ParseError', error: { ...result.error, toolCallId: tool.toolCallId, tagName: tool.tagName } }]
}

function enterChildBody(tool: ToolBodyFrame, state: ParseStack, childTagName: string, attrs: Map<string, AttributeValue>): ParseEvent[] {
  const attrsCopy = new Map(attrs)
  const idx = getChildIndex(tool, childTagName)
  state[state.length - 1] = { _tag: 'ChildBody', childTagName, childAttrs: attrsCopy, childBody: '', pendingLt: false, tool }
  return [{ _tag: 'ChildOpened', parentToolCallId: tool.toolCallId, childTagName, childIndex: idx, attributes: attrsCopy }]
}

export function stepToolBody({ frame, state, ch, config }: { frame: ToolBodyFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (frame.pendingLt) {
    frame.pendingLt = false
    if (ch === '/') {
      state.push({
        _tag: 'ToolClosePrefixMatch',
        tool: frame,
        prefix: { candidates: [frame.tagName], matched: '', raw: '</' },
      })
    } else if (ch === '!') {
      state.push({ _tag: 'Cdata', cdata: { _tag: 'Prefix', index: 1, buffer: '<!' }, origin: frame })
    } else {
      const candidates = candidatesStartingWith(childOpenCandidates(frame, config), ch)
      if (candidates.length > 0) {
        state.push({
          _tag: 'ChildOpenPrefixMatch',
          tool: frame,
          prefix: { candidates, matched: ch, raw: '<' + ch },
        })
      } else {
        frame.body += '<' + ch
        events.push({ _tag: 'BodyChunk', toolCallId: frame.toolCallId, text: '<' + ch })
      }
    }
  } else if (ch === '<') frame.pendingLt = true
  else {
    frame.body += ch
    events.push({ _tag: 'BodyChunk', toolCallId: frame.toolCallId, text: ch })
  }
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepToolClosePrefixMatch({ frame, state, ch }: { frame: ToolClosePrefixMatchFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const advanced = advancePrefixMatch(frame.prefix, ch)

  if (advanced._tag === 'Continue') {
    frame.prefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    return NOOP
  }

  if (advanced._tag === 'NoMatch') {
    frame.tool.body += advanced.literal
    events.push({ _tag: 'BodyChunk', toolCallId: frame.tool.toolCallId, text: advanced.literal })
    state.pop()
    return emit(...events)
  }

  // Only accept > for tool close
  if (advanced.delimiter !== '>') {
    frame.tool.body += advanced.raw
    events.push({ _tag: 'BodyChunk', toolCallId: frame.tool.toolCallId, text: advanced.raw })
    state.pop()
    return emit(...events)
  }

  // Matched tool close tag
  const tool = frame.tool
  const element: ParsedElement = { tagName: tool.tagName, toolCallId: tool.toolCallId, attributes: new Map(tool.attrs), body: tool.body, children: [...tool.children] }
  events.push({ _tag: 'TagClosed', toolCallId: tool.toolCallId, tagName: tool.tagName, element })
  state.pop() // pop ToolClosePrefixMatch
  state.pop() // pop ToolBody
  return emit(...events)
}

export function stepChildOpenPrefixMatch({ frame, state, ch, config }: { frame: ChildOpenPrefixMatchFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const advanced = advancePrefixMatch(frame.prefix, ch)

  if (advanced._tag === 'Continue') {
    frame.prefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    return NOOP
  }

  if (advanced._tag === 'NoMatch') {
    frame.tool.body += advanced.literal
    events.push({ _tag: 'BodyChunk', toolCallId: frame.tool.toolCallId, text: advanced.literal })
    state.pop()
    return emit(...events)
  }

  // Matched a child tag
  const childTagName = advanced.tagName
  if (advanced.delimiter === '>') {
    events.push(...enterChildBody(frame.tool, state, childTagName, new Map()))
  } else if (/\s/.test(advanced.delimiter)) {
    state[state.length - 1] = { _tag: 'ChildAttrs', childTagName, attr: mkAttrState(), tool: frame.tool } as any
  } else if (advanced.delimiter === '/') {
    const attr = mkAttrState()
    attr.phase = { _tag: 'PendingSlash' }
    state[state.length - 1] = { _tag: 'ChildAttrs', childTagName, attr, tool: frame.tool } as any
  } else {
    frame.tool.body += advanced.raw
    events.push({ _tag: 'BodyChunk', toolCallId: frame.tool.toolCallId, text: advanced.raw })
    state.pop()
  }
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepChildAttrs({ frame, state, ch, config }: { frame: ChildAttrsFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const attr = frame.attr
  const tool = frame.tool
  if (attr.phase._tag === 'PendingSlash') {
    if (ch === '>') {
      attr.phase = { _tag: 'Idle' }
      const attrsCopy = new Map(attr.attrs)
      const idx = getChildIndex(tool, frame.childTagName)
      events.push({ _tag: 'ChildOpened', parentToolCallId: tool.toolCallId, childTagName: frame.childTagName, childIndex: idx, attributes: attrsCopy })
      events.push({ _tag: 'ChildComplete', parentToolCallId: tool.toolCallId, childTagName: frame.childTagName, childIndex: idx, attributes: attrsCopy, body: '' })
      tool.children.push({ tagName: frame.childTagName, attributes: attrsCopy, body: '' })
      incrementChildIndex(tool, frame.childTagName)
      state.pop()
    } else {
      attr.key += '/'
      attr.phase = { _tag: 'Idle' }
      return stepChildAttrs({ frame, state, ch, config })
    }
    return events.length > 0 ? emit(...events) : NOOP
  }
  if (attr.phase._tag === 'PendingEquals') {
    const key = attr.phase.key
    attr.phase = { _tag: 'Idle' }
    if (ch === '"') {
      attr.value = ''
      attr.key = key
      state[state.length - 1] = { _tag: 'ChildAttrValue', childTagName: frame.childTagName, attr, tool }
    } else if (ch === '>' || ch === '/') {
      events.push(...finalizeChildAttrVal(tool, config, frame.childTagName, attr, key, ''))
      attr.key = ''
      return stepChildAttrs({ frame, state, ch, config })
    } else if (/\s/.test(ch)) {
      events.push(...finalizeChildAttrVal(tool, config, frame.childTagName, attr, key, ''))
      attr.key = ''
    } else {
      attr.value = ch
      attr.key = key
      state[state.length - 1] = { _tag: 'ChildUnquotedAttrValue', childTagName: frame.childTagName, attr, tool }
    }
    return events.length > 0 ? emit(...events) : NOOP
  }
  if (ch === '>') {
    if (attr.key) {
      events.push(...finalizeChildAttrVal(tool, config, frame.childTagName, attr, attr.key, ''))
      attr.key = ''
    }
    events.push(...enterChildBody(tool, state, frame.childTagName, attr.attrs))
  } else if (ch === '/') {
    if (attr.key) {
      events.push(...finalizeChildAttrVal(tool, config, frame.childTagName, attr, attr.key, ''))
      attr.key = ''
    }
    attr.phase = { _tag: 'PendingSlash' }
  } else if (ch === '=') {
    attr.phase = { _tag: 'PendingEquals', key: attr.key }
    attr.key = ''
  } else if (ch === '"') {
    attr.value = ''
    state[state.length - 1] = { _tag: 'ChildAttrValue', childTagName: frame.childTagName, attr, tool }
  } else if (/\s/.test(ch)) {
    if (attr.key) {
      events.push(...finalizeChildAttrVal(tool, config, frame.childTagName, attr, attr.key, ''))
      attr.key = ''
    }
  } else if (ch === '<') {
    // Bail out: tag was never closed. Flush as literal body text, reprocess '<'
    let reconstructed = '<' + frame.childTagName
    for (const [k, v] of attr.attrs) reconstructed += ` ${k}="${v}"`
    if (attr.key) reconstructed += ` ${attr.key}`
    frame.tool.body += reconstructed
    events.push({ _tag: 'BodyChunk', toolCallId: frame.tool.toolCallId, text: reconstructed })
    state.pop()
    frame.tool.pendingLt = true
  } else attr.key += ch
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepChildAttrValue({ frame, state, ch, config }: { frame: ChildAttrValueFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (ch === '"') {
    events.push(...finalizeChildAttrVal(frame.tool, config, frame.childTagName, frame.attr, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    frame.attr.phase = { _tag: 'Idle' }
    state[state.length - 1] = { _tag: 'ChildAttrs', childTagName: frame.childTagName, attr: frame.attr, tool: frame.tool }
  } else frame.attr.value += ch
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepChildUnquotedAttrValue({ frame, state, ch, config }: { frame: ChildUnquotedAttrValueFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (/\s/.test(ch)) {
    events.push(...finalizeChildAttrVal(frame.tool, config, frame.childTagName, frame.attr, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    frame.attr.phase = { _tag: 'Idle' }
    state[state.length - 1] = { _tag: 'ChildAttrs', childTagName: frame.childTagName, attr: frame.attr, tool: frame.tool }
  } else if (ch === '>') {
    events.push(...finalizeChildAttrVal(frame.tool, config, frame.childTagName, frame.attr, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    events.push(...enterChildBody(frame.tool, state, frame.childTagName, frame.attr.attrs))
  } else if (ch === '/') {
    events.push(...finalizeChildAttrVal(frame.tool, config, frame.childTagName, frame.attr, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    frame.attr.phase = { _tag: 'PendingSlash' }
    state[state.length - 1] = { _tag: 'ChildAttrs', childTagName: frame.childTagName, attr: frame.attr, tool: frame.tool }
  } else frame.attr.value += ch
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepChildBody({ frame, state, ch }: { frame: ChildBodyFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const idx = getChildIndex(frame.tool, frame.childTagName)
  if (frame.pendingLt) {
    frame.pendingLt = false
    if (ch === '/') {
      state[state.length - 1] = {
        _tag: 'ChildClosePrefixMatch',
        childTagName: frame.childTagName,
        childAttrs: frame.childAttrs,
        childBody: frame.childBody,
        tool: frame.tool,
        prefix: { candidates: [frame.childTagName, frame.tool.tagName], matched: '', raw: '</' },
      }
    } else if (ch === '!') {
      state.push({ _tag: 'Cdata', cdata: { _tag: 'Prefix', index: 1, buffer: '<!' }, origin: frame })
    } else {
      frame.childBody += '<' + ch
      events.push({ _tag: 'ChildBodyChunk', parentToolCallId: frame.tool.toolCallId, childTagName: frame.childTagName, childIndex: idx, text: '<' + ch })
    }
  } else if (ch === '<') frame.pendingLt = true
  else {
    frame.childBody += ch
    events.push({ _tag: 'ChildBodyChunk', parentToolCallId: frame.tool.toolCallId, childTagName: frame.childTagName, childIndex: idx, text: ch })
  }
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepChildClosePrefixMatch({ frame, state, ch }: { frame: ChildClosePrefixMatchFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const tool = frame.tool
  const advanced = advancePrefixMatch(frame.prefix, ch)

  if (advanced._tag === 'Continue') {
    frame.prefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    return NOOP
  }

  const idx = getChildIndex(tool, frame.childTagName)

  if (advanced._tag === 'NoMatch') {
    frame.childBody += advanced.literal
    events.push({ _tag: 'ChildBodyChunk', parentToolCallId: tool.toolCallId, childTagName: frame.childTagName, childIndex: idx, text: advanced.literal })
    state[state.length - 1] = { _tag: 'ChildBody', childTagName: frame.childTagName, childAttrs: frame.childAttrs, childBody: frame.childBody, pendingLt: false, tool }
    return emit(...events)
  }

  // Only accept >
  if (advanced.delimiter !== '>') {
    frame.childBody += advanced.raw
    events.push({ _tag: 'ChildBodyChunk', parentToolCallId: tool.toolCallId, childTagName: frame.childTagName, childIndex: idx, text: advanced.raw })
    state[state.length - 1] = { _tag: 'ChildBody', childTagName: frame.childTagName, childAttrs: frame.childAttrs, childBody: frame.childBody, pendingLt: false, tool }
    return emit(...events)
  }

  if (advanced.tagName === frame.childTagName) {
    // Child close
    events.push({ _tag: 'ChildComplete', parentToolCallId: tool.toolCallId, childTagName: frame.childTagName, childIndex: idx, attributes: frame.childAttrs, body: frame.childBody })
    tool.children.push({ tagName: frame.childTagName, attributes: new Map(frame.childAttrs), body: frame.childBody })
    incrementChildIndex(tool, frame.childTagName)
    state.pop()
    return emit(...events)
  }

  if (advanced.tagName === tool.tagName) {
    // Tool close found inside child — unclosed child error
    events.push({ _tag: 'ParseError', error: { _tag: 'UnclosedChildTag', toolCallId: tool.toolCallId, tagName: tool.tagName, childTagName: frame.childTagName, detail: `Child tag <${frame.childTagName}> inside <${tool.tagName}> was never closed` } })
    const element: ParsedElement = { tagName: tool.tagName, toolCallId: tool.toolCallId, attributes: new Map(tool.attrs), body: tool.body, children: [...tool.children] }
    events.push({ _tag: 'TagClosed', toolCallId: tool.toolCallId, tagName: tool.tagName, element })
    state.pop() // pop ChildClosePrefixMatch
    state.pop() // pop ToolBody
    return emit(...events)
  }

  // Unrecognized close tag — literal
  frame.childBody += advanced.raw
  events.push({ _tag: 'ChildBodyChunk', parentToolCallId: tool.toolCallId, childTagName: frame.childTagName, childIndex: idx, text: advanced.raw })
  state[state.length - 1] = { _tag: 'ChildBody', childTagName: frame.childTagName, childAttrs: frame.childAttrs, childBody: frame.childBody, pendingLt: false, tool }
  return emit(...events)
}