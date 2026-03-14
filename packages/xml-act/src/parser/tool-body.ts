import type {
  AttributeValue,
  ChildAttrsFrame,
  ChildAttrValueFrame,
  ChildBodyFrame,
  ChildCloseTagFrame,
  ChildTagNameFrame,
  ChildUnquotedAttrValueFrame,
  ParseEvent,
  ParseStack,
  ParsedElement,
  ParserConfig,
  StepResult,
  ToolBodyFrame,
  ToolCloseTagFrame,
} from './types'
import { emit, mkAttrState, mkCloseTag, NOOP } from './types'
import { validateChildAttr } from './validate-attrs'

function isValidChildTag(tool: ToolBodyFrame, childTag: string, config: ParserConfig): boolean {
  if (childTag === 'ref') return config.resolveRef !== undefined
  const validSet = config.childTagMap.get(tool.tagName)
  return validSet ? validSet.has(childTag) : false
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

export function stepToolBody({ frame, state, ch }: { frame: ToolBodyFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (frame.pendingLt) {
    frame.pendingLt = false
    if (ch === '/') {
      state.push({ _tag: 'ToolCloseTag', tagName: frame.tagName, toolCallId: frame.toolCallId, attrs: frame.attrs, body: frame.body, children: frame.children, childCounts: frame.childCounts, close: mkCloseTag(), tool: frame })
    } else if (ch === '!') {
      state.push({ _tag: 'Cdata', cdata: { _tag: 'Prefix', index: 1, buffer: '<!' }, origin: frame })
    } else if (/[a-zA-Z0-9_-]/.test(ch)) {
      state.push({ _tag: 'ChildTagName', childName: ch, tool: frame })
    } else {
      frame.body += '<' + ch
      events.push({ _tag: 'BodyChunk', toolCallId: frame.toolCallId, text: '<' + ch })
    }
  } else if (ch === '<') frame.pendingLt = true
  else {
    frame.body += ch
    events.push({ _tag: 'BodyChunk', toolCallId: frame.toolCallId, text: ch })
  }
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepToolCloseTag({ frame, state, ch }: { frame: ToolCloseTagFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.close.raw += ch
  if (ch !== '>') { frame.close.name += ch; return NOOP }
  const tool = frame.tool
  if (frame.close.name === frame.tagName) {
    const element: ParsedElement = { tagName: frame.tagName, toolCallId: frame.toolCallId, attributes: new Map(frame.attrs), body: tool.body, children: [...tool.children] }
    events.push({ _tag: 'TagClosed', toolCallId: frame.toolCallId, tagName: frame.tagName, element })
    state.pop()
    state.pop()
    return emit(...events)
  }
  tool.body += frame.close.raw
  events.push({ _tag: 'BodyChunk', toolCallId: tool.toolCallId, text: frame.close.raw })
  state.pop()
  return emit(...events)
}

function flushInvalidChildToBody(frame: ChildTagNameFrame, state: ParseStack, text: string): ParseEvent[] {
  frame.tool.body += text
  state.pop()
  return [{ _tag: 'BodyChunk', toolCallId: frame.tool.toolCallId, text }]
}

function enterChildBody(tool: ToolBodyFrame, state: ParseStack, childTagName: string, attrs: Map<string, AttributeValue>): ParseEvent[] {
  const attrsCopy = new Map(attrs)
  const idx = getChildIndex(tool, childTagName)
  state[state.length - 1] = { _tag: 'ChildBody', childTagName, childAttrs: attrsCopy, childBody: '', pendingLt: false, tool }
  return [{ _tag: 'ChildOpened', parentToolCallId: tool.toolCallId, childTagName, childIndex: idx, attributes: attrsCopy }]
}

export function stepChildTagName({ frame, state, ch, config }: { frame: ChildTagNameFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (/[a-zA-Z0-9_-]/.test(ch)) frame.childName += ch
  else if (/\s/.test(ch)) {
    if (isValidChildTag(frame.tool, frame.childName, config)) state[state.length - 1] = { _tag: 'ChildAttrs', childTagName: frame.childName, attr: mkAttrState(), tool: frame.tool }
    else events.push(...flushInvalidChildToBody(frame, state, '<' + frame.childName + ch))
  } else if (ch === '>') {
    if (isValidChildTag(frame.tool, frame.childName, config)) events.push(...enterChildBody(frame.tool, state, frame.childName, new Map()))
    else events.push(...flushInvalidChildToBody(frame, state, '<' + frame.childName + '>'))
  } else if (ch === '/') {
    if (isValidChildTag(frame.tool, frame.childName, config)) {
      const attr = mkAttrState()
      attr.phase = { _tag: 'PendingSlash' }
      state[state.length - 1] = { _tag: 'ChildAttrs', childTagName: frame.childName, attr, tool: frame.tool }
    } else events.push(...flushInvalidChildToBody(frame, state, '<' + frame.childName + '/'))
  } else events.push(...flushInvalidChildToBody(frame, state, '<' + frame.childName + ch))
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepChildAttrs({ frame, state, ch, config }: { frame: ChildAttrsFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const attr = frame.attr
  const tool = frame.tool
  if (attr.phase._tag === 'PendingSlash') {
    if (ch === '>') {
      attr.phase = { _tag: 'Idle' }
      if (frame.childTagName === 'ref' && config.resolveRef) {
        const toolRef = attr.attrs.get('tool')
        if (typeof toolRef === 'string') {
          const m = /^([a-zA-Z0-9_-]+)(?:~(\d+))?$/.exec(toolRef)
          if (m) {
            const resolved = config.resolveRef(m[1], m[2] ? Number(m[2]) : 0, typeof attr.attrs.get('query') === 'string' ? String(attr.attrs.get('query')) : undefined)
            if (resolved !== undefined) {
              tool.body += resolved
              events.push({ _tag: 'BodyChunk', toolCallId: tool.toolCallId, text: resolved })
            }
          }
        }
        state.pop()
        return events.length > 0 ? emit(...events) : NOOP
      }
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
    if (ch === '/') state[state.length - 1] = { _tag: 'ChildCloseTag', childTagName: frame.childTagName, childAttrs: frame.childAttrs, childBody: frame.childBody, close: mkCloseTag(), tool: frame.tool }
    else if (ch === '!') state.push({ _tag: 'Cdata', cdata: { _tag: 'Prefix', index: 1, buffer: '<!' }, origin: frame })
    else {
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

export function stepChildCloseTag({ frame, state, ch }: { frame: ChildCloseTagFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const tool = frame.tool
  frame.close.raw += ch
  if (ch !== '>') { frame.close.name += ch; return NOOP }
  const idx = getChildIndex(tool, frame.childTagName)
  if (frame.close.name === frame.childTagName) {
    events.push({ _tag: 'ChildComplete', parentToolCallId: tool.toolCallId, childTagName: frame.close.name, childIndex: idx, attributes: frame.childAttrs, body: frame.childBody })
    tool.children.push({ tagName: frame.close.name, attributes: new Map(frame.childAttrs), body: frame.childBody })
    incrementChildIndex(tool, frame.close.name)
    state.pop()
    return emit(...events)
  }
  if (frame.close.name === tool.tagName) {
    events.push({ _tag: 'ParseError', error: { _tag: 'UnclosedChildTag', toolCallId: tool.toolCallId, tagName: tool.tagName, childTagName: frame.childTagName, detail: `Child tag <${frame.childTagName}> inside <${tool.tagName}> was never closed` } })
    const element: ParsedElement = { tagName: tool.tagName, toolCallId: tool.toolCallId, attributes: new Map(tool.attrs), body: tool.body, children: [...tool.children] }
    events.push({ _tag: 'TagClosed', toolCallId: tool.toolCallId, tagName: tool.tagName, element })
    state.pop()
    state.pop()
    return emit(...events)
  }
  frame.childBody += frame.close.raw
  events.push({ _tag: 'ChildBodyChunk', parentToolCallId: tool.toolCallId, childTagName: frame.childTagName, childIndex: idx, text: frame.close.raw })
  state[state.length - 1] = { _tag: 'ChildBody', childTagName: frame.childTagName, childAttrs: frame.childAttrs, childBody: frame.childBody, pendingLt: false, tool }
  return emit(...events)
}