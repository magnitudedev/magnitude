import type {
  AttributeValue,
  ParseEvent,
  ParseStack,
  ParsedElement,
  PendingStructuralOpenFrame,
  PendingTopLevelCloseFrame,
  ParserConfig,
  ProseFrame,
  StepResult,
  TagAttrsFrame,
  TagAttrValueFrame,
  TagNameFrame,
  TagUnquotedAttrValueFrame,
  TopLevelCloseTagFrame,
} from './types'
import { emit, emitAndReprocess, mkAttrState, mkRootProse, NOOP, reprocess } from './types'
import { validateToolAttr } from './validate-attrs'
import { activeTags, containerDepth } from './stack-ops'
import { emitProseChunk, endProseBlock } from './prose'

export function isPrefixOfAny(prefix: string, tags: ReadonlySet<string>): boolean {
  for (const tag of tags) if (tag.startsWith(prefix)) return true
  return false
}

export function parseToolRef(raw: string): { tag: string; recency: number } | undefined {
  const match = /^([a-zA-Z0-9_-]+)(?:~(\d+))?$/.exec(raw)
  if (!match) return undefined
  return { tag: match[1], recency: match[2] ? Number(match[2]) : 0 }
}

export function finalizeToolAttr(config: ParserConfig, tagName: string, toolCallId: string, attr: { attrs: Map<string, AttributeValue>; hasError: boolean }, key: string, raw: string): ParseEvent[] {
  if (attr.hasError) return []
  if (!config.knownTags.has(tagName)) {
    attr.attrs.set(key, raw)
    return []
  }
  const schema = config.tagSchemas?.get(tagName)
  if (!schema) {
    attr.attrs.set(key, raw)
    return []
  }
  const result = validateToolAttr(tagName, schema, key, raw)
  if (result.ok) {
    attr.attrs.set(key, result.value)
    return []
  }
  attr.hasError = true
  return [{ _tag: 'ParseError', error: { ...result.error, toolCallId, tagName } }]
}

export function resolveOpenTag(state: ParseStack, tagName: string, toolCallId: string, attrs: Map<string, AttributeValue>, raw: string, config: ParserConfig): ParseEvent[] {
  const events: ParseEvent[] = []
  const proseRoot = state[0]
  const kw = config.keywords
  if (tagName === kw.actions) {
    events.push(...endProseBlock(state))
    const outermost = containerDepth(state, 'Actions') === 0
    state.push({ _tag: 'Actions' })
    {
      const prose = mkRootProse()
      prose.lastCharNewline = false
      state.push(prose)
    }
    proseRoot.lastCharNewline = false
    if (outermost) events.push({ _tag: 'ActionsOpen' })
    return events
  }
  if (tagName === 'inspect') {
    events.push(...endProseBlock(state))
    const outermost = containerDepth(state, 'Inspect') === 0
    state.push({ _tag: 'Inspect' })
    {
      const prose = mkRootProse()
      prose.lastCharNewline = false
      state.push(prose)
    }
    proseRoot.lastCharNewline = false
    if (outermost) events.push({ _tag: 'InspectOpen' })
    return events
  }
  if (tagName === kw.comms) {
    events.push(...endProseBlock(state))
    const outermost = containerDepth(state, 'Comms') === 0
    state.push({ _tag: 'Comms' })
    {
      const prose = mkRootProse()
      prose.lastCharNewline = false
      state.push(prose)
    }
    proseRoot.lastCharNewline = false
    if (outermost) events.push({ _tag: 'CommsOpen' })
    return events
  }
  if (tagName === kw.think || tagName === kw.thinking || tagName === kw.lenses) {
    events.push(...endProseBlock(state))
    proseRoot.lastCharNewline = false
    const aboutValue = attrs.get('about')
    const about = tagName === kw.lenses ? null : typeof aboutValue === 'string' ? aboutValue : null
    state.push({ _tag: 'Think', think: { tagName, body: '', depth: 0, openTagBuf: '', openAfterNewline: false, lastCharNewline: false, about, lenses: [], activeLens: null }, pendingLt: false })
    return events
  }
  if (containerDepth(state, 'Inspect') === 0 && tagName === 'message') {
    events.push(...endProseBlock(state))
    const id = config.generateId().slice(0, 8)
    const toValue = attrs.get('to')
    const artifactsValue = attrs.get('artifacts')
    const dest = typeof toValue === 'string' ? toValue : config.defaultMessageDest
    const artifactsRaw = typeof artifactsValue === 'string' ? artifactsValue : null
    events.push({ _tag: 'MessageTagOpen', id, dest, artifactsRaw })
    state.push({ _tag: 'MessageBody', id, dest, artifactsRaw, body: '', pendingLt: false, depth: 0, pendingNewline: false })
    return events
  }
  if (containerDepth(state, 'Inspect') === 0 && containerDepth(state, 'Comms') === 0 && config.knownTags.has(tagName)) {
    events.push(...endProseBlock(state))
    if (!toolCallId) toolCallId = config.generateId()
    events.push({ _tag: 'TagOpened', tagName, toolCallId, attributes: new Map(attrs) })
    state.push({ _tag: 'ToolBody', tagName, toolCallId, attrs, body: '', children: [], childCounts: new Map(), pendingLt: false })
    return events
  }
  events.push(...emitProseChunk(state, raw))
  return events
}

export function resolveSelfClose(state: ParseStack, tagName: string, toolCallId: string, attrs: Map<string, AttributeValue>, raw: string, config: ParserConfig): ParseEvent[] {
  const events: ParseEvent[] = []
  const kw = config.keywords
  if (tagName === kw.actions || tagName === 'inspect' || tagName === kw.comms) return events
  if (containerDepth(state, 'Actions') === 0 && containerDepth(state, 'Inspect') === 0 && containerDepth(state, 'Comms') === 0 && (tagName === config.keywords.next || tagName === config.keywords.yield)) {
    const decision: 'continue' | 'yield' = tagName === config.keywords.next ? 'continue' : 'yield'
    events.push({ _tag: 'TurnControl', decision })
    state.push({ _tag: 'Done' })
    return events
  }
  if (containerDepth(state, 'Inspect') > 0 && tagName === 'ref') {
    const toolRef = attrs.get('tool')
    if (typeof toolRef === 'string') {
      const parsed = parseToolRef(toolRef)
      const query = attrs.get('query')
      if (parsed) {
        if (config.resolveRef) {
          const queryText = typeof query === 'string' ? query : undefined
          const resolved = config.resolveRef(parsed.tag, parsed.recency, queryText)
          if (resolved !== undefined) events.push({ _tag: 'InspectResult', toolRef, query: queryText, content: resolved })
          else events.push({ _tag: 'ParseError', error: { _tag: 'InvalidRef', toolRef, detail: `Ref "${toolRef}" does not match any tool result from this response` } })
        } else if (parsed.tag !== 'fs-write') {
          events.push({ _tag: 'ParseError', error: { _tag: 'InvalidRef', toolRef, detail: `Ref "${toolRef}" does not match any tool result from this response` } })
        }
      } else {
        events.push({ _tag: 'ParseError', error: { _tag: 'InvalidRef', toolRef, detail: `Invalid tool ref "${toolRef}". Expected format "tag" or "tag~N"` } })
      }
    }
    return events
  }
  if (containerDepth(state, 'Inspect') === 0 && tagName === 'message') {
    const id = config.generateId().slice(0, 8)
    const toValue = attrs.get('to')
    const artifactsValue = attrs.get('artifacts')
    const dest = typeof toValue === 'string' ? toValue : config.defaultMessageDest
    const artifactsRaw = typeof artifactsValue === 'string' ? artifactsValue : null
    events.push({ _tag: 'MessageTagOpen', id, dest, artifactsRaw })
    events.push({ _tag: 'MessageTagClose', id })
    return events
  }
  if (containerDepth(state, 'Inspect') === 0 && containerDepth(state, 'Comms') === 0 && config.knownTags.has(tagName)) {
    events.push(...endProseBlock(state))
    if (!toolCallId) toolCallId = config.generateId()
    const attrsCopy = new Map(attrs)
    events.push({ _tag: 'TagOpened', tagName, toolCallId, attributes: attrsCopy })
    const element: ParsedElement = { tagName, toolCallId, attributes: attrsCopy, body: '', children: [] }
    events.push({ _tag: 'TagClosed', toolCallId, tagName, element })
    return events
  }
  events.push(...emitProseChunk(state, raw))
  return events
}

export function stepTagName({ frame, state, ch, config }: { frame: TagNameFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.raw += ch
  if (frame.name.length === 0 && ch === '/') {
    state[state.length - 1] = { _tag: 'TopLevelCloseTag', close: { name: '', raw: '</' }, afterNewline: frame.afterNewline }
  } else if (frame.name.length === 0 && ch === '!') {
    state.push({ _tag: 'Cdata', cdata: { _tag: 'Prefix', index: 1, buffer: '<!' }, origin: state[0] })
  } else if (/[a-zA-Z0-9_-]/.test(ch)) {
    const candidate = frame.name + ch
    if (isPrefixOfAny(candidate, activeTags(state, frame.afterNewline, config)) || isPrefixOfAny(candidate, config.structuralTags)) frame.name = candidate
    else {
      state.pop()
      events.push(...emitProseChunk(state, '<' + candidate))
    }
  } else if (frame.name.length === 0) {
    state.pop()
    events.push(...emitProseChunk(state, '<' + ch))
  } else if (/\s/.test(ch)) {
    const tags = activeTags(state, frame.afterNewline, config)
    if (tags.has(frame.name)) {
      const toolCallId = config.knownTags.has(frame.name) ? config.generateId() : ''
      state[state.length - 1] = { _tag: 'TagAttrs', tagName: frame.name, toolCallId, attr: mkAttrState(), raw: frame.raw }
    } else {
      state.pop()
      events.push(...emitProseChunk(state, '<' + frame.name + ch))
    }
  } else if (ch === '>') {
    const tags = activeTags(state, frame.afterNewline, config)
    if (tags.has(frame.name)) {
      state.pop()
      events.push(...resolveOpenTag(state, frame.name, '', new Map(), frame.raw, config))
    } else if (!frame.afterNewline && (frame.name === config.keywords.think || frame.name === config.keywords.thinking || frame.name === config.keywords.lenses)) {
      state[state.length - 1] = { _tag: 'PendingStructuralOpen', tagName: frame.name, raw: frame.raw }
    } else {
      state.pop()
      events.push(...emitProseChunk(state, '<' + frame.name + '>'))
    }
  } else if (ch === '/') {
    const tags = activeTags(state, frame.afterNewline, config)
    if (tags.has(frame.name)) {
      const toolCallId = config.knownTags.has(frame.name) ? config.generateId() : ''
      const attr = mkAttrState()
      attr.phase = { _tag: 'PendingSlash' }
      state[state.length - 1] = { _tag: 'TagAttrs', tagName: frame.name, toolCallId, attr, raw: frame.raw }
    } else {
      state.pop()
      events.push(...emitProseChunk(state, '<' + frame.name + '/'))
    }
  } else {
    state.pop()
    events.push(...emitProseChunk(state, '<' + frame.name + ch))
  }
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepTopLevelCloseTag({ frame, state, ch, config }: { frame: TopLevelCloseTagFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.close.raw += ch
  if (ch !== '>') { frame.close.name += ch; return NOOP }
  const proseRoot = state[0]
  const kw = config.keywords
  const closeIf = (name: string, kind: 'Actions' | 'Inspect' | 'Comms', ev: 'ActionsClose' | 'InspectClose' | 'CommsClose') => {
    if (frame.afterNewline && frame.close.name === name && containerDepth(state, kind) > 0) {
      state.pop()
      if (state[state.length - 1]?._tag === 'Prose') state.pop()
      for (let i = state.length - 1; i >= 0; i--) {
        if (state[i]?._tag === kind) { state.splice(i, 1); break }
      }
      proseRoot.justClosedStructural = true
      proseRoot.lastCharNewline = false
      if (containerDepth(state, kind) === 0) events.push({ _tag: ev })
      const prose = mkRootProse()
      prose.justClosedStructural = true
      prose.lastCharNewline = false
      state.push(prose)
      return true
    }
    if (!frame.afterNewline && frame.close.name === name && containerDepth(state, kind) > 0) {
      state[state.length - 1] = { _tag: 'PendingTopLevelClose', tagName: name, closeRaw: frame.close.raw }
      return true
    }
    return false
  }
  if (closeIf(kw.actions, 'Actions', 'ActionsClose')) return events.length > 0 ? emit(...events) : NOOP
  if (closeIf('inspect', 'Inspect', 'InspectClose')) return events.length > 0 ? emit(...events) : NOOP
  if (closeIf(kw.comms, 'Comms', 'CommsClose')) return events.length > 0 ? emit(...events) : NOOP
  state.pop()
  events.push(...emitProseChunk(state, frame.close.raw))
  return emit(...events)
}

export function stepTagAttrs({ frame, state, ch, config }: { frame: TagAttrsFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.raw += ch
  const attr = frame.attr
  if (attr.phase._tag === 'PendingSlash') {
    if (ch === '>') {
      attr.phase = { _tag: 'Idle' }
      state.pop()
      events.push(...resolveSelfClose(state, frame.tagName, frame.toolCallId, attr.attrs, frame.raw, config))
    } else {
      attr.key += '/'
      attr.phase = { _tag: 'Idle' }
      return stepTagAttrs({ frame, state, ch, config })
    }
    return events.length > 0 ? emit(...events) : NOOP
  }
  if (attr.phase._tag === 'PendingEquals') {
    const key = attr.phase.key
    attr.phase = { _tag: 'Idle' }
    if (ch === '"') {
      attr.key = key
      attr.value = ''
      state[state.length - 1] = { _tag: 'TagAttrValue', tagName: frame.tagName, toolCallId: frame.toolCallId, attr, raw: frame.raw }
    } else if (ch === '>' || ch === '/') {
      events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr, key, ''))
      attr.key = ''
      return stepTagAttrs({ frame, state, ch, config })
    } else if (/\s/.test(ch)) {
      events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr, key, ''))
      attr.key = ''
    } else {
      attr.key = key
      attr.value = ch
      state[state.length - 1] = { _tag: 'TagUnquotedAttrValue', tagName: frame.tagName, toolCallId: frame.toolCallId, attr, raw: frame.raw }
    }
    return events.length > 0 ? emit(...events) : NOOP
  }
  if (ch === '>') {
    if (attr.key) {
      events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr, attr.key, ''))
      attr.key = ''
    }
    state.pop()
    events.push(...resolveOpenTag(state, frame.tagName, frame.toolCallId, attr.attrs, frame.raw, config))
  } else if (ch === '/') {
    if (attr.key) {
      events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr, attr.key, ''))
      attr.key = ''
    }
    attr.phase = { _tag: 'PendingSlash' }
  } else if (ch === '=') {
    attr.phase = { _tag: 'PendingEquals', key: attr.key }
    attr.key = ''
  } else if (/\s/.test(ch)) {
    if (attr.key) {
      events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr, attr.key, ''))
      attr.key = ''
    }
  } else attr.key += ch
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepTagAttrValue({ frame, state, ch, config }: { frame: TagAttrValueFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.raw += ch
  if (ch === '"') {
    events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, frame.attr, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    frame.attr.phase = { _tag: 'Idle' }
    state[state.length - 1] = { _tag: 'TagAttrs', tagName: frame.tagName, toolCallId: frame.toolCallId, attr: frame.attr, raw: frame.raw }
  } else frame.attr.value += ch
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepTagUnquotedAttrValue({ frame, state, ch, config }: { frame: TagUnquotedAttrValueFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.raw += ch
  if (/\s/.test(ch)) {
    events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, frame.attr, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    frame.attr.phase = { _tag: 'Idle' }
    state[state.length - 1] = { _tag: 'TagAttrs', tagName: frame.tagName, toolCallId: frame.toolCallId, attr: frame.attr, raw: frame.raw }
  } else if (ch === '>') {
    events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, frame.attr, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    state.pop()
    events.push(...resolveOpenTag(state, frame.tagName, frame.toolCallId, frame.attr.attrs, frame.raw, config))
  } else if (ch === '/') {
    events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, frame.attr, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    frame.attr.phase = { _tag: 'PendingSlash' }
    state[state.length - 1] = { _tag: 'TagAttrs', tagName: frame.tagName, toolCallId: frame.toolCallId, attr: frame.attr, raw: frame.raw }
  } else frame.attr.value += ch
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepPendingStructuralOpen({ frame, state, ch, config }: { frame: PendingStructuralOpenFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  if (ch === '\n') {
    state.pop()
    events.push(...resolveOpenTag(state, frame.tagName, '', new Map(), frame.raw, config))
    const top = state[state.length - 1]
    if (top?._tag === 'Think') {
      top.think.body += '\n'
      if (top.think.tagName !== config.keywords.lenses) events.push({ _tag: 'ProseChunk', patternId: 'think', text: '\n' })
      top.think.lastCharNewline = true
    } else {
      const prose = state[0]
      prose.lastCharNewline = true
    }
    return events.length > 0 ? emit(...events) : NOOP
  }
  state.pop()
  events.push(...emitProseChunk(state, frame.raw))
  return events.length > 0 ? emitAndReprocess(...events) : reprocess()
}

export function stepPendingTopLevelClose({ frame, state, ch, config }: { frame: PendingTopLevelCloseFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const proseRoot = state[0]
  if (ch === '\n') {
    const close = (name: string, kind: 'Actions' | 'Inspect' | 'Comms', ev: 'ActionsClose' | 'InspectClose' | 'CommsClose') => {
      if (frame.tagName !== name) return false
      state.pop()
      if (state[state.length - 1]?._tag === 'Prose') state.pop()
      for (let i = state.length - 1; i >= 0; i--) {
        if (state[i]?._tag === kind) { state.splice(i, 1); break }
      }
      proseRoot.justClosedStructural = true
      if (containerDepth(state, kind) === 0) events.push({ _tag: ev })
      proseRoot.lastCharNewline = true
      const prose = mkRootProse()
      prose.justClosedStructural = true
      prose.lastCharNewline = true
      state.push(prose)
      return true
    }
    if (close(config.keywords.actions, 'Actions', 'ActionsClose')) return events.length > 0 ? emit(...events) : NOOP
    if (close('inspect', 'Inspect', 'InspectClose')) return events.length > 0 ? emit(...events) : NOOP
    close(config.keywords.comms, 'Comms', 'CommsClose')
    return events.length > 0 ? emit(...events) : NOOP
  }
  state.pop()
  events.push(...emitProseChunk(state, frame.closeRaw))
  return events.length > 0 ? emitAndReprocess(...events) : reprocess()
}