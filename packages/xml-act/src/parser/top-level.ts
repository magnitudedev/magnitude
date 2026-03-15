import type {
  AttributeValue,
  ClosePrefixMatchFrame,
  OpenPrefixMatchFrame,
  ParseEvent,
  ParseStack,
  ParsedElement,
  PendingStructuralOpenFrame,
  PendingTopLevelCloseFrame,
  ParserConfig,
  StepResult,
  TagAttrsFrame,
  TagAttrValueFrame,
  TagUnquotedAttrValueFrame,
} from './types'
import { emit, emitAndReprocess, mkAttrState, mkRootProse, NOOP, reprocess } from './types'
import { validateToolAttr } from './validate-attrs'
import { activeCloseCandidates, activeTags, containerDepth } from './stack-ops'
import { emitProseChunk, endProseBlock } from './prose'
import { advancePrefixMatch } from './prefix-match'

export function parseToolRef(raw: string): { tag: string; recency: number } | undefined {
  const match = /^([a-zA-Z0-9_-]+)(?:~(\d+))?$/.exec(raw)
  if (!match) return undefined
  return { tag: match[1], recency: match[2] ? Number(match[2]) : 0 }
}

export function finalizeToolAttr(config: ParserConfig, tagName: string, toolCallId: string, attrs: Map<string, AttributeValue>, key: string, raw: string): ParseEvent[] {
  const schema = config.tagSchemas?.get(tagName)
  if (!schema) {
    attrs.set(key, raw)
    return []
  }
  const result = validateToolAttr(tagName, schema, key, raw)
  if (result.ok) {
    attrs.set(key, result.value)
    return []
  }
  return [{ _tag: 'ParseError', error: { ...result.error, toolCallId, tagName } }]
}

export function resolveOpenTag(state: ParseStack, tagName: string, toolCallId: string, attrs: Map<string, AttributeValue>, raw: string, config: ParserConfig): ParseEvent[] {
  const events: ParseEvent[] = []
  const kw = config.keywords
  const proseRoot = state[0]
  if (tagName === kw.actions) {
    events.push(...endProseBlock(state))
    const outermost = containerDepth(state, 'Actions') === 0
    state.push({ _tag: 'Actions' })
    const actionsProse = mkRootProse()
    actionsProse.lastCharNewline = false
    state.push(actionsProse)
    proseRoot.lastCharNewline = false
    if (outermost) events.push({ _tag: 'ActionsOpen' })
    return events
  }
  if (tagName === 'inspect') {
    events.push(...endProseBlock(state))
    const outermost = containerDepth(state, 'Inspect') === 0
    state.push({ _tag: 'Inspect' })
    const inspectProse = mkRootProse()
    inspectProse.lastCharNewline = false
    state.push(inspectProse)
    proseRoot.lastCharNewline = false
    if (outermost) events.push({ _tag: 'InspectOpen' })
    return events
  }
  if (tagName === kw.comms) {
    events.push(...endProseBlock(state))
    const outermost = containerDepth(state, 'Comms') === 0
    state.push({ _tag: 'Comms' })
    const commsProse = mkRootProse()
    commsProse.lastCharNewline = false
    state.push(commsProse)
    proseRoot.lastCharNewline = false
    if (outermost) events.push({ _tag: 'CommsOpen' })
    return events
  }
  if (tagName === kw.think || tagName === kw.thinking || tagName === kw.lenses) {
    events.push(...endProseBlock(state))
    proseRoot.lastCharNewline = false
    const aboutValue = attrs.get('about')
    const about = tagName === kw.lenses ? null : typeof aboutValue === 'string' ? aboutValue : null
    state.push({ _tag: 'Think', think: { tagName, body: '', depth: 0, openPrefix: null, openAfterNewline: false, lastCharNewline: false, about, lenses: [], activeLens: null }, pendingLt: false })
    return events
  }
  if (containerDepth(state, 'Inspect') === 0 && tagName === 'message') {
    events.push(...endProseBlock(state))
    const dest = typeof attrs.get('to') === 'string' ? String(attrs.get('to')) : config.defaultMessageDest
    const artifactsRaw = typeof attrs.get('artifacts') === 'string' ? String(attrs.get('artifacts')) : null
    const id = config.generateId()
    events.push({ _tag: 'MessageTagOpen', id, dest, artifactsRaw })
    state.push({ _tag: 'MessageBody', id, dest, artifactsRaw, body: '', pendingLt: false, depth: 0, pendingNewline: false })
    return events
  }
  if (config.knownTags.has(tagName)) {
    events.push(...endProseBlock(state))
    proseRoot.lastCharNewline = false
    events.push({ _tag: 'TagOpened', tagName, toolCallId, attributes: new Map(attrs) })
    state.push({ _tag: 'ToolBody', tagName, toolCallId, attrs, body: '', children: [], childCounts: new Map(), pendingLt: false })
    return events
  }
  return events
}

export function resolveSelfClose(state: ParseStack, tagName: string, toolCallId: string, attrs: Map<string, AttributeValue>, raw: string, config: ParserConfig): ParseEvent[] {
  const events: ParseEvent[] = []
  const kw = config.keywords
  if (tagName === kw.actions || tagName === 'inspect' || tagName === kw.comms) return events
  if (containerDepth(state, 'Actions') === 0 && containerDepth(state, 'Inspect') === 0 && containerDepth(state, 'Comms') === 0 && (tagName === kw.next || tagName === kw.yield)) {
    events.push({ _tag: 'TurnControl', decision: tagName === kw.next ? 'continue' : 'yield' })
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

export function stepOpenPrefixMatch({ frame, state, ch, config }: { frame: OpenPrefixMatchFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []

  // First char dispatch
  if (frame.prefix.matched === '' && ch === '!') {
    state.pop()
    state.push({ _tag: 'Cdata', cdata: { _tag: 'Prefix', index: 1, buffer: '<!' }, origin: state[0] })
    return NOOP
  }

  if (frame.prefix.matched === '' && ch === '/') {
    const closeCandidates = activeCloseCandidates(state, frame.afterNewline, config)
    if (closeCandidates.length === 0) {
      state.pop()
      events.push(...emitProseChunk(state, '</'))
      return emit(...events)
    }
    state[state.length - 1] = {
      _tag: 'ClosePrefixMatch',
      prefix: { candidates: closeCandidates, matched: '', raw: '</' },
      afterNewline: frame.afterNewline,
    }
    return NOOP
  }

  const advanced = advancePrefixMatch(frame.prefix, ch)

  if (advanced._tag === 'Continue') {
    frame.prefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    return NOOP
  }

  if (advanced._tag === 'NoMatch') {
    state.pop()
    events.push(...emitProseChunk(state, advanced.literal))
    return emit(...events)
  }

  // Matched a tag name
  const tags = activeTags(state, frame.afterNewline, config)

  if (advanced.delimiter === '>') {
    if (tags.has(advanced.tagName)) {
      state.pop()
      events.push(...resolveOpenTag(state, advanced.tagName, '', new Map(), advanced.raw, config))
    } else if (!frame.afterNewline && (advanced.tagName === config.keywords.think || advanced.tagName === config.keywords.thinking || advanced.tagName === config.keywords.lenses)) {
      state[state.length - 1] = { _tag: 'PendingStructuralOpen', tagName: advanced.tagName, raw: advanced.raw }
    } else {
      state.pop()
      events.push(...emitProseChunk(state, advanced.raw))
    }
    return emit(...events)
  }

  // Delimiter is whitespace or /
  if (!tags.has(advanced.tagName)) {
    state.pop()
    events.push(...emitProseChunk(state, advanced.raw))
    return emit(...events)
  }

  const toolCallId = config.knownTags.has(advanced.tagName) ? config.generateId() : ''
  const attr = mkAttrState()
  if (advanced.delimiter === '/') attr.phase = { _tag: 'PendingSlash' }
  state[state.length - 1] = { _tag: 'TagAttrs', tagName: advanced.tagName, toolCallId, attr, raw: advanced.raw }
  return NOOP
}

export function stepClosePrefixMatch({ frame, state, ch, config }: { frame: ClosePrefixMatchFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  const advanced = advancePrefixMatch(frame.prefix, ch)

  if (advanced._tag === 'Continue') {
    frame.prefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    return NOOP
  }

  if (advanced._tag === 'NoMatch') {
    state.pop()
    events.push(...emitProseChunk(state, advanced.literal))
    return emit(...events)
  }

  // Only accept > for structural close tags
  if (advanced.delimiter !== '>') {
    state.pop()
    events.push(...emitProseChunk(state, advanced.raw))
    return emit(...events)
  }

  const proseRoot = state[0]
  const kw = config.keywords
  const closeIf = (name: string, kind: 'Actions' | 'Inspect' | 'Comms', ev: 'ActionsClose' | 'InspectClose' | 'CommsClose') => {
    if (frame.afterNewline && advanced.tagName === name && containerDepth(state, kind) > 0) {
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
    if (!frame.afterNewline && advanced.tagName === name && containerDepth(state, kind) > 0) {
      state[state.length - 1] = { _tag: 'PendingTopLevelClose', tagName: name, closeRaw: advanced.raw }
      return true
    }
    return false
  }

  if (closeIf(kw.actions, 'Actions', 'ActionsClose')) return events.length > 0 ? emit(...events) : NOOP
  if (closeIf('inspect', 'Inspect', 'InspectClose')) return events.length > 0 ? emit(...events) : NOOP
  if (closeIf(kw.comms, 'Comms', 'CommsClose')) return events.length > 0 ? emit(...events) : NOOP

  state.pop()
  events.push(...emitProseChunk(state, advanced.raw))
  return emit(...events)
}

export function stepTagAttrs({ frame, state, ch, config }: { frame: TagAttrsFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.raw += ch
  const attr = frame.attr
  if (attr.phase._tag === 'PendingSlash') {
    if (ch === '>') {
      attr.phase = { _tag: 'Idle' }
      if (attr.key) {
        events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr.attrs, attr.key, ''))
        attr.key = ''
      }
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
      attr.value = ''
      attr.key = key
      state[state.length - 1] = { _tag: 'TagAttrValue', tagName: frame.tagName, toolCallId: frame.toolCallId, attr, raw: frame.raw }
    } else if (ch === '>' || ch === '/') {
      events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr.attrs, key, ''))
      attr.key = ''
      return stepTagAttrs({ frame, state, ch, config })
    } else if (/\s/.test(ch)) {
      events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr.attrs, key, ''))
      attr.key = ''
    } else {
      attr.value = ch
      attr.key = key
      state[state.length - 1] = { _tag: 'TagUnquotedAttrValue', tagName: frame.tagName, toolCallId: frame.toolCallId, attr, raw: frame.raw }
    }
    return events.length > 0 ? emit(...events) : NOOP
  }
  if (ch === '>') {
    if (attr.key) {
      events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr.attrs, attr.key, ''))
      attr.key = ''
    }
    state.pop()
    events.push(...resolveOpenTag(state, frame.tagName, frame.toolCallId, attr.attrs, frame.raw, config))
  } else if (ch === '/') {
    if (attr.key) {
      events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr.attrs, attr.key, ''))
      attr.key = ''
    }
    attr.phase = { _tag: 'PendingSlash' }
  } else if (ch === '=') {
    attr.phase = { _tag: 'PendingEquals', key: attr.key }
    attr.key = ''
  } else if (ch === '"') {
    attr.value = ''
    state[state.length - 1] = { _tag: 'TagAttrValue', tagName: frame.tagName, toolCallId: frame.toolCallId, attr, raw: frame.raw }
  } else if (/\s/.test(ch)) {
    if (attr.key) {
      events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, attr.attrs, attr.key, ''))
      attr.key = ''
    }
  } else attr.key += ch
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepTagAttrValue({ frame, state, ch, config }: { frame: TagAttrValueFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const events: ParseEvent[] = []
  frame.raw += ch
  if (ch === '"') {
    events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, frame.attr.attrs, frame.attr.key, frame.attr.value))
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
    events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, frame.attr.attrs, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    frame.attr.phase = { _tag: 'Idle' }
    state[state.length - 1] = { _tag: 'TagAttrs', tagName: frame.tagName, toolCallId: frame.toolCallId, attr: frame.attr, raw: frame.raw }
  } else if (ch === '>') {
    events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, frame.attr.attrs, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    state.pop()
    events.push(...resolveOpenTag(state, frame.tagName, frame.toolCallId, frame.attr.attrs, frame.raw, config))
  } else if (ch === '/') {
    events.push(...finalizeToolAttr(config, frame.tagName, frame.toolCallId, frame.attr.attrs, frame.attr.key, frame.attr.value))
    frame.attr.key = ''
    frame.attr.value = ''
    frame.attr.phase = { _tag: 'PendingSlash' }
    state[state.length - 1] = { _tag: 'TagAttrs', tagName: frame.tagName, toolCallId: frame.toolCallId, attr: frame.attr, raw: frame.raw }
  } else frame.attr.value += ch
  return events.length > 0 ? emit(...events) : NOOP
}

export function stepPendingStructuralOpen({ frame, state, ch, config }: { frame: PendingStructuralOpenFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  if (ch === '\n') {
    state.pop()
    return emit(...resolveOpenTag(state, frame.tagName, '', new Map(), frame.raw, config))
  }
  state.pop()
  const events = emitProseChunk(state, frame.raw)
  return emitAndReprocess(...events)
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
  return emitAndReprocess(...events)
}