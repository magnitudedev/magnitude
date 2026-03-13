/**
 * Streaming XML Parser
 *
 * Character-by-character state machine that parses XML from LLM output.
 * Chunk boundaries are transparent — the parser maintains state across chunks.
 *
 * Uses a strongly-typed discriminated union (ParserState) where each state
 * variant carries only its own data. Impossible states are unrepresentable.
 *
 * Emits ParseEvent[] for each chunk processed.
 * The runtime maps these internal events to tool-aware XmlRuntimeEvents.
 *
 * Delegates attribute validation/coercion to parser/validate-attrs.ts.
 */

import type { ParseEvent, ParsedChild, ParsedElement, AttributeValue } from './types'
import type { TagSchema } from '../execution/binding-validator'
import { validateToolAttr, validateChildAttr } from './validate-attrs'
import type {
  ParserState, StructuralCtx, FenceState, AttrState, ThinkState,
  ParentCtx, CloseTagBuf, CdataOrigin,
} from './state'
import { FencePhase, mkFence, mkProse, mkStructural, mkAttrState, mkCloseTag } from './state'
import { createId, createShortId } from '../util'
import { getKeywords, TURN_CONTROL_NEXT, TURN_CONTROL_YIELD } from '../constants'

function getStructuralTags(): Set<string> {
  const kw = getKeywords()
  return new Set([kw.actions, kw.think, kw.thinking, kw.lenses, 'inspect', kw.comms, TURN_CONTROL_NEXT, TURN_CONTROL_YIELD])
}
const REF_TAG_SET = new Set(['ref'])
const MESSAGE_TAG_SET = new Set(['message'])
const EMPTY_TAG_SET: ReadonlySet<string> = new Set()
const MESSAGE_TAG = 'message'
const createMessageId = createShortId

/** ID generator function type — injectable for replay determinism. */
export type IdGenerator = () => string

/** Default ID generator using cuid2. */
export const defaultIdGenerator: IdGenerator = createId

export interface StreamingXmlParser {
  processChunk(chunk: string): ParseEvent[]
  /** Flush remaining state and emit final events. Call once when the stream ends. */
  flush(): ParseEvent[]
}

/** Callback to resolve a ref inline. Returns the resolved text, or undefined if unresolvable. */
export type RefResolver = (tag: string, recency: number, query?: string) => string | undefined

const CDATA_PREFIX = '![CDATA['

function parseToolRef(raw: string): { tag: string; recency: number } | undefined {
  const match = /^([a-zA-Z0-9_-]+)(?:~(\d+))?$/.exec(raw)
  if (!match) return undefined
  return {
    tag: match[1],
    recency: match[2] ? Number(match[2]) : 0,
  }
}

/**
 * Create a streaming XML parser.
 *
 * @param knownTags - Set of registered top-level tool tag names
 * @param childTagMap - Map of tool tag name → set of valid child tag names for that tool
 * @param tagSchemas - Map of tool tag name → schema for inline validation/coercion
 * @param resolveRef - Optional callback to resolve <ref tool="tag~N" query="..."/> inline
 * @param generateId - Optional ID generator for toolCallIds. Defaults to cuid2.
 *                     Injectable for replay determinism: on replay, provide a generator
 *                     that returns the prior run's IDs in order.
 */
export function createStreamingXmlParser(
  knownTags: ReadonlySet<string>,
  childTagMap: ReadonlyMap<string, ReadonlySet<string>>,
  tagSchemas?: ReadonlyMap<string, TagSchema>,
  resolveRef?: RefResolver,
  generateId: IdGenerator = defaultIdGenerator,
  defaultMessageDest: string = 'user',
): StreamingXmlParser {
  // The two mutable variables — everything else is inside the state discriminant
  let state: ParserState = mkProse()
  let ctx: StructuralCtx = mkStructural()
  let emittedTurnControl: 'continue' | 'yield' | null = null

  // Event accumulator for current processChunk/flush call
  let events: ParseEvent[] = []

  function emit(event: ParseEvent): void {
    events.push(event)
  }

  // ===========================================================================
  // Fence / Prose helpers
  // ===========================================================================

  function isFenceComplete(phase: FencePhase): boolean {
    return phase === FencePhase.Tick3 || phase === FencePhase.XML || phase === FencePhase.TrailingWs
  }

  function structuralDepth(tagName: string): number {
    return ctx.structuralDepths.get(tagName) ?? 0
  }

  function isInStructural(tagName: string): boolean {
    return structuralDepth(tagName) > 0
  }

  function enterStructural(tagName: string): void {
    ctx.structuralDepths.set(tagName, structuralDepth(tagName) + 1)
  }

  function exitStructural(tagName: string): void {
    const depth = structuralDepth(tagName)
    if (depth <= 1) ctx.structuralDepths.delete(tagName)
    else ctx.structuralDepths.set(tagName, depth - 1)
  }

  /** Emit prose chunk, updating proseAccum */
  function rawEmitProse(fence: FenceState, proseAccum: string, text: string): string {
    if (text.length === 0) return proseAccum
    const acc = flushDeferredFence(fence, proseAccum)
    emit({ _tag: 'ProseChunk', patternId: 'prose', text })
    return acc + text
  }

  function flushDeferredFence(fence: FenceState, proseAccum: string): string {
    if (fence.deferred.length > 0) {
      const text = fence.deferred
      fence.deferred = ''
      emit({ _tag: 'ProseChunk', patternId: 'prose', text })
      return proseAccum + text
    }
    return proseAccum
  }

  function flushPendingWhitespace(fence: FenceState, proseAccum: string): string {
    if (fence.pendingWhitespace.length > 0) {
      const ws = fence.pendingWhitespace
      fence.pendingWhitespace = ''
      return rawEmitProse(fence, proseAccum, ws)
    }
    return proseAccum
  }

  function flushFenceBuffer(fence: FenceState, proseAccum: string): string {
    if (fence.buffer.length > 0) {
      let acc = flushPendingWhitespace(fence, proseAccum)
      acc = rawEmitProse(fence, acc, fence.buffer)
      fence.buffer = ''
      return acc
    }
    return proseAccum
  }

  function resetFence(fence: FenceState): void {
    fence.phase = FencePhase.LeadingWs
    fence.buffer = ''
  }

  /** Emit prose chunk, flushing fence/whitespace first */
  function emitProseChunk(fence: FenceState, proseAccum: string, text: string): string {
    if (text.length === 0) return proseAccum
    let acc = flushFenceBuffer(fence, proseAccum)
    acc = flushPendingWhitespace(fence, acc)
    return rawEmitProse(fence, acc, text)
  }

  /** Flush buffered state and emit ProseEnd if prose was accumulated */
  function endProseBlock(fence: FenceState, proseAccum: string): void {
    if (fence.phase !== FencePhase.Broken && isFenceComplete(fence.phase)) {
      fence.buffer = ''
      fence.pendingWhitespace = ''
    } else {
      proseAccum = flushFenceBuffer(fence, proseAccum)
    }
    fence.deferred = ''
    fence.pendingWhitespace = ''
    if (proseAccum.length > 0) {
      const content = proseAccum.trim()
      if (content.length > 0) {
        emit({ _tag: 'ProseEnd', patternId: 'prose', content, about: null })
      }
    }
    resetFence(fence)
  }

  function breakFence(fence: FenceState, proseAccum: string, ch: string): string {
    ctx.justClosedStructural = false
    fence.buffer += ch
    fence.phase = FencePhase.Broken
    return flushFenceBuffer(fence, proseAccum)
  }

  function appendProseChar(s: Extract<ParserState, { _tag: 'Prose' }>, ch: string): void {
    if (isInStructural('inspect')) return
    const fence = s.fence

    if (ch === '\n') {
      if (fence.phase !== FencePhase.Broken && isFenceComplete(fence.phase)) {
        if (ctx.justClosedStructural) {
          fence.buffer = ''
          fence.pendingWhitespace = ''
          ctx.justClosedStructural = false
          resetFence(fence)
        } else {
          fence.deferred = fence.pendingWhitespace + fence.buffer + '\n'
          fence.pendingWhitespace = ''
          fence.buffer = ''
          resetFence(fence)
        }
      } else {
        s.proseAccum = flushFenceBuffer(fence, s.proseAccum)
        fence.pendingWhitespace += '\n'
        resetFence(fence)
      }
      return
    }

    if (fence.pendingWhitespace.length > 0 && (ch === ' ' || ch === '\t' || ch === '\r')) {
      fence.pendingWhitespace += ch
      return
    }

    if (fence.phase === FencePhase.Broken) {
      s.proseAccum = flushPendingWhitespace(fence, s.proseAccum)
      s.proseAccum = rawEmitProse(fence, s.proseAccum, ch)
      return
    }

    switch (fence.phase) {
      case FencePhase.LeadingWs:
        if (ch === ' ' || ch === '\t') { fence.buffer += ch }
        else if (ch === '`') { fence.buffer += ch; fence.phase = FencePhase.Tick1 }
        else { s.proseAccum = breakFence(fence, s.proseAccum, ch) }
        break
      case FencePhase.Tick1:
        if (ch === '`') { fence.buffer += ch; fence.phase = FencePhase.Tick2 }
        else { s.proseAccum = breakFence(fence, s.proseAccum, ch) }
        break
      case FencePhase.Tick2:
        if (ch === '`') { fence.buffer += ch; fence.phase = FencePhase.Tick3 }
        else { s.proseAccum = breakFence(fence, s.proseAccum, ch) }
        break
      case FencePhase.Tick3:
        if (ch === 'x' || ch === 'X') { fence.buffer += ch; fence.phase = FencePhase.X }
        else if (ch === ' ' || ch === '\t') { fence.buffer += ch; fence.phase = FencePhase.TrailingWs }
        else { s.proseAccum = breakFence(fence, s.proseAccum, ch) }
        break
      case FencePhase.X:
        if (ch === 'm' || ch === 'M') { fence.buffer += ch; fence.phase = FencePhase.XM }
        else { s.proseAccum = breakFence(fence, s.proseAccum, ch) }
        break
      case FencePhase.XM:
        if (ch === 'l' || ch === 'L') { fence.buffer += ch; fence.phase = FencePhase.XML }
        else { s.proseAccum = breakFence(fence, s.proseAccum, ch) }
        break
      case FencePhase.XML:
        if (ch === ' ' || ch === '\t') { fence.buffer += ch; fence.phase = FencePhase.TrailingWs }
        else { s.proseAccum = breakFence(fence, s.proseAccum, ch) }
        break
      case FencePhase.TrailingWs:
        if (ch === ' ' || ch === '\t') { fence.buffer += ch }
        else { s.proseAccum = breakFence(fence, s.proseAccum, ch) }
        break
    }
  }

  // ===========================================================================
  // Attribute validation helpers
  // ===========================================================================

  function finalizeToolAttr(tagName: string, toolCallId: string, attr: AttrState, key: string, raw: string): void {
    if (attr.hasError) return
    if (!knownTags.has(tagName)) {
      attr.attrs.set(key, raw)
      return
    }
    const schema = tagSchemas?.get(tagName)
    if (!schema) {
      attr.attrs.set(key, raw)
      return
    }
    const result = validateToolAttr(tagName, schema, key, raw)
    if (result.ok) {
      attr.attrs.set(key, result.value)
    } else {
      attr.hasError = true
      emit({ _tag: 'ParseError', error: { ...result.error, toolCallId, tagName } })
    }
  }

  function finalizeChildAttrVal(parent: ParentCtx, childTagName: string, attr: AttrState, key: string, raw: string): void {
    if (attr.hasError) return
    const childSchema = tagSchemas?.get(parent.tagName)?.children.get(childTagName)
    if (!childSchema) {
      attr.attrs.set(key, raw)
      return
    }
    const result = validateChildAttr(parent.tagName, childTagName, childSchema, key, raw)
    if (result.ok) {
      attr.attrs.set(key, result.value)
    } else {
      attr.hasError = true
      emit({ _tag: 'ParseError', error: { ...result.error, toolCallId: parent.toolCallId, tagName: parent.tagName } })
    }
  }

  // ===========================================================================
  // Tag helpers
  // ===========================================================================

  function isValidChildTag(parentTag: string, childTag: string): boolean {
    if (childTag === 'ref') return resolveRef !== undefined
    const validSet = childTagMap.get(parentTag)
    return validSet ? validSet.has(childTag) : false
  }

  function getChildIndex(parent: ParentCtx, childTag: string): number {
    return parent.childCounts.get(childTag) ?? 0
  }

  function incrementChildIndex(parent: ParentCtx, childTag: string): void {
    parent.childCounts.set(childTag, (parent.childCounts.get(childTag) ?? 0) + 1)
  }

  // ===========================================================================
  // Think helpers
  // ===========================================================================

  function emitThinkChar(think: ThinkState, ch: string): void {
    if (think.tagName === kw.lenses && think.activeLens) {
      think.activeLens.content += ch
      emit({ _tag: 'LensChunk', text: ch })
      think.lastCharNewline = ch === '\n'
      return
    }

    think.body += ch
    if (think.tagName !== kw.lenses) {
      emit({ _tag: 'ProseChunk', patternId: 'think', text: ch })
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
  }

  // ===========================================================================
  // Transition: resolveOpenTag
  // ===========================================================================

  /**
   * Called when a top-level open tag is complete (saw '>').
   * Dispatches based on tag identity: structural, think, known tool, or unknown.
   * fence/proseAccum are the prose context from the originating Prose state.
   */
  function resolveOpenTag(fence: FenceState, proseAccum: string, tagName: string, toolCallId: string, attrs: Map<string, AttributeValue>, raw: string): void {
    if (tagName === kw.actions) {
      endProseBlock(fence, proseAccum)
      const outermost = !isInStructural(kw.actions)
      enterStructural(kw.actions)
      ctx.lastCharNewline = false
      if (outermost) emit({ _tag: 'ActionsOpen' })
      state = mkProse()
      return
    }
    if (tagName === 'inspect') {
      endProseBlock(fence, proseAccum)
      const outermost = !isInStructural('inspect')
      enterStructural('inspect')
      ctx.lastCharNewline = false
      if (outermost) emit({ _tag: 'InspectOpen' })
      state = mkProse()
      return
    }
    if (tagName === kw.comms) {
      endProseBlock(fence, proseAccum)
      const outermost = !isInStructural(kw.comms)
      enterStructural(kw.comms)
      ctx.lastCharNewline = false
      if (outermost) emit({ _tag: 'CommsOpen' })
      state = mkProse()
      return
    }
    if (tagName === kw.think || tagName === kw.thinking || tagName === kw.lenses) {
      endProseBlock(fence, proseAccum)
      ctx.lastCharNewline = false
      const about = tagName === kw.lenses ? null : (attrs.get('about') as string | undefined) ?? null
      state = {
        _tag: 'Think',
        think: {
          tagName,
          body: '',
          depth: 0,
          openTagBuf: '',
          openAfterNewline: false,
          lastCharNewline: false,
          about,
          lenses: [],
          activeLens: null,
        },
        pendingLt: false,
      }
      return
    }
    if (!isInStructural('inspect') && tagName === MESSAGE_TAG) {
      endProseBlock(fence, proseAccum)
      const id = createMessageId()
      const dest = (attrs.get('to') as string | undefined) ?? defaultMessageDest
      const artifactsRaw = (attrs.get('artifacts') as string | undefined) ?? null
      emit({ _tag: 'MessageTagOpen', id, dest, artifactsRaw })
      state = {
        _tag: 'MessageBody',
        id,
        dest,
        artifactsRaw,
        body: '',
        pendingLt: false,
        depth: 0,
        pendingNewline: false,
      }
      return
    }
    if (!isInStructural('inspect') && !isInStructural(kw.comms) && knownTags.has(tagName)) {
      endProseBlock(fence, proseAccum)
      if (!toolCallId) toolCallId = generateId()
      emit({ _tag: 'TagOpened', tagName, toolCallId, attributes: new Map(attrs) })
      state = {
        _tag: 'ToolBody',
        tagName,
        toolCallId,
        attrs,
        body: '',
        pendingLt: false,
      }
      return
    }
    // Unknown tag — emit raw as prose, return to Prose with existing context
    const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence, proseAccum }
    prose.proseAccum = emitProseChunk(fence, proseAccum, raw)
    state = prose
  }

  /**
   * Called when a self-closing tag is complete (saw '/>').
   * fence/proseAccum are the prose context from the originating Prose state.
   */
  function resolveSelfClose(fence: FenceState, proseAccum: string, tagName: string, toolCallId: string, attrs: Map<string, AttributeValue>, raw: string): void {
    if (tagName === kw.actions || tagName === 'inspect' || tagName === kw.comms) {
      state = { _tag: 'Prose', fence, proseAccum }
      return
    }
    if (!isInStructural(kw.actions) && !isInStructural('inspect') && !isInStructural(kw.comms) && (tagName === TURN_CONTROL_NEXT || tagName === TURN_CONTROL_YIELD)) {
      if (emittedTurnControl === null) {
        const decision: 'continue' | 'yield' = tagName === TURN_CONTROL_NEXT ? 'continue' : 'yield'
        emittedTurnControl = decision
        emit({ _tag: 'TurnControl', decision })
      }
      state = { _tag: 'Done' }
      return
    }
    if (isInStructural('inspect') && tagName === 'ref') {
      const toolRef = attrs.get('tool')
      if (typeof toolRef === 'string') {
        const parsed = parseToolRef(toolRef)
        const query = attrs.get('query')
        if (parsed) {
          if (resolveRef) {
            const resolved = resolveRef(parsed.tag, parsed.recency, typeof query === 'string' ? query : undefined)
            if (resolved !== undefined) {
              emit({ _tag: 'InspectResult', toolRef, query: typeof query === 'string' ? query : undefined, content: resolved })
            } else {
              emit({ _tag: 'ParseError', error: { _tag: 'InvalidRef', toolRef, detail: `Ref "${toolRef}" does not match any tool result from this response` } })
            }
          } else if (parsed.tag !== 'fs-write') {
            emit({ _tag: 'ParseError', error: { _tag: 'InvalidRef', toolRef, detail: `Ref "${toolRef}" does not match any tool result from this response` } })
          }
        } else {
          emit({ _tag: 'ParseError', error: { _tag: 'InvalidRef', toolRef, detail: `Invalid tool ref "${toolRef}". Expected format "tag" or "tag~N"` } })
        }
      }
      state = { _tag: 'Prose', fence, proseAccum }
      return
    }
    if (!isInStructural('inspect') && tagName === MESSAGE_TAG) {
      const id = createMessageId()
      const dest = (attrs.get('to') as string | undefined) ?? defaultMessageDest
      const artifactsRaw = (attrs.get('artifacts') as string | undefined) ?? null
      emit({ _tag: 'MessageTagOpen', id, dest, artifactsRaw })
      emit({ _tag: 'MessageTagClose', id })
      state = mkProse()
    } else if (!isInStructural('inspect') && !isInStructural(kw.comms) && knownTags.has(tagName)) {
      endProseBlock(fence, proseAccum)
      if (!toolCallId) toolCallId = generateId()
      const attrsCopy = new Map(attrs)
      emit({ _tag: 'TagOpened', tagName, toolCallId, attributes: attrsCopy })
      const element: ParsedElement = {
        tagName, toolCallId, attributes: attrsCopy, body: '', children: [],
      }
      emit({ _tag: 'TagClosed', toolCallId, tagName, element })
      state = mkProse()
    } else {
      // Unknown self-closing tag — emit raw as prose
      const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence, proseAccum }
      prose.proseAccum = emitProseChunk(fence, proseAccum, raw)
      state = prose
    }
  }

  // ===========================================================================
  // processChar — main dispatch
  // ===========================================================================

  function processChar(ch: string): void {
    switch (state._tag) {
      case 'Prose': return stepProse(state, ch)
      case 'TagName': return stepTagName(state, ch)
      case 'TopLevelCloseTag': return stepTopLevelCloseTag(state, ch)
      case 'TagAttrs': return stepTagAttrs(state, ch)
      case 'TagAttrValue': return stepTagAttrValue(state, ch)
      case 'TagUnquotedAttrValue': return stepTagUnquotedAttrValue(state, ch)
      case 'Think': return stepThink(state, ch)
      case 'ThinkCloseTag': return stepThinkCloseTag(state, ch)
      case 'LensTagName': return stepLensTagName(state, ch)
      case 'LensTagAttrs': return stepLensTagAttrs(state, ch)
      case 'PendingThinkClose': return stepPendingThinkClose(state, ch)
      case 'PendingStructuralOpen': return stepPendingStructuralOpen(state, ch)
      case 'PendingTopLevelClose': return stepPendingTopLevelClose(state, ch)
      case 'MessageBody': return stepMessageBody(state, ch)
      case 'MessageBodyOpenTag': return stepMessageBodyOpenTag(state, ch)
      case 'MessageCloseTag': return stepMessageCloseTag(state, ch)
      case 'ToolBody': return stepToolBody(state, ch)
      case 'ToolCloseTag': return stepToolCloseTag(state, ch)
      case 'ParentBody': return stepParentBody(state, ch)
      case 'ParentCloseTag': return stepParentCloseTag(state, ch)
      case 'ChildTagName': return stepChildTagName(state, ch)
      case 'ChildAttrs': return stepChildAttrs(state, ch)
      case 'ChildAttrValue': return stepChildAttrValue(state, ch)
      case 'ChildUnquotedAttrValue': return stepChildUnquotedAttrValue(state, ch)
      case 'ChildBody': return stepChildBody(state, ch)
      case 'ChildCloseTag': return stepChildCloseTag(state, ch)
      case 'Cdata': return stepCdata(state, ch)
    }
  }

  // ===========================================================================
  // Step: Prose
  // ===========================================================================

  function stepProse(s: Extract<ParserState, { _tag: 'Prose' }>, ch: string): void {
    if (ch === '<') {
      s.proseAccum = flushFenceBuffer(s.fence, s.proseAccum)
      state = { _tag: 'TagName', name: '', raw: '<', fence: s.fence, proseAccum: s.proseAccum, afterNewline: ctx.lastCharNewline }
      ctx.lastCharNewline = false
    } else {
      ctx.lastCharNewline = ch === '\n'
      appendProseChar(s, ch)
    }
  }

  // ===========================================================================
  // Step: TagName
  // ===========================================================================

  // Capture structural tags at parser creation time
  const structuralTags = getStructuralTags()
  const kw = getKeywords()

  // knownTags + structural + message — used inside <actions> where all are valid
  const actionsTags: ReadonlySet<string> = new Set([...knownTags, ...structuralTags, MESSAGE_TAG])
  const topLevelTags: ReadonlySet<string> = new Set([...structuralTags, ...knownTags, MESSAGE_TAG])

  /** Return the set of tags recognized in the current context.
   *  When afterNewline is false, structural tags are excluded — they only match after \n or start of input. */
  function activeTags(afterNewline: boolean): ReadonlySet<string> {
    if (isInStructural('inspect')) return afterNewline ? new Set([...REF_TAG_SET, 'inspect']) : REF_TAG_SET
    if (isInStructural(kw.comms)) return afterNewline ? new Set([...MESSAGE_TAG_SET, kw.comms]) : MESSAGE_TAG_SET
    if (isInStructural(kw.actions)) return afterNewline ? actionsTags : knownTags
    return afterNewline ? topLevelTags : EMPTY_TAG_SET
  }

  /** Check if prefix is a valid prefix of any tag in the given set */
  function isPrefixOfAny(prefix: string, tags: ReadonlySet<string>): boolean {
    for (const tag of tags) {
      if (tag.startsWith(prefix)) return true
    }
    return false
  }

  function stepTagName(s: Extract<ParserState, { _tag: 'TagName' }>, ch: string): void {
    s.raw += ch
    if (s.name.length === 0 && ch === '/') {
      state = { _tag: 'TopLevelCloseTag', close: { name: '', raw: '</' }, fence: s.fence, proseAccum: s.proseAccum, afterNewline: s.afterNewline }
    } else if (s.name.length === 0 && ch === '!') {
      state = {
        _tag: 'Cdata',
        cdata: { _tag: 'Prefix', index: 1, buffer: '<!' },
        origin: { _tag: 'FromProse', fence: s.fence, proseAccum: s.proseAccum },
      }
    } else if (/[a-zA-Z0-9_-]/.test(ch)) {
      const candidate = s.name + ch
      if (isPrefixOfAny(candidate, activeTags(s.afterNewline)) || isPrefixOfAny(candidate, structuralTags)) {
        s.name = candidate
      } else {
        // Can't match any known tag — emit as prose
        const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: s.fence, proseAccum: s.proseAccum }
        prose.proseAccum = emitProseChunk(s.fence, s.proseAccum, '<' + candidate)
        state = prose
      }
    } else if (s.name.length === 0) {
      // '<' followed by non-alpha non-/ non-! — emit both as prose
      const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: s.fence, proseAccum: s.proseAccum }
      appendProseChar(prose, '<')
      appendProseChar(prose, ch)
      state = prose
    } else if (/\s/.test(ch)) {
      const tags = activeTags(s.afterNewline)
      if (tags.has(s.name)) {
        const toolCallId = knownTags.has(s.name) ? generateId() : ''
        state = {
          _tag: 'TagAttrs',
          tagName: s.name,
          toolCallId,
          attr: mkAttrState(),
          raw: s.raw,
          fence: s.fence,
          proseAccum: s.proseAccum,
        }
      } else {
        const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: s.fence, proseAccum: s.proseAccum }
        prose.proseAccum = emitProseChunk(s.fence, s.proseAccum, '<' + s.name + ch)
        state = prose
      }
    } else if (ch === '>') {
      const tags = activeTags(s.afterNewline)
      if (tags.has(s.name)) {
        resolveOpenTag(s.fence, s.proseAccum, s.name, '', new Map(), s.raw)
      } else if (!s.afterNewline && (s.name === kw.think || s.name === kw.thinking || s.name === kw.lenses)) {
        // Think tag inline — defer: valid if next char is \n
        state = { _tag: 'PendingStructuralOpen', tagName: s.name, fence: s.fence, proseAccum: s.proseAccum, raw: s.raw }
      } else {
        const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: s.fence, proseAccum: s.proseAccum }
        prose.proseAccum = emitProseChunk(s.fence, s.proseAccum, '<' + s.name + '>')
        state = prose
      }
    } else if (ch === '/') {
      const tags = activeTags(s.afterNewline)
      if (tags.has(s.name)) {
        const toolCallId = knownTags.has(s.name) ? generateId() : ''
        const attr = mkAttrState()
        attr.phase = { _tag: 'PendingSlash' }
        state = {
          _tag: 'TagAttrs',
          tagName: s.name,
          toolCallId,
          attr,
          raw: s.raw,
          fence: s.fence,
          proseAccum: s.proseAccum,
        }
      } else {
        const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: s.fence, proseAccum: s.proseAccum }
        prose.proseAccum = emitProseChunk(s.fence, s.proseAccum, '<' + s.name + ch)
        state = prose
      }
    } else {
      // Invalid char in tag name — emit as prose
      const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: s.fence, proseAccum: s.proseAccum }
      prose.proseAccum = emitProseChunk(s.fence, s.proseAccum, '<' + s.name + ch)
      state = prose
    }
  }

  // ===========================================================================
  // Step: TopLevelCloseTag
  // ===========================================================================

  function stepTopLevelCloseTag(s: Extract<ParserState, { _tag: 'TopLevelCloseTag' }>, ch: string): void {
    s.close.raw += ch
    if (ch === '>') {
      if (s.afterNewline && s.close.name === kw.actions && isInStructural(kw.actions)) {
        exitStructural(kw.actions)
        ctx.justClosedStructural = true
        if (!isInStructural(kw.actions)) emit({ _tag: 'ActionsClose' })
        state = mkProse()
      } else if (s.afterNewline && s.close.name === 'inspect' && isInStructural('inspect')) {
        exitStructural('inspect')
        ctx.justClosedStructural = true
        if (!isInStructural('inspect')) emit({ _tag: 'InspectClose' })
        state = mkProse()
      } else if (s.afterNewline && s.close.name === kw.comms && isInStructural(kw.comms)) {
        exitStructural(kw.comms)
        ctx.justClosedStructural = true
        if (!isInStructural(kw.comms)) emit({ _tag: 'CommsClose' })
        state = mkProse()
      } else if (!s.afterNewline && s.close.name === kw.actions && isInStructural(kw.actions)) {
        // Actions close inline — defer: valid if next char is \n
        state = { _tag: 'PendingTopLevelClose', tagName: kw.actions, fence: s.fence, proseAccum: s.proseAccum, closeRaw: s.close.raw }
      } else if (!s.afterNewline && s.close.name === 'inspect' && isInStructural('inspect')) {
        // Inspect close inline — defer: valid if next char is \n
        state = { _tag: 'PendingTopLevelClose', tagName: 'inspect', fence: s.fence, proseAccum: s.proseAccum, closeRaw: s.close.raw }
      } else if (!s.afterNewline && s.close.name === kw.comms && isInStructural(kw.comms)) {
        // Comms close inline — defer: valid if next char is \n
        state = { _tag: 'PendingTopLevelClose', tagName: kw.comms, fence: s.fence, proseAccum: s.proseAccum, closeRaw: s.close.raw }
      } else {
        // Unknown top-level close tag or not after newline — emit as prose
        const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: s.fence, proseAccum: s.proseAccum }
        prose.proseAccum = emitProseChunk(s.fence, s.proseAccum, s.close.raw)
        state = prose
      }
    } else {
      s.close.name += ch
    }
  }

  // ===========================================================================
  // Step: TagAttrs / TagAttrValue / TagUnquotedAttrValue
  // ===========================================================================

  function stepTagAttrs(s: Extract<ParserState, { _tag: 'TagAttrs' }>, ch: string): void {
    s.raw += ch
    const attr = s.attr

    if (attr.phase._tag === 'PendingSlash') {
      if (ch === '>') {
        attr.phase = { _tag: 'Idle' }
        resolveSelfClose(s.fence, s.proseAccum, s.tagName, s.toolCallId, attr.attrs, s.raw)
      } else {
        attr.key += '/'
        attr.phase = { _tag: 'Idle' }
        // Re-process the char
        stepTagAttrs(s, ch)
      }
      return
    }

    if (attr.phase._tag === 'PendingEquals') {
      const key = attr.phase.key
      attr.phase = { _tag: 'Idle' }
      if (ch === '"') {
        attr.key = key
        attr.value = ''
        state = { _tag: 'TagAttrValue', tagName: s.tagName, toolCallId: s.toolCallId, attr, raw: s.raw, fence: s.fence, proseAccum: s.proseAccum }
      } else if (ch === '>' || ch === '/') {
        finalizeToolAttr(s.tagName, s.toolCallId, attr, key, '')
        attr.key = ''
        stepTagAttrs(s, ch)
      } else if (/\s/.test(ch)) {
        finalizeToolAttr(s.tagName, s.toolCallId, attr, key, '')
        attr.key = ''
      } else {
        attr.key = key
        attr.value = ch
        state = { _tag: 'TagUnquotedAttrValue', tagName: s.tagName, toolCallId: s.toolCallId, attr, raw: s.raw, fence: s.fence, proseAccum: s.proseAccum }
      }
      return
    }

    if (ch === '>') {
      if (attr.key) { finalizeToolAttr(s.tagName, s.toolCallId, attr, attr.key, ''); attr.key = '' }
      resolveOpenTag(s.fence, s.proseAccum, s.tagName, s.toolCallId, attr.attrs, s.raw)
    } else if (ch === '/') {
      if (attr.key) { finalizeToolAttr(s.tagName, s.toolCallId, attr, attr.key, ''); attr.key = '' }
      attr.phase = { _tag: 'PendingSlash' }
    } else if (ch === '=') {
      attr.phase = { _tag: 'PendingEquals', key: attr.key }
      attr.key = ''
    } else if (/\s/.test(ch)) {
      if (attr.key) { finalizeToolAttr(s.tagName, s.toolCallId, attr, attr.key, ''); attr.key = '' }
    } else {
      attr.key += ch
    }
  }

  function stepTagAttrValue(s: Extract<ParserState, { _tag: 'TagAttrValue' }>, ch: string): void {
    s.raw += ch
    if (ch === '"') {
      finalizeToolAttr(s.tagName, s.toolCallId, s.attr, s.attr.key, s.attr.value)
      s.attr.key = ''
      s.attr.value = ''
      s.attr.phase = { _tag: 'Idle' }
      state = { _tag: 'TagAttrs', tagName: s.tagName, toolCallId: s.toolCallId, attr: s.attr, raw: s.raw, fence: s.fence, proseAccum: s.proseAccum }
    } else {
      s.attr.value += ch
    }
  }

  function stepTagUnquotedAttrValue(s: Extract<ParserState, { _tag: 'TagUnquotedAttrValue' }>, ch: string): void {
    s.raw += ch
    if (/\s/.test(ch)) {
      finalizeToolAttr(s.tagName, s.toolCallId, s.attr, s.attr.key, s.attr.value)
      s.attr.key = ''
      s.attr.value = ''
      s.attr.phase = { _tag: 'Idle' }
      state = { _tag: 'TagAttrs', tagName: s.tagName, toolCallId: s.toolCallId, attr: s.attr, raw: s.raw, fence: s.fence, proseAccum: s.proseAccum }
    } else if (ch === '>') {
      finalizeToolAttr(s.tagName, s.toolCallId, s.attr, s.attr.key, s.attr.value)
      s.attr.key = ''
      s.attr.value = ''
      resolveOpenTag(s.fence, s.proseAccum, s.tagName, s.toolCallId, s.attr.attrs, s.raw)
    } else if (ch === '/') {
      finalizeToolAttr(s.tagName, s.toolCallId, s.attr, s.attr.key, s.attr.value)
      s.attr.key = ''
      s.attr.value = ''
      s.attr.phase = { _tag: 'PendingSlash' }
      state = { _tag: 'TagAttrs', tagName: s.tagName, toolCallId: s.toolCallId, attr: s.attr, raw: s.raw, fence: s.fence, proseAccum: s.proseAccum }
    } else {
      s.attr.value += ch
    }
  }

  // ===========================================================================
  // Step: Think / ThinkCloseTag
  // ===========================================================================

  function stepThink(s: Extract<ParserState, { _tag: 'Think' }>, ch: string): void {
    if (s.pendingLt) {
      s.pendingLt = false
      const wasAfterNewline = s.think.lastCharNewline
      if (ch === '/') {
        state = {
          _tag: 'ThinkCloseTag',
          think: s.think,
          close: { name: '', raw: '</' },
          afterNewline: wasAfterNewline,
        }
      } else if (ch === '!') {
        // CDATA inside think — treat as think body text (think doesn't support CDATA)
        emitThinkChar(s.think, '<')
        emitThinkChar(s.think, '!')
      } else if (/[a-zA-Z0-9_-]/.test(ch)) {
        if (s.think.tagName === kw.lenses) {
          state = { _tag: 'LensTagName', think: s.think, name: ch }
        } else {
          emitThinkChar(s.think, '<')
          emitThinkChar(s.think, ch)
          s.think.openTagBuf = ch
          s.think.openAfterNewline = wasAfterNewline
        }
      } else {
        emitThinkChar(s.think, '<')
        s.think.openTagBuf = ''
        emitThinkChar(s.think, ch)
      }
    } else if (ch === '<') {
      s.pendingLt = true
    } else {
      emitThinkChar(s.think, ch)
    }
  }

  function stepThinkCloseTag(s: Extract<ParserState, { _tag: 'ThinkCloseTag' }>, ch: string): void {
    s.close.raw += ch
    if (ch === '>') {
      if (s.think.tagName === kw.lenses && s.close.name === 'lens') {
        if (s.think.activeLens) {
          if (s.think.activeLens.depth > 0) {
            s.think.activeLens.depth--
            for (const c of s.close.raw) emitThinkChar(s.think, c)
          } else {
            const content = s.think.activeLens.content.trim()
            emit({ _tag: 'LensEnd', name: s.think.activeLens.name, content })
            s.think.lenses.push({ name: s.think.activeLens.name, content })
            s.think.activeLens = null
          }
        } else {
          for (const c of s.close.raw) emitThinkChar(s.think, c)
        }
        state = { _tag: 'Think', think: s.think, pendingLt: false }
      } else if (s.close.name === s.think.tagName && s.afterNewline) {
        if (s.think.depth > 0) {
          s.think.depth--
          for (const c of s.close.raw) emitThinkChar(s.think, c)
          state = { _tag: 'Think', think: s.think, pendingLt: false }
        } else {
          if (s.think.tagName !== kw.lenses) {
            emit({ _tag: 'ProseEnd', patternId: 'think', content: s.think.body, about: s.think.about })
          }
          state = mkProse()
        }
      } else if (s.close.name === s.think.tagName && !s.afterNewline) {
        state = { _tag: 'PendingThinkClose', think: s.think, closeRaw: s.close.raw }
      } else {
        for (const c of s.close.raw) emitThinkChar(s.think, c)
        state = { _tag: 'Think', think: s.think, pendingLt: false }
      }
    } else {
      s.close.name += ch
    }
  }

  function stepPendingThinkClose(s: Extract<ParserState, { _tag: 'PendingThinkClose' }>, ch: string): void {
    if (ch === '\n') {
      if (s.think.depth > 0) {
        s.think.depth--
        for (const c of s.closeRaw) emitThinkChar(s.think, c)
        emitThinkChar(s.think, '\n')
        state = { _tag: 'Think', think: s.think, pendingLt: false }
      } else {
        if (s.think.tagName !== kw.lenses) {
          emit({ _tag: 'ProseEnd', patternId: 'think', content: s.think.body, about: s.think.about })
        }
        const prose = mkProse()
        ctx.lastCharNewline = true
        state = prose
      }
    } else {
      for (const c of s.closeRaw) emitThinkChar(s.think, c)
      state = { _tag: 'Think', think: s.think, pendingLt: false }
      processChar(ch)
    }
  }

  function stepLensTagName(s: Extract<ParserState, { _tag: 'LensTagName' }>, ch: string): void {
    if (/[a-zA-Z]/.test(ch)) {
      const next = s.name + ch
      if ('lens'.startsWith(next)) {
        s.name = next
      } else {
        for (const c of '<' + next) emitThinkChar(s.think, c)
        state = { _tag: 'Think', think: s.think, pendingLt: false }
      }
      return
    }

    if (s.name !== 'lens') {
      for (const c of '<' + s.name + ch) emitThinkChar(s.think, c)
      state = { _tag: 'Think', think: s.think, pendingLt: false }
      return
    }

    if (/\s/.test(ch)) {
      state = {
        _tag: 'LensTagAttrs',
        think: s.think,
        attrKey: '',
        attrValue: '',
        phase: 'key',
        nameAttr: null,
        pendingSlash: false,
      }
      return
    }

    if (ch === '/') {
      state = {
        _tag: 'LensTagAttrs',
        think: s.think,
        attrKey: '',
        attrValue: '',
        phase: 'key',
        nameAttr: null,
        pendingSlash: true,
      }
      return
    }

    if (ch === '>') {
      s.think.activeLens = { name: '', content: '', depth: 0 }
      state = { _tag: 'Think', think: s.think, pendingLt: false }
      return
    }

    for (const c of '<' + s.name + ch) emitThinkChar(s.think, c)
    state = { _tag: 'Think', think: s.think, pendingLt: false }
  }

  function stepLensTagAttrs(s: Extract<ParserState, { _tag: 'LensTagAttrs' }>, ch: string): void {
    if (s.pendingSlash) {
      if (/\s/.test(ch)) return
      if (ch === '>') {
        const name = s.nameAttr ?? ''
        if (s.think.activeLens) {
          for (const c of `<lens${name ? ` name="${name}"` : ''} />`) emitThinkChar(s.think, c)
        } else {
          emit({ _tag: 'LensStart', name })
          emit({ _tag: 'LensEnd', name, content: '' })
          s.think.lenses.push({ name, content: null })
        }
        state = { _tag: 'Think', think: s.think, pendingLt: false }
        return
      }
      s.pendingSlash = false
    }

    if (s.phase === 'equals') {
      if (/\s/.test(ch)) return
      if (ch === '"') {
        s.attrValue = ''
        s.phase = 'value'
        return
      }
      for (const c of `<lens ${s.attrKey}=${ch}`) emitThinkChar(s.think, c)
      state = { _tag: 'Think', think: s.think, pendingLt: false }
      return
    }

    if (s.phase === 'value') {
      if (ch === '"') {
        if (s.attrKey === 'name') s.nameAttr = s.attrValue
        s.attrKey = ''
        s.attrValue = ''
        s.phase = 'key'
      } else {
        s.attrValue += ch
      }
      return
    }

    if (ch === '>') {
      const name = s.nameAttr ?? ''
      if (s.think.activeLens) {
        s.think.activeLens.depth++
        for (const c of `<lens${name ? ` name="${name}"` : ''}>`) emitThinkChar(s.think, c)
      } else {
        s.think.activeLens = { name, content: '', depth: 0 }
        emit({ _tag: 'LensStart', name })
      }
      state = { _tag: 'Think', think: s.think, pendingLt: false }
      return
    }

    if (ch === '/') {
      s.pendingSlash = true
      return
    }

    if (/\s/.test(ch)) {
      if (s.attrKey.length > 0) s.attrKey = ''
      return
    }

    if (ch === '=') {
      s.phase = 'equals'
      return
    }

    s.attrKey += ch
  }

  function stepPendingStructuralOpen(s: Extract<ParserState, { _tag: 'PendingStructuralOpen' }>, ch: string): void {
    if (ch === '\n') {
      // Confirmed: open the structural tag, \n is first body char (for think) or prose
      resolveOpenTag(s.fence, s.proseAccum, s.tagName, '', new Map(), s.raw)
      // After resolveOpenTag, state is now Think or Prose (for actions/inspect)
      // For think, emit \n as first body char
      if (state._tag === 'Think') {
        emitThinkChar(state.think, '\n')
      } else {
        // For actions/inspect, \n is prose — set lastCharNewline
        ctx.lastCharNewline = true
      }
    } else {
      // Not followed by newline — emit raw as prose
      const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: s.fence, proseAccum: s.proseAccum }
      prose.proseAccum = emitProseChunk(s.fence, s.proseAccum, s.raw)
      state = prose
      // Re-process ch
      processChar(ch)
    }
  }

  function stepPendingTopLevelClose(s: Extract<ParserState, { _tag: 'PendingTopLevelClose' }>, ch: string): void {
    if (ch === '\n') {
      if (s.tagName === kw.actions) {
        exitStructural(kw.actions)
        ctx.justClosedStructural = true
        if (!isInStructural(kw.actions)) emit({ _tag: 'ActionsClose' })
      } else if (s.tagName === 'inspect') {
        exitStructural('inspect')
        ctx.justClosedStructural = true
        if (!isInStructural('inspect')) emit({ _tag: 'InspectClose' })
      } else if (s.tagName === kw.comms) {
        exitStructural(kw.comms)
        ctx.justClosedStructural = true
        if (!isInStructural(kw.comms)) emit({ _tag: 'CommsClose' })
      }
      ctx.lastCharNewline = true
      state = mkProse()
    } else {
      // Not followed by newline — emit close raw as prose
      const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: s.fence, proseAccum: s.proseAccum }
      prose.proseAccum = emitProseChunk(s.fence, s.proseAccum, s.closeRaw)
      state = prose
      processChar(ch)
    }
  }

  // ===========================================================================
  // Step: MessageBody / MessageCloseTag
  // ===========================================================================

  function flushPendingMessageNewline(s: Extract<ParserState, { _tag: 'MessageBody' }>): void {
    if (!s.pendingNewline) return
    s.pendingNewline = false
    s.body += '\n'
    emit({ _tag: 'MessageBodyChunk', id: s.id, text: '\n' })
  }

  function stepMessageBody(s: Extract<ParserState, { _tag: 'MessageBody' }>, ch: string): void {
    if (s.pendingLt) {
      s.pendingLt = false
      if (ch === '/') {
        state = {
          _tag: 'MessageCloseTag',
          id: s.id,
          dest: s.dest,
          artifactsRaw: s.artifactsRaw,
          body: s.body,
          close: mkCloseTag(),
          depth: s.depth,
          pendingNewline: s.pendingNewline,
        }
        return
      }

      if (/[a-zA-Z0-9_-]/.test(ch)) {
        state = {
          _tag: 'MessageBodyOpenTag',
          id: s.id,
          dest: s.dest,
          artifactsRaw: s.artifactsRaw,
          body: s.body,
          depth: s.depth,
          pendingNewline: s.pendingNewline,
          raw: '<' + ch,
          name: ch,
          matchingName: MESSAGE_TAG.startsWith(ch),
          inName: true,
          selfClosing: false,
        }
        return
      }

      flushPendingMessageNewline(s)
      const text = '<' + ch
      s.body += text
      emit({ _tag: 'MessageBodyChunk', id: s.id, text })
      return
    }

    if (ch === '<') {
      s.pendingLt = true
      return
    }

    if (ch === '\n') {
      if (s.body.length === 0 && s.depth === 0) return
      s.pendingNewline = true
      return
    }

    flushPendingMessageNewline(s)
    s.body += ch
    emit({ _tag: 'MessageBodyChunk', id: s.id, text: ch })
  }

  function stepMessageBodyOpenTag(s: Extract<ParserState, { _tag: 'MessageBodyOpenTag' }>, ch: string): void {
    s.raw += ch

    if (s.inName && /[a-zA-Z0-9_-]/.test(ch)) {
      s.name += ch
      s.matchingName = s.matchingName && MESSAGE_TAG.startsWith(s.name)
      return
    }

    if (s.inName) {
      s.inName = false
      if (s.name !== MESSAGE_TAG) s.matchingName = false
    }

    if (ch === '>') {
      if (s.pendingNewline) {
        s.body += '\n'
        emit({ _tag: 'MessageBodyChunk', id: s.id, text: '\n' })
      }
      const text = s.raw
      s.body += text
      emit({ _tag: 'MessageBodyChunk', id: s.id, text })
      state = {
        _tag: 'MessageBody',
        id: s.id,
        dest: s.dest,
        artifactsRaw: s.artifactsRaw,
        body: s.body,
        pendingLt: false,
        depth: s.matchingName && s.name === MESSAGE_TAG && !s.selfClosing ? s.depth + 1 : s.depth,
        pendingNewline: false,
      }
      return
    }

    if (ch === '/') {
      s.selfClosing = true
    }
  }

  function stepMessageCloseTag(s: Extract<ParserState, { _tag: 'MessageCloseTag' }>, ch: string): void {
    s.close.raw += ch
    if (ch === '>') {
      if (s.close.name === MESSAGE_TAG && s.depth === 0) {
        emit({ _tag: 'MessageTagClose', id: s.id })
        state = mkProse()
        return
      }

      if (s.pendingNewline) {
        s.body += '\n'
        emit({ _tag: 'MessageBodyChunk', id: s.id, text: '\n' })
      }
      const text = s.close.raw
      s.body += text
      emit({ _tag: 'MessageBodyChunk', id: s.id, text })
      state = {
        _tag: 'MessageBody',
        id: s.id,
        dest: s.dest,
        artifactsRaw: s.artifactsRaw,
        body: s.body,
        pendingLt: false,
        depth: s.close.name === MESSAGE_TAG ? s.depth - 1 : s.depth,
        pendingNewline: false,
      }
    } else {
      s.close.name += ch
    }
  }

  // ===========================================================================
  // Step: ToolBody / ToolCloseTag
  // ===========================================================================

  function stepToolBody(s: Extract<ParserState, { _tag: 'ToolBody' }>, ch: string): void {
    if (s.pendingLt) {
      s.pendingLt = false
      if (ch === '/') {
        state = {
          _tag: 'ToolCloseTag',
          tagName: s.tagName,
          toolCallId: s.toolCallId,
          attrs: s.attrs,
          body: s.body,
          close: mkCloseTag(),
        }
      } else if (ch === '!') {
        state = {
          _tag: 'Cdata',
          cdata: { _tag: 'Prefix', index: 1, buffer: '<!' },
          origin: { _tag: 'FromToolBody', tagName: s.tagName, toolCallId: s.toolCallId, attrs: s.attrs, body: s.body },
        }
      } else if (/[a-zA-Z0-9_-]/.test(ch)) {
        if (knownTags.has(s.tagName)) {
          // Potential child tag — create parent context
          const parent: ParentCtx = {
            tagName: s.tagName,
            toolCallId: s.toolCallId,
            attrs: new Map(s.attrs),
            bodyBefore: s.body,
            children: [],
            childCounts: new Map(),
          }
          state = {
            _tag: 'ChildTagName',
            parent,
            parentBody: '',
            childName: ch,
          }
        } else {
          s.body += '<' + ch
        }
      } else {
        s.body += '<' + ch
        if (knownTags.has(s.tagName) && s.toolCallId) {
          emit({ _tag: 'BodyChunk', toolCallId: s.toolCallId, text: '<' + ch })
        }
      }
    } else if (ch === '<') {
      s.pendingLt = true
    } else {
      s.body += ch
      if (knownTags.has(s.tagName) && s.toolCallId) {
        emit({ _tag: 'BodyChunk', toolCallId: s.toolCallId, text: ch })
      }
    }
  }

  function stepToolCloseTag(s: Extract<ParserState, { _tag: 'ToolCloseTag' }>, ch: string): void {
    s.close.raw += ch
    if (ch === '>') {
      if (s.close.name === s.tagName && knownTags.has(s.tagName)) {
        // Close the tool tag
        const element: ParsedElement = {
          tagName: s.tagName,
          toolCallId: s.toolCallId,
          attributes: new Map(s.attrs),
          body: s.body,
          children: [],
        }
        emit({ _tag: 'TagClosed', toolCallId: s.toolCallId, tagName: s.tagName, element })
        state = mkProse()
      } else if (knownTags.has(s.tagName) && s.toolCallId) {
        // Mismatched close inside known tool body — treat as literal body text
        s.body += s.close.raw
        emit({ _tag: 'BodyChunk', toolCallId: s.toolCallId, text: s.close.raw })
        state = {
          _tag: 'ToolBody',
          tagName: s.tagName,
          toolCallId: s.toolCallId,
          attrs: s.attrs,
          body: s.body,
          pendingLt: false,
        }
      } else {
        // Mismatched close outside known tool — emit as prose
        const prose = mkProse()
        prose.proseAccum = emitProseChunk(prose.fence, prose.proseAccum, s.close.raw)
        state = prose
      }
    } else {
      s.close.name += ch
    }
  }

  // ===========================================================================
  // Step: ParentBody / ParentCloseTag
  // ===========================================================================

  function stepParentBody(s: Extract<ParserState, { _tag: 'ParentBody' }>, ch: string): void {
    if (s.pendingLt) {
      s.pendingLt = false
      if (ch === '/') {
        state = {
          _tag: 'ParentCloseTag',
          parent: s.parent,
          body: s.body,
          close: mkCloseTag(),
        }
      } else if (ch === '!') {
        state = {
          _tag: 'Cdata',
          cdata: { _tag: 'Prefix', index: 1, buffer: '<!' },
          origin: { _tag: 'FromParentBody', parent: s.parent, body: s.body },
        }
      } else if (/[a-zA-Z0-9_-]/.test(ch)) {
        state = {
          _tag: 'ChildTagName',
          parent: s.parent,
          parentBody: s.body,
          childName: ch,
        }
      } else {
        s.body += '<' + ch
        emit({ _tag: 'BodyChunk', toolCallId: s.parent.toolCallId, text: '<' + ch })
      }
    } else if (ch === '<') {
      s.pendingLt = true
    } else {
      s.body += ch
      emit({ _tag: 'BodyChunk', toolCallId: s.parent.toolCallId, text: ch })
    }
  }

  function stepParentCloseTag(s: Extract<ParserState, { _tag: 'ParentCloseTag' }>, ch: string): void {
    s.close.raw += ch
    if (ch === '>') {
      if (s.close.name === s.parent.tagName) {
        // Close the parent tool tag
        const element: ParsedElement = {
          tagName: s.parent.tagName,
          toolCallId: s.parent.toolCallId,
          attributes: s.parent.attrs,
          body: s.parent.bodyBefore + s.body,
          children: s.parent.children,
        }
        emit({ _tag: 'TagClosed', toolCallId: s.parent.toolCallId, tagName: s.parent.tagName, element })
        state = mkProse()
      } else {
        // Mismatched close tag — treat as body text
        s.body += s.close.raw
        emit({ _tag: 'BodyChunk', toolCallId: s.parent.toolCallId, text: s.close.raw })
        state = {
          _tag: 'ParentBody',
          parent: s.parent,
          body: s.body,
          pendingLt: false,
        }
      }
    } else {
      s.close.name += ch
    }
  }

  // ===========================================================================
  // Step: ChildTagName
  // ===========================================================================

  function stepChildTagName(s: Extract<ParserState, { _tag: 'ChildTagName' }>, ch: string): void {
    if (/[a-zA-Z0-9_-]/.test(ch)) {
      s.childName += ch
    } else if (/\s/.test(ch)) {
      if (isValidChildTag(s.parent.tagName, s.childName)) {
        state = {
          _tag: 'ChildAttrs',
          parent: s.parent,
          parentBody: s.parentBody,
          childTagName: s.childName,
          attr: mkAttrState(),
        }
      } else {
        flushInvalidChildToBody(s, '<' + s.childName + ch)
      }
    } else if (ch === '>') {
      if (isValidChildTag(s.parent.tagName, s.childName)) {
        enterChildBody(s.parent, s.parentBody, s.childName, new Map())
      } else {
        flushInvalidChildToBody(s, '<' + s.childName + '>')
      }
    } else if (ch === '/') {
      if (isValidChildTag(s.parent.tagName, s.childName)) {
        const attr = mkAttrState()
        attr.phase = { _tag: 'PendingSlash' }
        state = {
          _tag: 'ChildAttrs',
          parent: s.parent,
          parentBody: s.parentBody,
          childTagName: s.childName,
          attr,
        }
      } else {
        flushInvalidChildToBody(s, '<' + s.childName + '/')
      }
    } else {
      // Invalid tag char
      const text = '<' + s.childName + ch
      s.parentBody += text
      emit({ _tag: 'BodyChunk', toolCallId: s.parent.toolCallId, text })
      state = {
        _tag: 'ParentBody',
        parent: s.parent,
        body: s.parentBody,
        pendingLt: false,
      }
    }
  }

  function flushInvalidChildToBody(s: Extract<ParserState, { _tag: 'ChildTagName' }>, text: string): void {
    s.parentBody += text
    if (knownTags.has(s.parent.tagName) && s.parent.toolCallId) {
      emit({ _tag: 'BodyChunk', toolCallId: s.parent.toolCallId, text })
    }
    state = {
      _tag: 'ParentBody',
      parent: s.parent,
      body: s.parentBody,
      pendingLt: false,
    }
  }

  function enterChildBody(parent: ParentCtx, parentBody: string, childTagName: string, attrs: Map<string, AttributeValue>): void {
    const attrsCopy = new Map(attrs)
    const idx = getChildIndex(parent, childTagName)
    emit({
      _tag: 'ChildOpened',
      parentToolCallId: parent.toolCallId,
      childTagName,
      childIndex: idx,
      attributes: attrsCopy,
    })
    state = {
      _tag: 'ChildBody',
      parent,
      parentBody,
      childTagName,
      childAttrs: attrsCopy,
      childBody: '',
      pendingLt: false,
    }
  }

  // ===========================================================================
  // Step: ChildAttrs / ChildAttrValue / ChildUnquotedAttrValue
  // ===========================================================================

  function stepChildAttrs(s: Extract<ParserState, { _tag: 'ChildAttrs' }>, ch: string): void {
    const attr = s.attr

    if (attr.phase._tag === 'PendingSlash') {
      if (ch === '>') {
        attr.phase = { _tag: 'Idle' }
        handleChildSelfClose(s.parent, s.parentBody, s.childTagName, attr.attrs)
      } else {
        attr.key += '/'
        attr.phase = { _tag: 'Idle' }
        stepChildAttrs(s, ch)
      }
      return
    }

    if (attr.phase._tag === 'PendingEquals') {
      const key = attr.phase.key
      attr.phase = { _tag: 'Idle' }
      if (ch === '"') {
        attr.value = ''
        attr.key = key
        state = {
          _tag: 'ChildAttrValue',
          parent: s.parent,
          parentBody: s.parentBody,
          childTagName: s.childTagName,
          attr,
        }
      } else if (ch === '>' || ch === '/') {
        finalizeChildAttrVal(s.parent, s.childTagName, attr, key, '')
        attr.key = ''
        stepChildAttrs(s, ch)
      } else if (/\s/.test(ch)) {
        finalizeChildAttrVal(s.parent, s.childTagName, attr, key, '')
        attr.key = ''
      } else {
        attr.value = ch
        attr.key = key
        state = {
          _tag: 'ChildUnquotedAttrValue',
          parent: s.parent,
          parentBody: s.parentBody,
          childTagName: s.childTagName,
          attr,
        }
      }
      return
    }

    if (ch === '>') {
      if (attr.key) { finalizeChildAttrVal(s.parent, s.childTagName, attr, attr.key, ''); attr.key = '' }
      enterChildBody(s.parent, s.parentBody, s.childTagName, attr.attrs)
    } else if (ch === '/') {
      if (attr.key) { finalizeChildAttrVal(s.parent, s.childTagName, attr, attr.key, ''); attr.key = '' }
      attr.phase = { _tag: 'PendingSlash' }
    } else if (ch === '=') {
      attr.phase = { _tag: 'PendingEquals', key: attr.key }
      attr.key = ''
    } else if (ch === '"') {
      attr.value = ''
      state = {
        _tag: 'ChildAttrValue',
        parent: s.parent,
        parentBody: s.parentBody,
        childTagName: s.childTagName,
        attr,
      }
    } else if (/\s/.test(ch)) {
      if (attr.key) { finalizeChildAttrVal(s.parent, s.childTagName, attr, attr.key, ''); attr.key = '' }
    } else {
      attr.key += ch
    }
  }

  function stepChildAttrValue(s: Extract<ParserState, { _tag: 'ChildAttrValue' }>, ch: string): void {
    if (ch === '"') {
      finalizeChildAttrVal(s.parent, s.childTagName, s.attr, s.attr.key, s.attr.value)
      s.attr.key = ''
      s.attr.value = ''
      s.attr.phase = { _tag: 'Idle' }
      state = {
        _tag: 'ChildAttrs',
        parent: s.parent,
        parentBody: s.parentBody,
        childTagName: s.childTagName,
        attr: s.attr,
      }
    } else {
      s.attr.value += ch
    }
  }

  function stepChildUnquotedAttrValue(s: Extract<ParserState, { _tag: 'ChildUnquotedAttrValue' }>, ch: string): void {
    if (/\s/.test(ch)) {
      finalizeChildAttrVal(s.parent, s.childTagName, s.attr, s.attr.key, s.attr.value)
      s.attr.key = ''
      s.attr.value = ''
      s.attr.phase = { _tag: 'Idle' }
      state = {
        _tag: 'ChildAttrs',
        parent: s.parent,
        parentBody: s.parentBody,
        childTagName: s.childTagName,
        attr: s.attr,
      }
    } else if (ch === '>') {
      finalizeChildAttrVal(s.parent, s.childTagName, s.attr, s.attr.key, s.attr.value)
      s.attr.key = ''
      s.attr.value = ''
      enterChildBody(s.parent, s.parentBody, s.childTagName, s.attr.attrs)
    } else if (ch === '/') {
      finalizeChildAttrVal(s.parent, s.childTagName, s.attr, s.attr.key, s.attr.value)
      s.attr.key = ''
      s.attr.value = ''
      s.attr.phase = { _tag: 'PendingSlash' }
      state = {
        _tag: 'ChildAttrs',
        parent: s.parent,
        parentBody: s.parentBody,
        childTagName: s.childTagName,
        attr: s.attr,
      }
    } else {
      s.attr.value += ch
    }
  }

  function handleChildSelfClose(parent: ParentCtx, parentBody: string, childTagName: string, childAttrs: Map<string, AttributeValue>): void {
    // <ref /> — resolve inline and inject as body text
    if (childTagName === 'ref' && resolveRef) {
      const toolRef = childAttrs.get('tool')
      if (typeof toolRef === 'string') {
        const parsed = parseToolRef(toolRef)
        if (parsed) {
          const query = childAttrs.get('query')
          const resolved = resolveRef(parsed.tag, parsed.recency, typeof query === 'string' ? query : undefined)
          if (resolved !== undefined) {
            parentBody += resolved
            emit({ _tag: 'BodyChunk', toolCallId: parent.toolCallId, text: resolved })
          }
        }
      }
      state = {
        _tag: 'ParentBody',
        parent,
        body: parentBody,
        pendingLt: false,
      }
      return
    }

    const attrsCopy = new Map(childAttrs)
    const idx = getChildIndex(parent, childTagName)
    emit({
      _tag: 'ChildOpened',
      parentToolCallId: parent.toolCallId,
      childTagName,
      childIndex: idx,
      attributes: attrsCopy,
    })
    emit({
      _tag: 'ChildComplete',
      parentToolCallId: parent.toolCallId,
      childTagName,
      childIndex: idx,
      attributes: attrsCopy,
      body: '',
    })
    parent.children.push({ tagName: childTagName, attributes: attrsCopy, body: '' })
    incrementChildIndex(parent, childTagName)
    state = {
      _tag: 'ParentBody',
      parent,
      body: '',
      pendingLt: false,
    }
  }

  // ===========================================================================
  // Step: ChildBody / ChildCloseTag
  // ===========================================================================

  function stepChildBody(s: Extract<ParserState, { _tag: 'ChildBody' }>, ch: string): void {
    if (s.pendingLt) {
      s.pendingLt = false
      if (ch === '/') {
        state = {
          _tag: 'ChildCloseTag',
          parent: s.parent,
          parentBody: s.parentBody,
          childTagName: s.childTagName,
          childAttrs: s.childAttrs,
          childBody: s.childBody,
          close: mkCloseTag(),
        }
      } else if (ch === '!') {
        state = {
          _tag: 'Cdata',
          cdata: { _tag: 'Prefix', index: 1, buffer: '<!' },
          origin: {
            _tag: 'FromChildBody',
            parent: s.parent,
            parentBody: s.parentBody,
            childTagName: s.childTagName,
            childAttrs: s.childAttrs,
            childBody: s.childBody,
          },
        }
      } else {
        s.childBody += '<' + ch
        emit({
          _tag: 'ChildBodyChunk',
          parentToolCallId: s.parent.toolCallId,
          childTagName: s.childTagName,
          childIndex: getChildIndex(s.parent, s.childTagName),
          text: '<' + ch,
        })
      }
    } else if (ch === '<') {
      s.pendingLt = true
    } else {
      s.childBody += ch
      emit({
        _tag: 'ChildBodyChunk',
        parentToolCallId: s.parent.toolCallId,
        childTagName: s.childTagName,
        childIndex: getChildIndex(s.parent, s.childTagName),
        text: ch,
      })
    }
  }

  function stepChildCloseTag(s: Extract<ParserState, { _tag: 'ChildCloseTag' }>, ch: string): void {
    s.close.raw += ch
    if (ch === '>') {
      if (s.close.name === s.childTagName) {
        // Close the child tag
        const idx = getChildIndex(s.parent, s.childTagName)
        emit({
          _tag: 'ChildComplete',
          parentToolCallId: s.parent.toolCallId,
          childTagName: s.close.name,
          childIndex: idx,
          attributes: s.childAttrs,
          body: s.childBody,
        })
        s.parent.children.push({ tagName: s.close.name, attributes: new Map(s.childAttrs), body: s.childBody })
        incrementChildIndex(s.parent, s.close.name)
        state = {
          _tag: 'ParentBody',
          parent: s.parent,
          body: '',
          pendingLt: false,
        }
      } else if (s.close.name === s.parent.tagName) {
        // Close parent from inside child body (child never closed)
        emit({
          _tag: 'ParseError',
          error: {
            _tag: 'UnclosedChildTag',
            toolCallId: s.parent.toolCallId,
            tagName: s.parent.tagName,
            childTagName: s.childTagName,
            detail: `Child tag <${s.childTagName}> inside <${s.parent.tagName}> was never closed`,
          },
        })
        const element: ParsedElement = {
          tagName: s.parent.tagName,
          toolCallId: s.parent.toolCallId,
          attributes: s.parent.attrs,
          body: s.parent.bodyBefore + s.parentBody,
          children: s.parent.children,
        }
        emit({ _tag: 'TagClosed', toolCallId: s.parent.toolCallId, tagName: s.parent.tagName, element })
        state = mkProse()
      } else {
        // Unknown close tag inside child body — treat as child body text
        s.childBody += s.close.raw
        emit({
          _tag: 'ChildBodyChunk',
          parentToolCallId: s.parent.toolCallId,
          childTagName: s.childTagName,
          childIndex: getChildIndex(s.parent, s.childTagName),
          text: s.close.raw,
        })
        state = {
          _tag: 'ChildBody',
          parent: s.parent,
          parentBody: s.parentBody,
          childTagName: s.childTagName,
          childAttrs: s.childAttrs,
          childBody: s.childBody,
          pendingLt: false,
        }
      }
    } else {
      s.close.name += ch
    }
  }

  // ===========================================================================
  // Step: Cdata
  // ===========================================================================

  function stepCdata(s: Extract<ParserState, { _tag: 'Cdata' }>, ch: string): void {
    const cdata = s.cdata

    if (cdata._tag === 'Prefix') {
      if (ch === CDATA_PREFIX[cdata.index]) {
        cdata.buffer += ch
        cdata.index++
        if (cdata.index === CDATA_PREFIX.length) {
          // Full prefix matched — switch to content mode
          s.cdata = { _tag: 'Body', buffer: '', closeBrackets: 0 }
        }
      } else {
        // Prefix mismatch — flush accumulated buffer + current char
        const text = cdata.buffer + ch
        returnFromCdata(s.origin, text)
      }
      return
    }

    // Body phase
    if (ch === ']') {
      cdata.closeBrackets++
    } else if (ch === '>' && cdata.closeBrackets >= 2) {
      const content = cdata.buffer
      returnFromCdata(s.origin, content)
    } else {
      if (cdata.closeBrackets > 0) {
        cdata.buffer += ']'.repeat(cdata.closeBrackets)
        cdata.closeBrackets = 0
      }
      cdata.buffer += ch
    }
  }

  function returnFromCdata(origin: CdataOrigin, text: string): void {
    switch (origin._tag) {
      case 'FromProse': {
        const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: origin.fence, proseAccum: origin.proseAccum }
        for (const c of text) appendProseChar(prose, c)
        state = prose
        break
      }
      case 'FromToolBody': {
        origin.body += text
        if (knownTags.has(origin.tagName) && origin.toolCallId) {
          emit({ _tag: 'BodyChunk', toolCallId: origin.toolCallId, text })
        }
        state = {
          _tag: 'ToolBody',
          tagName: origin.tagName,
          toolCallId: origin.toolCallId,
          attrs: origin.attrs,
          body: origin.body,
          pendingLt: false,
        }
        break
      }
      case 'FromParentBody': {
        origin.body += text
        emit({ _tag: 'BodyChunk', toolCallId: origin.parent.toolCallId, text })
        state = {
          _tag: 'ParentBody',
          parent: origin.parent,
          body: origin.body,
          pendingLt: false,
        }
        break
      }
      case 'FromChildBody': {
        origin.childBody += text
        emit({
          _tag: 'ChildBodyChunk',
          parentToolCallId: origin.parent.toolCallId,
          childTagName: origin.childTagName,
          childIndex: getChildIndex(origin.parent, origin.childTagName),
          text,
        })
        state = {
          _tag: 'ChildBody',
          parent: origin.parent,
          parentBody: origin.parentBody,
          childTagName: origin.childTagName,
          childAttrs: origin.childAttrs,
          childBody: origin.childBody,
          pendingLt: false,
        }
        break
      }
    }
  }

  // ===========================================================================
  // Flush — handle incomplete state at stream end
  // ===========================================================================

  /** Extract prose context (fence + proseAccum) from states that carry it, or create fresh */
  function getProseCtx(s: ParserState): { fence: FenceState; proseAccum: string } {
    switch (s._tag) {
      case 'Prose': return { fence: s.fence, proseAccum: s.proseAccum }
      case 'TagName': return { fence: s.fence, proseAccum: s.proseAccum }
      case 'TopLevelCloseTag': return { fence: s.fence, proseAccum: s.proseAccum }
      case 'TagAttrs': return { fence: s.fence, proseAccum: s.proseAccum }
      case 'TagAttrValue': return { fence: s.fence, proseAccum: s.proseAccum }
      case 'TagUnquotedAttrValue': return { fence: s.fence, proseAccum: s.proseAccum }
      case 'Cdata':
        if (s.origin._tag === 'FromProse') return { fence: s.origin.fence, proseAccum: s.origin.proseAccum }
        return { fence: mkFence(), proseAccum: '' }
      default: return { fence: mkFence(), proseAccum: '' }
    }
  }

  function flushState(): void {
    if (state._tag === 'Prose') return

    let reconstructed = ''
    const { fence: pFence, proseAccum: pAccum } = getProseCtx(state)

    switch (state._tag) {
      case 'TagName': {
        reconstructed = '<' + state.name
        break
      }
      case 'TopLevelCloseTag': {
        reconstructed = state.close.raw
        break
      }
      case 'TagAttrs':
      case 'TagAttrValue':
      case 'TagUnquotedAttrValue': {
        const s = state
        if (knownTags.has(s.tagName)) {
          emitIncompleteError(s.toolCallId, s.tagName,
            `Tag <${s.tagName}> was opened but never closed (incomplete attributes)`)
        }
        reconstructed = `<${s.tagName}`
        for (const [k, v] of s.attr.attrs) reconstructed += ` ${k}="${v}"`
        if (s._tag === 'TagAttrValue') {
          reconstructed += ` ${s.attr.key}="${s.attr.value}`
        } else if (s._tag === 'TagUnquotedAttrValue') {
          reconstructed += ` ${s.attr.key}=${s.attr.value}`
        } else if (s.attr.key) {
          reconstructed += ` ${s.attr.key}`
        }
        if (s.attr.phase._tag === 'PendingSlash') reconstructed += '/'
        break
      }
      case 'Think':
      case 'ThinkCloseTag':
      case 'LensTagName':
      case 'LensTagAttrs': {
        emit({ _tag: 'ParseError', error: { _tag: 'UnclosedThink', detail: 'Think block was opened but never closed' } })
        if (state.think.tagName === kw.lenses) {

        } else {
          emit({ _tag: 'ProseEnd', patternId: 'think', content: state.think.body, about: state.think.about })
        }
        break
      }
      case 'PendingThinkClose': {
        if (state.think.depth > 0) {
          emit({ _tag: 'ParseError', error: { _tag: 'UnclosedThink', detail: 'Think block was opened but never closed' } })
          if (state.think.tagName !== kw.lenses) {
            emit({ _tag: 'ProseEnd', patternId: 'think', content: state.think.body + state.closeRaw, about: state.think.about })
          }
        } else {
          if (state.think.tagName !== kw.lenses) {
            emit({ _tag: 'ProseEnd', patternId: 'think', content: state.think.body, about: state.think.about })
          }
        }
        break
      }
      case 'PendingStructuralOpen': {
        // Open tag was not confirmed (no following newline before stream end) — emit as prose
        reconstructed = state.raw
        break
      }
      case 'PendingTopLevelClose': {
        // Close tag was not confirmed — emit as prose
        reconstructed = state.closeRaw
        break
      }
      case 'MessageBody':
      case 'MessageBodyOpenTag': {
        emit({ _tag: 'MessageTagClose', id: state.id })
        break
      }
      case 'MessageCloseTag': {
        emit({ _tag: 'MessageTagClose', id: state.id })
        break
      }
      case 'ToolBody': {
        emitIncompleteError(state.toolCallId, state.tagName,
          `Tag <${state.tagName}> was opened but never closed`)
        reconstructed = `<${state.tagName}`
        for (const [k, v] of state.attrs) reconstructed += ` ${k}="${v}"`
        reconstructed += `>${state.body}`
        if (state.pendingLt) reconstructed += '<'
        break
      }
      case 'ToolCloseTag': {
        emitIncompleteError(state.toolCallId, state.tagName,
          `Tag <${state.tagName}> was opened but never closed`)
        reconstructed = `<${state.tagName}`
        for (const [k, v] of state.attrs) reconstructed += ` ${k}="${v}"`
        reconstructed += `>${state.body}${state.close.raw}`
        break
      }
      case 'ParentBody': {
        emitIncompleteError(state.parent.toolCallId, state.parent.tagName,
          `Tag <${state.parent.tagName}> was opened but never closed`)
        reconstructed = `<${state.parent.tagName}`
        for (const [k, v] of state.parent.attrs) reconstructed += ` ${k}="${v}"`
        reconstructed += `>${state.parent.bodyBefore}${state.body}`
        if (state.pendingLt) reconstructed += '<'
        break
      }
      case 'ParentCloseTag': {
        emitIncompleteError(state.parent.toolCallId, state.parent.tagName,
          `Tag <${state.parent.tagName}> was opened but never closed`)
        reconstructed = `<${state.parent.tagName}`
        for (const [k, v] of state.parent.attrs) reconstructed += ` ${k}="${v}"`
        reconstructed += `>${state.parent.bodyBefore}${state.body}${state.close.raw}`
        break
      }
      case 'ChildTagName':
      case 'ChildAttrs':
      case 'ChildAttrValue':
      case 'ChildUnquotedAttrValue':
      case 'ChildBody':
      case 'ChildCloseTag': {
        const s = state
        emitIncompleteError(s.parent.toolCallId, s.parent.tagName,
          `Tag <${s.parent.tagName}> was opened but never closed (incomplete child <${
            s._tag === 'ChildTagName' ? s.childName : s.childTagName
          }>)`)
        reconstructed = `<${s.parent.tagName}`
        for (const [k, v] of s.parent.attrs) reconstructed += ` ${k}="${v}"`
        reconstructed += `>${s.parent.bodyBefore}`
        const cName = s._tag === 'ChildTagName' ? s.childName : s.childTagName
        reconstructed += `<${cName}`
        if (s._tag !== 'ChildTagName') {
          const cAttr = s._tag === 'ChildBody' || s._tag === 'ChildCloseTag' ? s.childAttrs : s.attr.attrs
          for (const [k, v] of cAttr) reconstructed += ` ${k}="${v}"`
        }
        if (s._tag === 'ChildBody') {
          reconstructed += `>${s.childBody}`
          if (s.pendingLt) reconstructed += '<'
        } else if (s._tag === 'ChildCloseTag') {
          reconstructed += `>${s.childBody}${s.close.raw}`
        }
        break
      }
      case 'Cdata': {
        const cdBuf = state.cdata.buffer
        const origin = state.origin
        switch (origin._tag) {
          case 'FromProse': {
            const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: origin.fence, proseAccum: origin.proseAccum }
            for (const c of cdBuf) appendProseChar(prose, c)
            state = prose
            return
          }
          case 'FromToolBody': {
            emitIncompleteError(origin.toolCallId, origin.tagName,
              `Tag <${origin.tagName}> was opened but never closed (incomplete CDATA section)`)
            const body = origin.body + cdBuf
            reconstructed = `<${origin.tagName}`
            for (const [k, v] of origin.attrs) reconstructed += ` ${k}="${v}"`
            reconstructed += `>${body}`
            break
          }
          case 'FromParentBody': {
            emitIncompleteError(origin.parent.toolCallId, origin.parent.tagName,
              `Tag <${origin.parent.tagName}> was opened but never closed (incomplete CDATA section)`)
            reconstructed = `<${origin.parent.tagName}`
            for (const [k, v] of origin.parent.attrs) reconstructed += ` ${k}="${v}"`
            reconstructed += `>${origin.parent.bodyBefore}${origin.body}${cdBuf}`
            break
          }
          case 'FromChildBody': {
            emitIncompleteError(origin.parent.toolCallId, origin.parent.tagName,
              `Tag <${origin.parent.tagName}> was opened but never closed (incomplete CDATA in child <${origin.childTagName}>)`)
            reconstructed = `<${origin.parent.tagName}`
            for (const [k, v] of origin.parent.attrs) reconstructed += ` ${k}="${v}"`
            reconstructed += `>${origin.parent.bodyBefore}<${origin.childTagName}>${origin.childBody}${cdBuf}`
            break
          }
        }
        break
      }
    }

    const prose: Extract<ParserState, { _tag: 'Prose' }> = { _tag: 'Prose', fence: pFence, proseAccum: pAccum }
    prose.proseAccum = emitProseChunk(pFence, pAccum, reconstructed)
    state = prose
  }

  function emitIncompleteError(toolCallId: string, tagName: string, detail: string): void {
    if (toolCallId && knownTags.has(tagName)) {
      emit({ _tag: 'ParseError', error: { _tag: 'IncompleteToolTag', toolCallId, tagName, detail } })
    }
  }

  // ===========================================================================
  // Public API
  // ===========================================================================

  return {
    processChunk(chunk: string): ParseEvent[] {
      events = []
      let i = 0
      while (i < chunk.length) {
        if (state._tag === 'Prose' && !isInStructural('inspect')) {
          // Fast prose scan optimization
          const fence = state.fence
          let canFastScan = fence.phase === FencePhase.Broken
          if (!canFastScan && fence.phase === FencePhase.LeadingWs && fence.buffer.length === 0) {
            const c = chunk[i]
            if (c !== ' ' && c !== '\t' && c !== '`' && c !== '\n' && c !== '<') {
              fence.phase = FencePhase.Broken
              canFastScan = true
            }
          }

          if (canFastScan) {
            const start = i
            while (i < chunk.length) {
              const c = chunk[i]
              if (c === '<' || c === '\n') break
              i++
            }
            if (i > start) {
              ctx.lastCharNewline = false
              state.proseAccum = flushPendingWhitespace(fence, state.proseAccum)
              state.proseAccum = rawEmitProse(fence, state.proseAccum, chunk.slice(start, i))
            }
            if (i < chunk.length) {
              processChar(chunk[i])
              i++
            }
          } else {
            processChar(chunk[i])
            i++
          }
        } else if (state._tag === 'MessageBody' && !state.pendingLt && !state.openTagBuf) {
          const start = i
          while (i < chunk.length) {
            const c = chunk[i]
            if (c === '<' || c === '\n') break
            i++
          }
          if (i > start) {
            flushPendingMessageNewline(state)
            const text = chunk.slice(start, i)
            state.body += text
            emit({ _tag: 'MessageBodyChunk', id: state.id, text })
          }
          if (i < chunk.length) {
            processChar(chunk[i])
            i++
          }
        } else {
          processChar(chunk[i])
          i++
        }
      }
      const result = events
      events = []
      return result
    },

    flush(): ParseEvent[] {
      events = []

      flushState()

      if (isInStructural(kw.actions)) {
        emit({ _tag: 'ParseError', error: { _tag: 'UnclosedActions', detail: 'Actions block was opened but never closed' } })
      }
      if (isInStructural('inspect')) {
        emit({ _tag: 'ParseError', error: { _tag: 'UnclosedInspect', detail: 'Inspect block was opened but never closed' } })
      }
      if (isInStructural(kw.comms)) {
        emit({ _tag: 'CommsClose' })
      }

      // state is now guaranteed to be Prose
      const prose = state as Extract<ParserState, { _tag: 'Prose' }>
      const fence = prose.fence

      flushDeferredFence(fence, prose.proseAccum)
      prose.proseAccum = flushDeferredFence(fence, prose.proseAccum)
      if (ctx.justClosedStructural && isFenceComplete(fence.phase)) {
        fence.buffer = ''
        fence.pendingWhitespace = ''
      } else {
        prose.proseAccum = flushFenceBuffer(fence, prose.proseAccum)
      }
      endProseBlock(fence, prose.proseAccum)

      const result = events
      events = []
      return result
    },
  }
}
