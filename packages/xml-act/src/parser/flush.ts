import type { ParseEvent, ParseStack, ParserConfig } from './types'
import { emitProseChunk, endProseBlock, flushDeferredFence, flushFenceBuffer, isFenceComplete } from './prose'
import { containerDepth } from './stack-ops'

function emitIncompleteError(config: ParserConfig, toolCallId: string, tagName: string, detail: string): ParseEvent[] {
  if (toolCallId && config.knownTags.has(tagName)) {
    return [{ _tag: 'ParseError', error: { _tag: 'IncompleteToolTag', toolCallId, tagName, detail } }]
  }
  return []
}

export function flushStack(state: ParseStack, config: ParserConfig): ParseEvent[] {
  const events: ParseEvent[] = []
  let sawActions = false
  let sawInspect = false
  let sawComms = false

  while (state.length > 1) {
    const frame = state.pop()
    if (!frame) break
    switch (frame._tag) {
      case 'Think':
      case 'ThinkCloseTag':
      case 'LensTagName':
      case 'LensTagAttrs':
        events.push({ _tag: 'ParseError', error: { _tag: 'UnclosedThink', detail: 'Think block was opened but never closed' } })
        if (frame.think.tagName !== config.keywords.lenses) {
          events.push({ _tag: 'ProseEnd', patternId: 'think', content: frame.think.body, about: frame.think.about })
        }
        break
      case 'PendingThinkClose':
        if (frame.think.depth > 0) events.push({ _tag: 'ParseError', error: { _tag: 'UnclosedThink', detail: 'Think block was opened but never closed' } })
        if (frame.think.tagName !== config.keywords.lenses) {
          events.push({ _tag: 'ProseEnd', patternId: 'think', content: frame.think.depth > 0 ? frame.think.body + frame.closeRaw : frame.think.body, about: frame.think.about })
        }
        break
      case 'PendingStructuralOpen':
        events.push(...emitProseChunk(state, frame.raw))
        break
      case 'PendingTopLevelClose':
        events.push(...emitProseChunk(state, frame.closeRaw))
        break
      case 'MessageBody':
      case 'MessageBodyOpenTag':
      case 'MessageCloseTag':
        events.push({ _tag: 'MessageTagClose', id: frame.id })
        break
      case 'ToolBody':
      case 'ToolCloseTag': {
        events.push(...emitIncompleteError(config, frame.toolCallId, frame.tagName, `Tag <${frame.tagName}> was opened but never closed`))
        let reconstructed = `<${frame.tagName}`
        for (const [k, v] of frame.attrs) reconstructed += ` ${k}="${v}"`
        reconstructed += `>${frame.body}`
        if (frame._tag === 'ToolBody' && frame.pendingLt) reconstructed += '<'
        if (frame._tag === 'ToolCloseTag') reconstructed += frame.close.raw
        events.push(...emitProseChunk(state, reconstructed))
        break
      }
      case 'ChildTagName':
      case 'ChildAttrs':
      case 'ChildAttrValue':
      case 'ChildUnquotedAttrValue':
      case 'ChildBody':
      case 'ChildCloseTag':
        events.push(...emitIncompleteError(config, frame.tool.toolCallId, frame.tool.tagName, `Tag <${frame.tool.tagName}> was opened but never closed (incomplete child <${frame._tag === 'ChildTagName' ? frame.childName : frame.childTagName}>)`))
        break
      case 'Cdata':
        if (frame.origin._tag === 'Prose') events.push(...emitProseChunk(state, frame.cdata.buffer))
        else {
          const tool = frame.origin._tag === 'ToolBody' ? frame.origin : frame.origin.tool
          events.push(...emitIncompleteError(config, tool.toolCallId, tool.tagName, `Tag <${tool.tagName}> was opened but never closed (incomplete CDATA section)`))
        }
        break
      case 'TagName':
        events.push(...emitProseChunk(state, '<' + frame.name))
        break
      case 'TopLevelCloseTag':
        events.push(...emitProseChunk(state, frame.close.raw))
        break
      case 'TagAttrs':
      case 'TagAttrValue':
      case 'TagUnquotedAttrValue': {
        if (config.knownTags.has(frame.tagName)) {
          events.push(...emitIncompleteError(config, frame.toolCallId, frame.tagName, `Tag <${frame.tagName}> was opened but never closed (incomplete attributes)`))
        }
        let reconstructed = `<${frame.tagName}`
        for (const [k, v] of frame.attr.attrs) reconstructed += ` ${k}="${v}"`
        if (frame._tag === 'TagAttrValue') reconstructed += ` ${frame.attr.key}="${frame.attr.value}`
        else if (frame._tag === 'TagUnquotedAttrValue') reconstructed += ` ${frame.attr.key}=${frame.attr.value}`
        else if (frame.attr.key) reconstructed += ` ${frame.attr.key}`
        if (frame.attr.phase._tag === 'PendingSlash') reconstructed += '/'
        events.push(...emitProseChunk(state, reconstructed))
        break
      }
      case 'Actions':
        sawActions = true
        break
      case 'Inspect':
        sawInspect = true
        break
      case 'Comms':
        sawComms = true
        break
      case 'Done':
        break
      default:
        break
    }
  }

  if (sawActions || containerDepth(state, 'Actions')) events.push({ _tag: 'ParseError', error: { _tag: 'UnclosedActions', detail: 'Actions block was opened but never closed' } })
  if (sawInspect || containerDepth(state, 'Inspect')) events.push({ _tag: 'ParseError', error: { _tag: 'UnclosedInspect', detail: 'Inspect block was opened but never closed' } })
  if (sawComms || containerDepth(state, 'Comms')) events.push({ _tag: 'CommsClose' })

  const prose = state[0]
  events.push(...flushDeferredFence(state))
  if (prose.justClosedStructural && isFenceComplete(prose.fence.phase)) {
    prose.fence.buffer = ''
    prose.fence.pendingWhitespace = ''
  } else events.push(...flushFenceBuffer(state))
  events.push(...endProseBlock(state))

  return events
}