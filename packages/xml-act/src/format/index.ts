import { xmlActContent } from './content'
import { xmlActFlush } from './flush'
import { finishHandler } from './handlers/finish'
import { messageHandler } from './handlers/message'
import { lensHandler, thinkHandler } from './handlers/think'
import { childHandler, toolHandler } from './handlers/tool'

import { endTurnHandler } from './handlers/end-turn'
import { TURN_CONTROL_FINISH_TAG, END_TURN_TAG } from '../constants'
import type { Format, TagHandler, TagMap, ToolDef, XmlActEvent, XmlActFrame } from './types'
import { xmlActUnknownClose, xmlActUnknownOpen } from './unknown'

export function createXmlActFormat(
  tools: readonly ToolDef[],
  _defaultMessageDest: string,
  aliases?: ReadonlyMap<string, string>,
): {
  readonly format: Format<XmlActFrame, XmlActEvent>
  readonly structuralTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>>
} {
  const handlers = new Map<string, TagHandler<XmlActFrame, XmlActEvent>>()
  const structuralTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()

  const aliasMap = aliases ?? new Map<string, string>([['thinking', 'think']])

  const messageTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
  const insideLensTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
  const plainThinkTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
  const topLevelTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
  const lens = lensHandler(structuralTags, insideLensTags)
  handlers.set('lens', lens)

  const think = thinkHandler(plainThinkTags)
  const thinking = thinkHandler(plainThinkTags)
  const message = messageHandler(messageTags)
  handlers.set('think', think)
  handlers.set('thinking', thinking)
  handlers.set('message', message)
  handlers.set(TURN_CONTROL_FINISH_TAG, finishHandler())

  handlers.set(END_TURN_TAG, endTurnHandler())

  for (const tool of tools) {
    // Skip tags that already have dedicated structural handlers
    if (handlers.has(tool.tag)) continue
    
    const toolTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
    for (const childTag of tool.childTags) {
      toolTags.set(childTag, childHandler())
    }
    const selfToolHandler = toolHandler(tool.tag, tool.childTags, tool.schema, toolTags)
    if (!tool.childTags.has(tool.tag)) {
      toolTags.set(tool.tag, selfToolHandler)
    }
    handlers.set(tool.tag, selfToolHandler)

  }

  const messageEntry = handlers.get('message')
  if (messageEntry) {
    messageTags.set('message', messageEntry)
  }

  for (const tag of ['message', TURN_CONTROL_FINISH_TAG, END_TURN_TAG, 'think', 'thinking', 'lens']) {
    const handler = handlers.get(tag)
    if (handler) topLevelTags.set(tag, handler)
  }
  for (const tool of tools) {
    const handler = handlers.get(tool.tag)
    if (handler) topLevelTags.set(tool.tag, handler)
  }

  const lensEntry = handlers.get('lens')
  if (lensEntry) insideLensTags.set('lens', lensEntry)

  const thinkEntry = handlers.get('think')
  if (thinkEntry) plainThinkTags.set('think', thinkEntry)
  const thinkingEntry = handlers.get('thinking')
  if (thinkingEntry) plainThinkTags.set('thinking', thinkingEntry)

  for (const [alias, canonical] of aliasMap) {
    if (canonical === 'think') {
      const aliasThink = handlers.get('think')
      if (aliasThink) plainThinkTags.set(alias, aliasThink)
    }
  }

  for (const [tag, handler] of topLevelTags) {
    structuralTags.set(tag, handler)
  }
  for (const [alias, canonical] of aliasMap) {
    const aliased = handlers.get(canonical)
    if (aliased) {
      structuralTags.set(alias, aliased)
    }
  }

  return {
    format: {
      resolve(tagName, stack) {
        const top = stack[stack.length - 1]
        if (!top) return { _tag: 'passthrough' }
        const handler = top.tags.get(tagName)
        if (!handler) return { _tag: 'passthrough' }
        return { _tag: 'handle', handler }
      },
      onContent: xmlActContent,
      onFlush: xmlActFlush,
      onUnknownOpen: xmlActUnknownOpen,
      onUnknownClose: xmlActUnknownClose,
    },
    structuralTags,
  }
}

export function createCurrentFormat(
  tools: readonly ToolDef[],
  defaultMessageDest = 'user',
): {
  readonly format: Format<XmlActFrame, XmlActEvent>
  readonly structuralTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>>
  readonly tags: {
    readonly think: 'think'
    readonly message: 'message'
    readonly continue: 'continue'
    readonly idle: 'idle'
    readonly finish: 'finish'
  }
} {
  const { format, structuralTags } = createXmlActFormat(tools, defaultMessageDest)
  return {
    format,
    structuralTags,
    tags: {
      think: 'think',
      message: 'message',
      continue: 'continue',
      idle: 'idle',
      finish: 'finish',
    },
  }
}

export type {
  XmlActFrame,
  XmlActEvent,
  ParseErrorDetail,
  ActiveLens,
  CompletedLens,
  ToolDef,
  Format,
  TagHandler,
  TagMap,
  ResolveResult,
  OpenContext,
  CloseContext,
  SelfCloseContext,
} from './types'
export { xmlActContent } from './content'
export { xmlActFlush } from './flush'
export { xmlActUnknownOpen, xmlActUnknownClose } from './unknown'
export { thinkHandler, lensHandler } from './handlers/think'
export { messageHandler } from './handlers/message'
export { toolHandler, childHandler } from './handlers/tool'

export { finishHandler } from './handlers/finish'
export { HANDLE, PASS } from './types'
