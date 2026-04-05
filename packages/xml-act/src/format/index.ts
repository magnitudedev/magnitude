import { xmlActContent } from './content'
import { xmlActFlush } from './flush'
import { finishHandler } from './handlers/finish'
import { assignHandler } from './handlers/assign'
import { reassignHandler } from './handlers/reassign'
import { messageHandler } from './handlers/message'
import { taskHandler } from './handlers/task'
import { lensHandler, thinkHandler } from './handlers/think'
import { childHandler, toolHandler } from './handlers/tool'
import { turnControlHandler } from './handlers/turn-control'
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

  const aliasMap = aliases ?? new Map<string, string>([
    ['thinking', 'think'],
    ['lenses', 'lenses'],
  ])

  const messageTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
  const insideLensTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
  const plainThinkTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
  const topLevelTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
  const taskFrameTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
  const betweenLensTags: TagMap = structuralTags

  const lens = lensHandler(betweenLensTags, insideLensTags)
  handlers.set('lens', lens)

  const think = thinkHandler('lenses', betweenLensTags, plainThinkTags)
  const lenses = thinkHandler('lenses', betweenLensTags, plainThinkTags)
  const thinking = thinkHandler('lenses', betweenLensTags, plainThinkTags)
  const message = messageHandler(messageTags)
  const assign = assignHandler(taskFrameTags)
  const reassign = reassignHandler(taskFrameTags)
  const task = taskHandler(taskFrameTags)

  handlers.set('think', think)
  handlers.set('lenses', lenses)
  handlers.set('thinking', thinking)
  handlers.set('message', message)
  handlers.set('task', task)
  handlers.set('assign', assign)
  handlers.set('reassign', reassign)
  handlers.set('idle', turnControlHandler('idle'))
  handlers.set('finish', finishHandler())

  for (const tool of tools) {
    const toolTags: Map<string, TagHandler<XmlActFrame, XmlActEvent>> = new Map()
    for (const childTag of tool.childTags) {
      toolTags.set(childTag, childHandler())
    }
    const selfToolHandler = toolHandler(tool.tag, tool.childTags, tool.schema, toolTags)
    if (!tool.childTags.has(tool.tag)) {
      toolTags.set(tool.tag, selfToolHandler)
    }
    handlers.set(tool.tag, selfToolHandler)
    taskFrameTags.set(tool.tag, selfToolHandler)
  }

  const messageEntry = handlers.get('message')
  if (messageEntry) {
    messageTags.set('message', messageEntry)
  }

  for (const tag of ['message', 'task', 'assign', 'reassign', 'idle', 'finish', 'think', 'lenses', 'thinking', 'lens']) {
    const handler = handlers.get(tag)
    if (handler) topLevelTags.set(tag, handler)
  }
  for (const tool of tools) {
    const handler = handlers.get(tool.tag)
    if (handler) topLevelTags.set(tool.tag, handler)
  }

  for (const tag of ['message', 'assign', 'reassign', 'task']) {
    const handler = handlers.get(tag)
    if (handler) taskFrameTags.set(tag, handler)
  }

  const lensEntry = handlers.get('lens')
  if (lensEntry) insideLensTags.set('lens', lensEntry)
  const lensesEntry = handlers.get('lenses')
  if (lensesEntry) insideLensTags.set('lenses', lensesEntry)

  const thinkEntry = handlers.get('think')
  if (thinkEntry) plainThinkTags.set('think', thinkEntry)
  const thinkingEntry = handlers.get('thinking')
  if (thinkingEntry) plainThinkTags.set('thinking', thinkingEntry)

  for (const [alias, canonical] of aliasMap) {
    if (canonical === 'think') {
      const aliasThink = handlers.get('think')
      if (aliasThink) plainThinkTags.set(alias, aliasThink)
    }
    if (canonical === 'lenses') {
      const aliasLenses = handlers.get('lenses')
      if (aliasLenses) insideLensTags.set(alias, aliasLenses)
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
    readonly task: 'task'
    readonly assign: 'assign'
    readonly think: 'think'
    readonly message: 'message'
    readonly idle: 'idle'
    readonly finish: 'finish'
  }
} {
  const { format, structuralTags } = createXmlActFormat(tools, defaultMessageDest)
  return {
    format,
    structuralTags,
    tags: {
      task: 'task',
      assign: 'assign',
      think: 'think',
      message: 'message',
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
export { taskHandler } from './handlers/task'
export { assignHandler } from './handlers/assign'
export { reassignHandler } from './handlers/reassign'
export { thinkHandler, lensHandler } from './handlers/think'
export { messageHandler } from './handlers/message'
export { toolHandler, childHandler } from './handlers/tool'
export { turnControlHandler } from './handlers/turn-control'
export { finishHandler } from './handlers/finish'
export { HANDLE, PASS } from './types'
