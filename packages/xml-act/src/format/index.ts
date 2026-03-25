import { xmlActContent } from './content'
import { xmlActFlush } from './flush'
import { finishHandler } from './handlers/finish'
import { containerHandler } from './handlers/container'
import { messageHandler } from './handlers/message'
import { lensHandler, thinkHandler } from './handlers/think'
import { childHandler, toolHandler } from './handlers/tool'
import { turnControlHandler } from './handlers/turn-control'
import type { Format, TagHandler, ToolDef, XmlActEvent, XmlActFrame } from './types'
import { xmlActUnknownClose, xmlActUnknownOpen } from './unknown'

export function createXmlActFormat(
  tools: readonly ToolDef[],
  defaultMessageDest: string,
  aliases?: ReadonlyMap<string, string>,
): Format<XmlActFrame, XmlActEvent> {
  const handlers = new Map<string, TagHandler<XmlActFrame, XmlActEvent>>()

  handlers.set('actions', containerHandler('actions'))
  handlers.set('comms', containerHandler('comms'))
  handlers.set('think', thinkHandler('lenses'))
  handlers.set('lenses', thinkHandler('lenses'))
  handlers.set('thinking', thinkHandler('lenses'))
  handlers.set('lens', lensHandler())
  handlers.set('message', messageHandler(defaultMessageDest))
  handlers.set('next', turnControlHandler('continue'))
  handlers.set('yield', turnControlHandler('yield'))
  handlers.set('finish', finishHandler())

  for (const tool of tools) {
    handlers.set(tool.tag, toolHandler(tool.tag, tool.childTags, tool.schema))
  }

  const aliasMap = aliases ?? new Map<string, string>([
    ['thinking', 'think'],
    ['lenses', 'lenses'],
    ['reason', 'think'],
    ['tooluse', 'actions'],
    ['respond', 'comms'],
  ])

  return {
    resolve(tagName, stack) {
      const top = stack[stack.length - 1]
      if (top?.type === 'child-body' && top.childTagName === tagName) {
        return childHandler()
      }
      if (top?.type === 'tool-body') {
        if (top.childTags.has(tagName)) return childHandler()
        if (top.tag === tagName) return handlers.get(tagName)
        return undefined
      }
      if (top?.type === 'message') {
        if (tagName === 'message') return handlers.get('message')
        return undefined
      }
      if (top?.type === 'think' && top.isLenses && tagName === 'lens') {
        return lensHandler()
      }
      const canonical = aliasMap.get(tagName) ?? tagName
      return handlers.get(canonical)
    },
    onContent: xmlActContent,
    onFlush: xmlActFlush,
    onUnknownOpen: xmlActUnknownOpen,
    onUnknownClose: xmlActUnknownClose,
  }
}

export function createCurrentFormat(
  tools: readonly ToolDef[],
  defaultMessageDest = 'user',
): {
  readonly format: Format<XmlActFrame, XmlActEvent>
  readonly tags: {
    readonly actions: 'actions'
    readonly comms: 'comms'
    readonly think: 'think'
    readonly message: 'message'
    readonly next: 'next'
    readonly yield: 'yield'
    readonly finish: 'finish'
  }
} {
  return {
    format: createXmlActFormat(tools, defaultMessageDest),
    tags: {
      actions: 'actions',
      comms: 'comms',
      think: 'think',
      message: 'message',
      next: 'next',
      yield: 'yield',
      finish: 'finish',
    },
  }
}

export function createAltFormat(
  tools: readonly ToolDef[],
  defaultMessageDest = 'user',
): {
  readonly format: Format<XmlActFrame, XmlActEvent>
  readonly tags: {
    readonly actions: 'tooluse'
    readonly comms: 'respond'
    readonly think: 'reason'
    readonly message: 'message'
    readonly next: 'next'
    readonly yield: 'yield'
    readonly finish: 'finish'
  }
} {
  const aliases = new Map<string, string>([
    ['tooluse', 'actions'],
    ['respond', 'comms'],
    ['reason', 'think'],
  ])
  return {
    format: createXmlActFormat(tools, defaultMessageDest, aliases),
    tags: {
      actions: 'tooluse',
      comms: 'respond',
      think: 'reason',
      message: 'message',
      next: 'next',
      yield: 'yield',
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
  OpenContext,
  CloseContext,
  SelfCloseContext,
} from './types'
export { xmlActContent } from './content'
export { xmlActFlush } from './flush'
export { xmlActUnknownOpen, xmlActUnknownClose } from './unknown'
export { containerHandler } from './handlers/container'
export { thinkHandler, lensHandler } from './handlers/think'
export { messageHandler } from './handlers/message'
export { toolHandler, childHandler } from './handlers/tool'
export { turnControlHandler } from './handlers/turn-control'
export { finishHandler } from './handlers/finish'