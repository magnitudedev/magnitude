import { xmlActContent } from './content'
import { xmlActFlush } from './flush'
import { finishHandler } from './handlers/finish'
import { containerHandler } from './handlers/container'
import { messageHandler } from './handlers/message'
import { lensHandler, thinkHandler } from './handlers/think'
import { childHandler, toolHandler } from './handlers/tool'
import { turnControlHandler } from './handlers/turn-control'
import { HANDLE, PASS } from './types'
import type { Format, Resolve, TagHandler, ToolDef, XmlActEvent, XmlActFrame } from './types'
import { xmlActUnknownClose, xmlActUnknownOpen } from './unknown'

export function createXmlActFormat(
  tools: readonly ToolDef[],
  defaultMessageDest: string,
  aliases?: ReadonlyMap<string, string>,
): Format<XmlActFrame, XmlActEvent> {
  const handlers = new Map<string, TagHandler<XmlActFrame, XmlActEvent>>()

  const aliasMap = aliases ?? new Map<string, string>([
    ['thinking', 'think'],
    ['lenses', 'lenses'],
    ['reason', 'think'],
    ['tooluse', 'actions'],
    ['respond', 'comms'],
  ])

  const structuralResolve: Resolve = (tagName) => {
    const canonical = aliasMap.get(tagName) ?? tagName
    const handler = handlers.get(canonical)
    if (!handler) return PASS
    return HANDLE(handler)
  }

  const lens = lensHandler()

  let message: TagHandler<XmlActFrame, XmlActEvent> | undefined
  const messageResolve: Resolve = (tagName) => {
    if (tagName === 'message' && message) return HANDLE(message)
    return PASS
  }
  message = messageHandler(defaultMessageDest, messageResolve)

  function makeThinkResolve(
    thinkTag: string,
    isLenses: boolean,
    getSelf: () => TagHandler<XmlActFrame, XmlActEvent> | undefined,
  ): Resolve {
    return (tagName) => {
      if (isLenses && tagName === 'lens') return HANDLE(lens)
      if (tagName === thinkTag) {
        const self = getSelf()
        if (self) return HANDLE(self)
      }
      return PASS
    }
  }

  let think: TagHandler<XmlActFrame, XmlActEvent> | undefined
  think = thinkHandler('lenses', (tagName, isLenses) => makeThinkResolve(tagName, isLenses, () => think))

  let lenses: TagHandler<XmlActFrame, XmlActEvent> | undefined
  lenses = thinkHandler('lenses', (tagName, isLenses) => makeThinkResolve(tagName, isLenses, () => lenses))

  let thinking: TagHandler<XmlActFrame, XmlActEvent> | undefined
  thinking = thinkHandler('lenses', (tagName, isLenses) => makeThinkResolve(tagName, isLenses, () => thinking))

  handlers.set('actions', containerHandler('actions', structuralResolve))
  handlers.set('comms', containerHandler('comms', structuralResolve))
  if (think) handlers.set('think', think)
  if (lenses) handlers.set('lenses', lenses)
  if (thinking) handlers.set('thinking', thinking)
  handlers.set('lens', lens)
  handlers.set('message', message)
  handlers.set('next', turnControlHandler('continue'))
  handlers.set('yield', turnControlHandler('yield'))
  handlers.set('finish', finishHandler())

  for (const tool of tools) {
    let selfHandler: TagHandler<XmlActFrame, XmlActEvent> | undefined
    const toolResolve: Resolve = (tagName) => {
      if (tool.childTags.has(tagName)) return HANDLE(childHandler())
      if (tagName === tool.tag && selfHandler) return HANDLE(selfHandler)
      return PASS
    }
    selfHandler = toolHandler(tool.tag, tool.childTags, tool.schema, toolResolve)
    handlers.set(tool.tag, selfHandler)
  }

  return {
    resolve(tagName, stack) {
      const top = stack[stack.length - 1]
      if (!top) return structuralResolve(tagName)
      const resolved = top.resolve(tagName)
      if (resolved._tag === 'handle') return resolved
      if (top.type === 'think' && top.isLenses && !top.activeLens) {
        return structuralResolve(tagName)
      }
      return resolved
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
  ResolveResult,
  Resolve,
  FrameResolve,
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
export { HANDLE, PASS, PASSTHROUGH } from './types'