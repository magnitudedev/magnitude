import type { ParseStack, ParserConfig } from './types'


export function isInContainer(stack: ParseStack, tag: 'Actions' | 'Inspect' | 'Comms'): boolean {
  return stack.some(frame => frame._tag === tag)
}

export function containerDepth(stack: ParseStack, tag: 'Actions' | 'Inspect' | 'Comms'): number {
  return stack.filter(frame => frame._tag === tag).length
}

export function innermostContainer(stack: ParseStack): 'Actions' | 'Inspect' | 'Comms' | null {
  for (let i = stack.length - 1; i >= 0; i--) {
    const frame = stack[i]
    if (!frame) continue
    const tag = frame._tag
    if (tag === 'Actions' || tag === 'Inspect' || tag === 'Comms') return tag
  }
  return null
}

export function activeTags(stack: ParseStack, afterNewline: boolean, config: ParserConfig): ReadonlySet<string> {
  const container = innermostContainer(stack)
  if (container === 'Inspect') return afterNewline ? new Set([...config.refTags, 'inspect']) : config.refTags
  if (container === 'Comms') return afterNewline ? new Set([...config.messageTags, config.keywords.comms]) : config.messageTags
  if (container === 'Actions') return afterNewline ? config.actionsTags : config.knownTags
  return afterNewline ? config.topLevelTags : new Set()
}