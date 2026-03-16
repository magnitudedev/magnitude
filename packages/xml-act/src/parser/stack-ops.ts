import type { ParseStack, ParserConfig } from './types'


export function isInContainer(stack: ParseStack, tag: 'Actions' | 'Comms'): boolean {
  return stack.some(frame => frame._tag === tag)
}

export function containerDepth(stack: ParseStack, tag: 'Actions' | 'Comms'): number {
  return stack.filter(frame => frame._tag === tag).length
}

export function innermostContainer(stack: ParseStack): 'Actions' | 'Comms' | null {
  for (let i = stack.length - 1; i >= 0; i--) {
    const frame = stack[i]
    if (!frame) continue
    const tag = frame._tag
    if (tag === 'Actions' || tag === 'Comms') return tag
  }
  return null
}

export function activeTags(stack: ParseStack, afterNewline: boolean, config: ParserConfig): ReadonlySet<string> {
  const container = innermostContainer(stack)
  if (container === 'Comms') {
    return afterNewline
      ? new Set([
          ...config.messageTags,
          config.keywords.comms,
          config.keywords.actions,
          config.keywords.next,
          config.keywords.yield,
        ])
      : config.messageTags
  }
  if (container === 'Actions') return afterNewline ? config.actionsTags : config.knownTags
  return afterNewline ? config.topLevelTags : new Set()
}

export function activeOpenCandidates(stack: ParseStack, afterNewline: boolean, config: ParserConfig): string[] {
  const tags = activeTags(stack, afterNewline, config)
  if (afterNewline) return Array.from(tags)
  // When not after newline, also include think/thinking/lenses for PendingStructuralOpen
  // but NOT other structural tags like actions/comms/next/yield
  const kw = config.keywords
  const combined = new Set([...tags, kw.think, kw.thinking, kw.lenses])
  return Array.from(combined)
}

export function activeCloseCandidates(stack: ParseStack, afterNewline: boolean, config: ParserConfig): string[] {
  const container = innermostContainer(stack)
  if (container === 'Actions') return [config.keywords.actions]
  if (container === 'Comms') return [config.keywords.comms]
  return []
}