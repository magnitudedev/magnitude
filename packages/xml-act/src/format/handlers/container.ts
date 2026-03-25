import { emit, pop, push, replace } from '../ops'
import { appendTopProse, endTopProse } from '../prose'
import { rawCloseTag, rawOpenTag } from '../raw'
import type { Fx } from '../ops'
import type { TagHandler, XmlActFrame, XmlActEvent } from '../types'

function closeOpenThink(stack: ReadonlyArray<XmlActFrame>): Fx[] {
  const top = stack[stack.length - 1]
  if (top?.type !== 'think') return []
  if (top.isLenses) {
    if (top.activeLens) {
      return [emit({ _tag: 'LensEnd', name: top.activeLens.name, content: top.activeLens.body }), pop]
    }
    return [pop]
  }
  return [emit({ _tag: 'ProseEnd', patternId: 'think', content: top.body, about: top.about }), pop]
}

function closeInnermostContainer(stack: ReadonlyArray<XmlActFrame>, skipTag: string): Fx[] {
  for (let i = stack.length - 1; i >= 0; i--) {
    const frame = stack[i]
    if (frame.type === 'container' && frame.tag !== skipTag) {
      if (skipTag === 'comms' && frame.tag === 'actions') return []
      return [emit({ _tag: 'ContainerClose', tag: frame.tag }), pop]
    }
  }
  return []
}

export function containerHandler(tag: string): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      if (!ctx.afterNewline) {
        return appendTopProse(ctx.stack, rawOpenTag(tag, ctx.attrs))
      }
      const active = ctx.stack.findLast((f): f is Extract<XmlActFrame, { type: 'container' }> => f.type === 'container' && f.tag === tag)
      if (active) {
        return [
          replace({ ...active, depth: active.depth + 1 }),
          ...appendTopProse(ctx.stack, rawOpenTag(tag, ctx.attrs)),
        ]
      }
      return [
        ...endTopProse(ctx.stack),
        ...closeOpenThink(ctx.stack),
        ...closeInnermostContainer(ctx.stack, tag),
        push({ type: 'container', tag, depth: 0 }),
        emit({ _tag: 'ContainerOpen', tag }),
      ]
    },
    close(ctx) {
      const active = ctx.stack.findLast((f): f is Extract<XmlActFrame, { type: 'container' }> => f.type === 'container' && f.tag === tag)
      if (!active) return []
      if (active.depth > 0) {
        return [
          replace({ ...active, depth: active.depth - 1 }),
          ...appendTopProse(ctx.stack, rawCloseTag(tag)),
        ]
      }
      if (tag === 'actions' && !ctx.afterNewline) {
        return appendTopProse(ctx.stack, rawCloseTag(tag))
      }
      return [emit({ _tag: 'ContainerClose', tag }), pop]
    },
    selfClose() {
      return []
    },
  }
}