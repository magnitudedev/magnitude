import { done, emit, pop } from '../ops'
import { endTopProse } from '../prose'
import type { TagHandler, XmlActEvent, XmlActFrame } from '../types'

function autoCloseContainer(stack: ReadonlyArray<XmlActFrame>) {
  for (let i = stack.length - 1; i >= 0; i--) {
    const frame = stack[i]
    if (frame.type === 'container') {
      return [emit({ _tag: 'ContainerClose', tag: frame.tag }), pop]
    }
  }
  return []
}

export function turnControlHandler(
  decision: 'continue' | 'yield',
): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open() {
      return []
    },
    close() {
      return []
    },
    selfClose(ctx) {
      return [...endTopProse(ctx.stack), ...autoCloseContainer(ctx.stack), emit({ _tag: 'TurnControl', decision }), done]
    },
  }
}