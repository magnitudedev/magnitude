import { done, emit, pop } from '../ops'
import { endTopProse } from '../prose'
import type { Fx } from '../ops'
import type { TagHandler, XmlActEvent, XmlActFrame } from '../types'

function autoCloseTasks(stack: ReadonlyArray<XmlActFrame>): Fx[] {
  const ops: Fx[] = []
  for (let i = stack.length - 1; i >= 0; i--) {
    const frame = stack[i]
    if (frame.type === 'task') {
      ops.push(emit({ _tag: 'TaskClose', id: frame.id }))
      ops.push(pop)
    }
  }
  return ops
}

export function turnControlHandler(
  decision: 'idle',
): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open() {
      return []
    },
    close() {
      return []
    },
    selfClose(ctx) {
      return [...endTopProse(ctx.stack), ...autoCloseTasks(ctx.stack), emit({ _tag: 'TurnControl', decision }), done]
    },
  }
}
