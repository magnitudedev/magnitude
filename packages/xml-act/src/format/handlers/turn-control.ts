import { done, emit } from '../ops'
import { endTopProse } from '../prose'
import type { TagHandler, XmlActEvent, XmlActFrame } from '../types'

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
      return [...endTopProse(ctx.stack), emit({ _tag: 'TurnControl', decision }), done]
    },
  }
}
