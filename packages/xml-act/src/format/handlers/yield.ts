
import { observe, emit } from '../ops'
import { endTopProse } from '../prose'
import type { TagHandler, XmlActEvent, XmlActFrame } from '../types'

/**
 * Handler for yield tags (<yield-user/>, <yield-tool/>, <yield-worker/>, <yield-parent/>).
 * Self-closing only - emits TurnControl with the target and terminates the turn.
 */
export function yieldHandler(
  target: 'user' | 'tool' | 'worker' | 'parent',
): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open() {
      return []
    },
    close() {
      return []
    },
    selfClose(ctx) {
      return [...endTopProse(ctx.stack), emit({ _tag: 'TurnControl', target, termination: 'natural' }), observe]
    },
  }
}
