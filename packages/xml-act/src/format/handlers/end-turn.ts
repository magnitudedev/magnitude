import { emit, observe, pop, push, replace } from '../ops'
import { endTopProse } from '../prose'
import type { TagHandler, XmlActFrame, XmlActEvent, TagMap } from '../types'

/**
 * Inner handlers for <idle/> and <continue/> when inside an <end-turn> block.
 * These just record the decision on the parent end-turn frame — no emit, no done.
 */
function innerDecisionHandler(
  decision: 'continue' | 'idle',
): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open() { return [] },
    close() { return [] },
    selfClose(ctx) {
      const frame = ctx.stack[ctx.stack.length - 1]
      if (frame?.type === 'end-turn') {
        return [replace({ ...frame, decision })]
      }
      return []
    },
  }
}

/**
 * Handler for the <end-turn> structural block.
 * Contains <idle/> or <continue/> as children to record the turn control decision.
 * On close (or flush on EOF), emits TurnControl with the captured decision.
 *
 * Receives a tags map that must include 'end-turn' mapped to this handler,
 * so the parser can resolve </end-turn> while inside the end-turn frame.
 */
export function endTurnHandler(): TagHandler<XmlActFrame, XmlActEvent> {
  const innerTags = new Map<string, TagHandler<XmlActFrame, XmlActEvent>>([
    ['idle', innerDecisionHandler('idle')],
    ['continue', innerDecisionHandler('continue')],
  ])

  const handler: TagHandler<XmlActFrame, XmlActEvent> = {
    open(ctx) {
      return [
        ...endTopProse(ctx.stack),
        push({
          type: 'end-turn',
          decision: null,
          tags: innerTags,
        }),
      ]
    },
    close(ctx) {
      const frame = ctx.stack[ctx.stack.length - 1]
      const decision = (frame?.type === 'end-turn' ? frame.decision : null) ?? 'idle'
      return [pop, emit({ _tag: 'TurnControl', decision, termination: 'natural' }), observe]
    },
    selfClose(ctx) {
      return [...endTopProse(ctx.stack), emit({ _tag: 'TurnControl', decision: 'idle', termination: 'natural' }), observe]
    },
  }

  // Add self to inner tags so </end-turn> resolves while inside the frame
  innerTags.set('end-turn', handler)

  return handler
}
