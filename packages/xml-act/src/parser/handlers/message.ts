/**
 * Message frame handlers — open and close.
 *
 * messageOpenHandler: OpenHandler<ProseFrame, MessageFrame>
 * messageCloseHandler: CloseHandler<MessageFrame>
 *
 * Both are stateless objects. All effects returned as ParserOp[].
 */

import type { ProseFrame, MessageFrame } from '../types'
import type { OpenHandler, CloseHandler } from '../handler'
import { emitEvent } from '../ops'

export const messageOpenHandler: OpenHandler<ProseFrame, MessageFrame> = {
  open(attrs, _parent, ctx) {
    const id = ctx.generateId()
    const to = attrs.get('to') ?? null
    return [
      {
        type: 'push',
        frame: {
          type: 'message',
          id,
          to,
          content: '',
          pendingNewlines: 0,
        },
      },
      emitEvent({ _tag: 'MessageStart', id, to }),
    ]
  },
}

export const messageCloseHandler: CloseHandler<MessageFrame> = {
  close(top, _ctx) {
    return [
      emitEvent({ _tag: 'MessageEnd', id: top.id }),
      { type: 'pop' },
    ]
  },
}
