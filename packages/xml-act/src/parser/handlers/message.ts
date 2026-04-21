/**
 * Message frame handlers — open and close.
 */

import type { TurnEngineEvent } from '../../types'
import type { Op } from '../../machine'
import type { Frame, MessageFrame } from '../types'
import { MESSAGE_VALID_TAGS } from '../types'

export function openMessage(
  variant: string | undefined,
  generateId: () => string,
  endCurrentProse: () => void,
  apply: (ops: Op<Frame, TurnEngineEvent>[]) => void,
): void {
  const id = generateId()
  const to = variant ?? null
  endCurrentProse()
  apply([
    {
      type: 'push',
      frame: {
        type: 'message',
        id,
        to,
        content: '',
        pendingNewlines: 0,
        validTags: MESSAGE_VALID_TAGS,
      },
    },
    { type: 'emit', event: { _tag: 'MessageStart', id, to } },
  ])
}

export function closeMessage(
  top: MessageFrame,
  apply: (ops: Op<Frame, TurnEngineEvent>[]) => void,
): void {
  apply([
    { type: 'emit', event: { _tag: 'MessageEnd', id: top.id } },
    { type: 'pop' },
  ])
}
