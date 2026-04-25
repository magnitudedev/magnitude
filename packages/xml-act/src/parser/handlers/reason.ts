/**
 * Reason/lens frame handlers — open and close.
 *
 * reasonOpenHandler: OpenHandler<ProseFrame, ReasonFrame>
 * reasonCloseHandler: CloseHandler<ReasonFrame>
 *
 * Both are stateless objects. The parent frame is passed at call time by bindOpen/bindClose.
 * All effects are returned as ParserOp[] — no direct emit calls.
 */

import type { ProseFrame, ReasonFrame } from '../types'
import type { OpenHandler, CloseHandler } from '../handler'
import { emitEvent, emitStructuralError, type ParserOp } from '../ops'
import { stripTrailingWhitespace } from '../content'

export const reasonOpenHandler: OpenHandler<ProseFrame, ReasonFrame> = {
  open(attrs, _parent, _ctx, tokenSpan) {
    const lensName = attrs.get('about') ?? 'analyze'
    return [
      {
        type: 'push',
        frame: {
          type: 'reason',
          openSpan: tokenSpan,
          name: lensName,
          content: '',
          hasContent: false,
          pendingNewlines: 0,
        },
      },
      emitEvent({ _tag: 'LensStart', name: lensName }),
    ]
  },
}

export const reasonCloseHandler: CloseHandler<ReasonFrame> = {
  close(top, _ctx, _tokenSpan) {
    const trimmed = stripTrailingWhitespace(top.content)
    return [
      emitEvent({ _tag: 'LensEnd', name: top.name, content: trimmed }),
      { type: 'pop' },
    ]
  },
}

/**
 * closeReasonAtEof — used by flush when a reason frame is unclosed at EOF.
 * Emits an UnclosedThink structural error before the normal LensEnd.
 */
export function closeReasonAtEof(top: ReasonFrame): ParserOp[] {
  const trimmed = stripTrailingWhitespace(top.content)
  return [
    emitStructuralError({ _tag: 'UnclosedThink', message: `Unclosed reason tag: ${top.name}`, primarySpan: top.openSpan }),
    emitEvent({ _tag: 'LensEnd', name: top.name, content: trimmed }),
    { type: 'pop' },
  ]
}
