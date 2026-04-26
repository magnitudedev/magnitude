/**
 * Think/lens frame handlers — open and close.
 *
 * thinkOpenHandler: OpenHandler<ProseFrame, ThinkFrame>
 * thinkCloseHandler: CloseHandler<ThinkFrame>
 *
 * Both are stateless objects. The parent frame is passed at call time by bindOpen/bindClose.
 * All effects are returned as ParserOp[] — no direct emit calls.
 */

import type { ProseFrame, ThinkFrame } from '../types'
import type { OpenHandler, CloseHandler } from '../handler'
import { emitEvent, emitStructuralError, type ParserOp } from '../ops'
import { stripTrailingWhitespace } from '../content'

export const thinkOpenHandler: OpenHandler<ProseFrame, ThinkFrame> = {
  open(attrs, _parent, _ctx, tokenSpan) {
    const lensName = attrs.get('about') ?? 'analyze'
    return [
      {
        type: 'push',
        frame: {
          type: 'think',
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

export const thinkCloseHandler: CloseHandler<ThinkFrame> = {
  close(top, _ctx, _tokenSpan) {
    const trimmed = stripTrailingWhitespace(top.content)
    return [
      emitEvent({ _tag: 'LensEnd', name: top.name, content: trimmed }),
      { type: 'pop' },
    ]
  },
}

/**
 * closeThinkAtEof — used by flush when a think frame is unclosed at EOF.
 * Emits an UnclosedThink structural error before the normal LensEnd.
 */
export function closeThinkAtEof(top: ThinkFrame): ParserOp[] {
  const trimmed = stripTrailingWhitespace(top.content)
  return [
    emitStructuralError({ _tag: 'UnclosedThink', message: `Unclosed think tag: ${top.name}`, primarySpan: top.openSpan }),
    emitEvent({ _tag: 'LensEnd', name: top.name, content: trimmed }),
    { type: 'pop' },
  ]
}
