/**
 * Think/lens frame handlers — open and close.
 */

import type { TurnEngineEvent } from '../../types'
import type { Op } from '../../machine'
import type { Frame, ThinkFrame } from '../types'
import { THINK_VALID_TAGS } from '../types'
import { stripTrailingWhitespace } from '../content'

export function openThink(
  variant: string | undefined,
  endCurrentProse: () => void,
  apply: (ops: Op<Frame, TurnEngineEvent>[]) => void,
): void {
  const lensName = variant ?? 'analyze'
  endCurrentProse()
  apply([
    {
      type: 'push',
      frame: {
        type: 'think',
        name: lensName,
        content: '',
        hasContent: false,
        pendingNewlines: 0,
        validTags: THINK_VALID_TAGS,
      },
    },
    { type: 'emit', event: { _tag: 'LensStart', name: lensName } },
  ])
}

export function closeThink(
  top: ThinkFrame,
  emitStructuralError: (error: { _tag: 'UnclosedThink'; message: string }) => void,
  apply: (ops: Op<Frame, TurnEngineEvent>[]) => void,
  atEof = false,
): void {
  const trimmed = stripTrailingWhitespace(top.content)
  if (atEof) {
    emitStructuralError({ _tag: 'UnclosedThink', message: `Unclosed think tag: ${top.name}` })
  }
  apply([
    { type: 'emit', event: { _tag: 'LensEnd', name: top.name, content: trimmed } },
    { type: 'pop' },
  ])
}
