/**
 * Yield self-close handler.
 *
 * makeYieldHandler — factory returning a SelfCloseHandler for yield_* tags.
 * endCurrentProse is handled by the parser loop before calling this handler.
 * The deferred yield target is set via a returned 'observe' op.
 */

import type { SelfCloseHandler } from '../handler'

const VALID_TARGETS = new Set(['user', 'invoke', 'worker', 'parent'])

export function makeYieldHandler(tagName: string): SelfCloseHandler {
  return {
    selfClose(_attrs, _ctx, _tokenSpan) {
      const suffix = tagName.replace(/^magnitude:yield_/, '')
      const target = (VALID_TARGETS.has(suffix) ? suffix : 'user') as 'user' | 'invoke' | 'worker' | 'parent'
      return [{ type: 'observe', target }]
    },
  }
}
