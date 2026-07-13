import type { TokenHandler, JsonOp, ArrayFrame, JsonParserContext, JsonToken } from '../types'
import type { ParsedValue } from '../../types'
import { popAndAttach } from './attach-child'
import { buildScalar } from './scalars'

function buildArr(frame: ArrayFrame): ParsedValue {
  return {
    _tag: 'array',
    items: [...frame.items],
    state: 'complete',
  }
}

export const arrayHandler: TokenHandler<ArrayFrame> = {
  handle(token: JsonToken, frame: ArrayFrame, ctx: JsonParserContext): JsonOp[] {
    switch (frame.phase) {
      case 'expectValue': {
        if (token._tag === 'arrayClose') {
          return popAndAttach(buildArr(frame), ctx.peekParent())
        }
        if (token._tag === 'objectOpen') {
          return [{ type: 'push', frame: { type: 'object', keys: [], values: [], phase: 'expectKey' } }]
        }
        if (token._tag === 'arrayOpen') {
          return [{ type: 'push', frame: { type: 'array', items: [], phase: 'expectValue' } }]
        }
        const val = buildScalar(token)
        if (val) {
          return [{ type: 'replace', frame: { ...frame, items: [...frame.items, val], phase: 'afterValue' } }]
        }
        return []
      }
      case 'afterValue': {
        if (token._tag === 'comma') {
          return [{ type: 'replace', frame: { ...frame, phase: 'expectValue' } }]
        }
        if (token._tag === 'arrayClose') {
          return popAndAttach(buildArr(frame), ctx.peekParent())
        }
        return []
      }
    }
  },
}
