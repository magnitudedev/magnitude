import type { TokenHandler, JsonOp, RootFrame, JsonParserContext, JsonToken } from '../types'
import { buildScalar } from './scalars'

export const rootHandler: TokenHandler<RootFrame> = {
  handle(token: JsonToken, frame: RootFrame, _ctx: JsonParserContext): JsonOp[] {
    switch (token._tag) {
      case 'objectOpen':
        return [{ type: 'push', frame: { type: 'object', keys: [], values: [], phase: 'expectKey' } }]
      case 'arrayOpen':
        return [{ type: 'push', frame: { type: 'array', items: [], phase: 'expectValue' } }]
      default: {
        const val = buildScalar(token)
        if (val) return [{ type: 'replace', frame: { type: 'root', value: val } }]
        return []
      }
    }
  },
}
