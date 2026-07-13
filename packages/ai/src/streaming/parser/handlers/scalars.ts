import type { JsonToken } from '../types'
import type { ParsedValue } from '../../types'

/** Convert a scalar token to a ParsedValue. Returns null for structural tokens. */
export function buildScalar(token: JsonToken): ParsedValue | null {
  switch (token._tag) {
    case 'string': return { _tag: 'string', value: token.value, state: token.complete ? 'complete' : 'incomplete' }
    case 'number': return { _tag: 'number', value: token.value, state: token.complete ? 'complete' : 'incomplete' }
    case 'true': return { _tag: 'boolean', value: true, state: 'complete' }
    case 'false': return { _tag: 'boolean', value: false, state: 'complete' }
    case 'null': return { _tag: 'null', state: 'complete' }
    case 'unquotedString': return { _tag: 'string', value: token.value, state: token.complete ? 'complete' : 'incomplete' }
    default: return null
  }
}