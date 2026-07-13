import type { JsonFrame, JsonOp } from '../types'
import type { ParsedValue } from '../../types'

/**
 * Given a parent frame and a child value, returns ops to pop the current frame
 * and replace the parent with the child attached, plus emit the completed value.
 */
export function popAndAttach(child: ParsedValue, parent: JsonFrame | undefined): JsonOp[] {
  const popOp: JsonOp = { type: 'pop' }
  const emitOp: JsonOp = { type: 'emit', event: { _tag: 'value', value: child } }

  if (!parent) {
    return [popOp, emitOp]
  }

  switch (parent.type) {
    case 'root': {
      const updated: JsonFrame = { type: 'root', value: child }
      return [popOp, { type: 'replace', frame: updated }, emitOp]
    }
    case 'object': {
      if (parent.phase !== 'expectValue') {
        throw new Error(`Invariant violation: popAndAttach called with object parent in phase "${parent.phase}"`)
      }
      const updated: JsonFrame = {
        type: 'object',
        keys: parent.keys,
        values: [...parent.values, child],
        phase: 'afterValue',
      }
      return [popOp, { type: 'replace', frame: updated }, emitOp]
    }
    case 'array': {
      if (parent.phase !== 'expectValue') {
        throw new Error(`Invariant violation: popAndAttach called with array parent in phase "${parent.phase}"`)
      }
      const updated: JsonFrame = {
        type: 'array',
        items: [...parent.items, child],
        phase: 'afterValue',
      }
      return [popOp, { type: 'replace', frame: updated }, emitOp]
    }
  }
}
