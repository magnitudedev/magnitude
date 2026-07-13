import type { JsonToken, JsonFrame, JsonEvent, JsonParserContext, JsonParser, JsonTokenizer, RootFrame, PendingToken } from './types'
import type { ParsedValue } from '../types'
import { createStackMachine } from './machine'
import { resolveHandler } from './resolve'

export function createJsonParser(tokenizer: JsonTokenizer): JsonParser {
  const initialFrame: RootFrame = { type: 'root', value: undefined }

  const machine = createStackMachine<JsonFrame, JsonEvent>(
    initialFrame,
    (_event: JsonEvent) => {
      // Events are emitted for external consumers; child-to-parent attachment
      // is handled through the Op system via popAndAttach in handlers.
    },
  )

  const ctx: JsonParserContext = {
    tokenizer,
    peekParent(): JsonFrame | undefined {
      const s = machine.stack
      return s.length >= 2 ? s[s.length - 2] : undefined
    },
  }

  function feed(token: JsonToken): void {
    const top = machine.peek()
    if (!top) return
    const handler = resolveHandler(top)
    const ops = handler.handle(token, ctx)
    machine.apply(ops)
  }

  function buildPartial(): ParsedValue | undefined {
    const stack = machine.stack
    if (stack.length === 0) return undefined

    const pending = tokenizer.pending
    let pendingValue: ParsedValue | null = null
    if (pending !== null) {
      pendingValue = pendingToParsedValue(pending)
    }

    // Build from bottom to top
    // The bottom frame is root
    // We build the innermost (top) frame first, then wrap outward

    if (stack.length === 1) {
      const bottom = stack[0]
      if (bottom.type !== 'root') return undefined
      if (pendingValue !== null && bottom.value === undefined) return pendingValue
      return bottom.value
    }

    // Build from top of stack down
    let innerValue: ParsedValue | undefined = pendingValue ?? undefined

    for (let i = stack.length - 1; i >= 0; i--) {
      const frame = stack[i]
      switch (frame.type) {
        case 'root': {
          return innerValue ?? frame.value
        }
        case 'object': {
          const entries: [string, ParsedValue][] = []
          for (let ki = 0; ki < frame.keys.length; ki++) {
            if (ki < frame.values.length) {
              entries.push([frame.keys[ki], frame.values[ki]])
            } else if (innerValue !== undefined) {
              entries.push([frame.keys[ki], innerValue])
            }
          }

          // If we're expecting a key and have a pending value, it could be a partial key
          // but we don't add partial keys to entries

          innerValue = {
            _tag: 'object',
            entries,
            state: 'incomplete',
          }
          break
        }
        case 'array': {
          const items = [...frame.items]
          if (innerValue !== undefined) {
            items.push(innerValue)
          }
          innerValue = {
            _tag: 'array',
            items,
            state: 'incomplete',
          }
          break
        }
      }
    }

    return innerValue
  }

  function buildCurrentPath(): readonly string[] {
    const path: string[] = []
    const stack = machine.stack

    for (let i = 1; i < stack.length; i++) {
      const frame = stack[i]
      switch (frame.type) {
        case 'object': {
          if (frame.keys.length > 0) {
            path.push(frame.keys[frame.keys.length - 1])
          }
          break
        }
        case 'array': {
          if (frame.phase === 'afterValue') {
            path.push(String(frame.items.length - 1))
          } else if (frame.phase === 'expectValue') {
            // Show index when we have items OR when there's a pending token (in-progress value)
            if (frame.items.length > 0 || (i === stack.length - 1 && tokenizer.pending !== null)) {
              path.push(String(frame.items.length))
            }
          }
          break
        }
      }
    }

    return path
  }

  return {
    feed,
    end(): void {
      // Nothing special needed — tokenizer.end() will flush pending tokens
      // which will be fed through the normal path
    },
    get partial(): ParsedValue | undefined {
      return buildPartial()
    },
    get currentPath(): readonly string[] {
      return buildCurrentPath()
    },
  }
}

function pendingToParsedValue(pending: PendingToken): ParsedValue {
  switch (pending._tag) {
    case 'string':
      return { _tag: 'string', value: pending.content, state: 'incomplete' }
    case 'number':
      return { _tag: 'number', value: pending.content, state: 'incomplete' }
    case 'keyword':
      // Partial keyword — treat as incomplete string
      return { _tag: 'string', value: pending.content, state: 'incomplete' }
    case 'unquoted':
      return { _tag: 'string', value: pending.content, state: 'incomplete' }
  }
}
