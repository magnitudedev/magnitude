import { emit, replace } from './ops'
import { completeChild } from './handlers/tool'
import type { Fx } from './ops'
import type { XmlActFrame } from './types'

export function xmlActContent(frame: XmlActFrame, text: string): Fx[] {
  switch (frame.type) {
    case 'prose': {
      const ops: Fx[] = []
      let next = { ...frame }

      // Split into lines, preserving newline boundaries
      const lines = text.split('\n')
      for (let idx = 0; idx < lines.length; idx++) {
        // Between segments: each split boundary represents a newline (except before first)
        if (idx > 0) {
          next = { ...next, pendingNewlines: next.pendingNewlines + 1 }
        }
        const line = lines[idx]
        if (line.length > 0) {
          // Flush any pending newlines before non-empty content
          if (next.pendingNewlines > 0) {
            const nl = '\n'.repeat(next.pendingNewlines)
            ops.push(emit({ _tag: 'ProseChunk', patternId: 'prose', text: nl }))
            next = { ...next, body: next.body + nl, pendingNewlines: 0 }
          }
          ops.push(emit({ _tag: 'ProseChunk', patternId: 'prose', text: line }))
          next = { ...next, body: next.body + line }
        }
      }
      return [...ops, replace(next)]
    }
    case 'think': {
      if (frame.activeLens) {
        return [
          replace({ ...frame, activeLens: { ...frame.activeLens, body: frame.activeLens.body + text } }),
          emit({ _tag: 'LensChunk', text }),
        ]
      }
      if (frame.isLenses && !frame.activeLens) {
        // In lenses mode outside an active lens, inter-lens text is structural whitespace — accumulate but don't emit
        return [replace({ ...frame, body: frame.body + text })]
      }
      return [replace({ ...frame, body: frame.body + text }), emit({ _tag: 'ProseChunk', patternId: 'think', text })]
    }
    case 'message': {
      const pending = frame.pendingNewlines
      if (/^\n+$/.test(text)) {
        return [replace({ ...frame, pendingNewlines: pending + text.length })]
      }

      let segment = frame.body.length === 0 ? text.replace(/^\n+/, '') : text
      const trailing = segment.match(/\n+$/)?.[0] ?? ''
      if (trailing.length > 0) segment = segment.slice(0, -trailing.length)

      const prefix = pending > 0 && frame.body.length > 0 ? '\n'.repeat(pending) : ''
      const full = prefix + segment
      const nextBody = full.length > 0 ? frame.body + full : frame.body

      const ops: Fx[] = [replace({ ...frame, body: nextBody, pendingNewlines: trailing.length })]
      if (full.length > 0) {
        ops.push(emit({ _tag: 'MessageChunk', id: frame.id, text: full }))
      }
      return ops
    }
    case 'assign': {
      const pending = frame.pendingNewlines
      if (/^\n+$/.test(text)) {
        return [replace({ ...frame, pendingNewlines: pending + text.length })]
      }

      let segment = frame.body.length === 0 ? text.replace(/^\n+/, '') : text
      const trailing = segment.match(/\n+$/)?.[0] ?? ''
      if (trailing.length > 0) segment = segment.slice(0, -trailing.length)

      const prefix = pending > 0 && frame.body.length > 0 ? '\n'.repeat(pending) : ''
      const full = prefix + segment
      return [replace({ ...frame, body: frame.body + full, pendingNewlines: trailing.length })]
    }
    case 'tool-body':
      return [replace({ ...frame, body: frame.body + text }), emit({ _tag: 'BodyChunk', toolCallId: frame.id, text })]
    case 'child-body':
      return [
        replace({ ...frame, body: frame.body + text }),
        emit({
          _tag: 'ChildBodyChunk',
          parentToolCallId: frame.parentToolId,
          childTagName: frame.childTagName,
          childIndex: frame.childIndex,
          text,
        }),
      ]
    case 'body-capture':
      return [replace({ ...frame, body: frame.body + text })]
    case 'task':
      return []
  }
}

export function maybeCompleteChildOnContentBoundary(
  stack: ReadonlyArray<XmlActFrame>,
): Fx[] {
  const top = stack[stack.length - 1]
  const parent = stack[stack.length - 2]
  if (top?.type === 'child-body' && parent?.type === 'tool-body') {
    return [...completeChild(parent, top)]
  }
  return []
}
