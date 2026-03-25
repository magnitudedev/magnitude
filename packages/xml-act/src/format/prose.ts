import { emit, replace } from './ops'
import type { Fx } from './ops'
import type { XmlActFrame } from './types'

type ProseFrame = Extract<XmlActFrame, { type: 'prose' }>

export function appendTopProse(stack: ReadonlyArray<XmlActFrame>, text: string): Fx[] {
  const top = stack[stack.length - 1]
  if (!top || top.type !== 'prose') {
    return [emit({ _tag: 'ProseChunk', patternId: 'prose', text })]
  }
  const prose: ProseFrame = top

  // At the start of a new prose section, skip leading whitespace
  if (prose.body === '' && prose.pendingNewlines === 0) {
    const stripped = text.replace(/^[ \t\r\n]+/, '')
    if (stripped.length === 0) return []
    return [
      replace({ type: 'prose', body: stripped, pendingNewlines: 0 }),
      emit({ _tag: 'ProseChunk', patternId: 'prose', text: stripped }),
    ]
  }

  // If body is still empty but we have pending newlines, check if adding
  // this text would result in non-whitespace content
  if (prose.body === '' && prose.pendingNewlines > 0) {
    const stripped = text.replace(/^[ \t\r\n]+/, '')
    if (stripped.length === 0) {
      // Still all whitespace — keep deferring
      return [replace({ type: 'prose', body: '', pendingNewlines: prose.pendingNewlines + text.split('\n').length - 1 })]
    }
    // Has real content — emit with prefix
    const prefix = '\n'.repeat(prose.pendingNewlines)
    return [
      replace({ type: 'prose', body: stripped, pendingNewlines: 0 }),
      emit({ _tag: 'ProseChunk', patternId: 'prose', text: stripped }),
    ]
  }

  const prefix = prose.pendingNewlines > 0 ? '\n'.repeat(prose.pendingNewlines) : ''
  const full = prefix + text
  return [
    replace({ type: 'prose', body: prose.body + full, pendingNewlines: 0 }),
    ...(prefix ? [emit({ _tag: 'ProseChunk', patternId: 'prose', text: prefix })] : []),
    emit({ _tag: 'ProseChunk', patternId: 'prose', text }),
  ]
}

export function endTopProse(stack: ReadonlyArray<XmlActFrame>): Fx[] {
  const top = stack[stack.length - 1]
  if (!top || top.type !== 'prose') return []
  const prose: ProseFrame = top
  const trimmed = prose.body.replace(/[ \t\r\n]+$/g, '')
  if (trimmed.length === 0) return [replace({ type: 'prose', body: '', pendingNewlines: 0 })]
  return [
    emit({ _tag: 'ProseEnd', patternId: 'prose', content: trimmed, about: null }),
    replace({ type: 'prose', body: '', pendingNewlines: 0 }),
  ]
}
