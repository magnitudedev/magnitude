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
