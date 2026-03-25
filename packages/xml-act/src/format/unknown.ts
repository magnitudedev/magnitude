import { emit, replace } from './ops'
import { appendTopProse } from './prose'
import type { Fx } from './ops'
import type { XmlActFrame } from './types'

export function xmlActUnknownOpen(
  tagName: string,
  attrs: ReadonlyMap<string, string>,
  _afterNewline: boolean,
  stack: ReadonlyArray<XmlActFrame>,
  raw: string,
): Fx[] {
  const top = stack[stack.length - 1]
  if (!top) return appendTopProse(stack, raw)

  if (top.type === 'tool-body') {
    return [replace({ ...top, body: top.body + raw }), emit({ _tag: 'BodyChunk', toolCallId: top.id, text: raw })]
  }
  if (top.type === 'child-body') {
    return [
      replace({ ...top, body: top.body + raw }),
      emit({ _tag: 'ChildBodyChunk', parentToolCallId: top.parentToolId, childTagName: top.childTagName, childIndex: top.childIndex, text: raw }),
    ]
  }
  if (top.type === 'message') {
    return [
      replace({ ...top, body: top.body + raw }),
      emit({ _tag: 'MessageChunk', id: top.id, text: raw }),

    ]
  }
  if (top.type === 'think') {
    if (top.activeLens) {
      return [
        replace({ ...top, activeLens: { ...top.activeLens, body: top.activeLens.body + raw } }),
        emit({ _tag: 'LensChunk', text: raw }),
      ]
    }
    return [replace({ ...top, body: top.body + raw }), emit({ _tag: 'ProseChunk', patternId: 'think', text: raw })]
  }

  return appendTopProse(stack, raw)
}

export function xmlActUnknownClose(
  _tagName: string,
  stack: ReadonlyArray<XmlActFrame>,
  raw: string,
): Fx[] {
  const top = stack[stack.length - 1]
  if (!top) return appendTopProse(stack, raw)

  if (top.type === 'tool-body') {
    return [replace({ ...top, body: top.body + raw }), emit({ _tag: 'BodyChunk', toolCallId: top.id, text: raw })]
  }
  if (top.type === 'child-body') {
    return [
      replace({ ...top, body: top.body + raw }),
      emit({ _tag: 'ChildBodyChunk', parentToolCallId: top.parentToolId, childTagName: top.childTagName, childIndex: top.childIndex, text: raw }),
    ]
  }
  if (top.type === 'message') {
    return [
      replace({ ...top, body: top.body + raw }),
      emit({ _tag: 'MessageChunk', id: top.id, text: raw }),

    ]
  }
  if (top.type === 'think') {
    if (top.activeLens) {
      return [
        replace({ ...top, activeLens: { ...top.activeLens, body: top.activeLens.body + raw } }),
        emit({ _tag: 'LensChunk', text: raw }),
      ]
    }
    return [replace({ ...top, body: top.body + raw }), emit({ _tag: 'ProseChunk', patternId: 'think', text: raw })]
  }

  return appendTopProse(stack, raw)
}