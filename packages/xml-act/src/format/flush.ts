import { done, emit, pop } from './ops'
import type { Fx } from './ops'
import type { XmlActFrame } from './types'

export function xmlActFlush(stack: ReadonlyArray<XmlActFrame>): Fx[] {
  const ops: Fx[] = []
  let suppressToolBodyForId: string | null = null

  for (let i = stack.length - 1; i >= 0; i--) {
    const frame = stack[i]
    switch (frame.type) {
      case 'prose': {
        const trimmed = frame.body.replace(/[ \t\r\n]+$/g, '')
        if (trimmed.length > 0) {
          ops.push(emit({ _tag: 'ProseEnd', patternId: 'prose', content: trimmed, about: null }))
        }
        break
      }

      case 'think':
        ops.push(emit({ _tag: 'ParseError', error: { _tag: 'UnclosedThink' } }))
        if (frame.tag !== 'lens') {
          ops.push(emit({ _tag: 'ProseEnd', patternId: 'think', content: frame.body, about: frame.about }))
        }
        ops.push(pop)
        break
      case 'message':
        ops.push(emit({ _tag: 'MessageEnd', id: frame.id }))
        ops.push(pop)
        break

      case 'tool-body': {
        const suppressThisToolBody = suppressToolBodyForId === frame.id
        if (suppressThisToolBody) {
          suppressToolBodyForId = null
        } else {
          ops.push(
            emit({
              _tag: 'ParseError',
              error: {
                _tag: 'IncompleteTag',
                id: frame.id,
                tagName: frame.tag,
                detail: `Unclosed <${frame.tag}>`,
              },
            }),
          )
        }

        if (frame.body) {
          ops.push(emit({ _tag: 'ProseChunk', patternId: 'prose', text: frame.body }))
        }
        ops.push(pop)
        break
      }
      case 'child-body':
        ops.push(
          emit({
            _tag: 'ParseError',
            error: {
              _tag: 'UnclosedChild',
              id: frame.parentToolId,
              tagName: frame.parentTag,
              childTagName: frame.childTagName,
              detail: `Unclosed child <${frame.childTagName}> inside <${frame.parentTag}>`,
            },
          }),
        )
        suppressToolBodyForId = frame.parentToolId
        ops.push(pop)
        break
      // Note: yield tags are self-closing and emit TurnControl + done immediately
      // No frame is pushed for yield, so no flush case needed
    }
  }
  return ops
}
