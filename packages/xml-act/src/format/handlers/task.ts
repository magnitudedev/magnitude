import { emit, pop, push } from '../ops'
import { endTopProse } from '../prose'
import type { TagMap, TagHandler, XmlActEvent, XmlActFrame } from '../types'

function findEnclosingTask(stack: ReadonlyArray<XmlActFrame>) {
  for (let i = stack.length - 1; i >= 0; i--) {
    const frame = stack[i]
    if (frame.type === 'task') return frame
  }
  return undefined
}

function missingIdError(id: string): XmlActEvent {
  return {
    _tag: 'ParseError',
    error: {
      _tag: 'InvalidAttributeValue',
      id,
      tagName: 'task',
      attribute: 'id',
      expected: 'non-empty string',
      received: '',
      detail: 'Task id is required',
    },
  }
}

export function taskHandler(
  tags: TagMap,
): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      const id = ctx.attrs.get('id')?.trim() ?? ''
      if (!id) {
        return [...endTopProse(ctx.stack), emit(missingIdError(ctx.generateId()))]
      }

      const parentTask = findEnclosingTask(ctx.stack)
      const explicitParent = ctx.attrs.get('parent') ?? null
      const parent = explicitParent ?? parentTask?.id ?? null

      return [
        ...endTopProse(ctx.stack),
        emit({
          _tag: 'TaskOpen',
          id,
          taskType: ctx.attrs.get('type') ?? null,
          title: ctx.attrs.get('title') ?? null,
          parent,
          after: ctx.attrs.get('after') ?? null,
          status: ctx.attrs.get('status') ?? null,
          explicitParent,
        }),
        push({
          type: 'task',
          id,
          taskType: ctx.attrs.get('type') ?? null,
          title: ctx.attrs.get('title') ?? null,
          parent,
          explicitParent,
          after: ctx.attrs.get('after') ?? null,
          status: ctx.attrs.get('status') ?? null,
          tags,
        }),
      ]
    },
    close(ctx) {
      const top = ctx.stack[ctx.stack.length - 1]
      if (!top || top.type !== 'task') return []
      return [emit({ _tag: 'TaskClose', id: top.id }), pop]
    },
    selfClose(ctx) {
      const id = ctx.attrs.get('id')?.trim() ?? ''
      if (!id) {
        return [...endTopProse(ctx.stack), emit(missingIdError(ctx.generateId()))]
      }

      const parentTask = findEnclosingTask(ctx.stack)
      const explicitParent = ctx.attrs.get('parent') ?? null
      const parent = explicitParent ?? parentTask?.id ?? null

      return [
        ...endTopProse(ctx.stack),
        emit({
          _tag: 'TaskUpdate',
          id,
          taskType: ctx.attrs.get('type') ?? null,
          title: ctx.attrs.get('title') ?? null,
          parent,
          after: ctx.attrs.get('after') ?? null,
          status: ctx.attrs.get('status') ?? null,
          explicitParent,
        }),
      ]
    },
  }
}
