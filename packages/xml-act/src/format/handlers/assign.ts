import { emit, pop, push, replace } from '../ops'
import { endTopProse } from '../prose'
import { rawCloseTag, rawOpenTag } from '../raw'
import type { TagMap, TagHandler, XmlActEvent, XmlActFrame } from '../types'
import { findFrame } from '../types'

export function assignHandler(
  tags: TagMap,
): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      const task = findFrame(ctx.stack, 'task')
      if (!task) {
        return [
          ...endTopProse(ctx.stack),
          emit({
            _tag: 'ParseError',
            error: {
              _tag: 'InvalidAttributeValue',
              id: ctx.generateId(),
              tagName: 'assign',
              attribute: 'role',
              expected: 'non-empty string inside task frame',
              received: '',
              detail: 'Assign must be used inside a task',
            },
          }),
        ]
      }

      const role = ctx.attrs.get('role')?.trim() ?? ''
      if (!role) {
        return [
          ...endTopProse(ctx.stack),
          emit({
            _tag: 'ParseError',
            error: {
              _tag: 'InvalidAttributeValue',
              id: ctx.generateId(),
              tagName: 'assign',
              attribute: 'role',
              expected: 'non-empty string',
              received: '',
              detail: 'Assign role is required',
            },
          }),
        ]
      }

      return [
        ...endTopProse(ctx.stack),
        push({
          type: 'assign',
          taskId: task.id,
          role,
          body: '',
          depth: 0,
          pendingNewlines: 0,
          tags,
        }),
      ]
    },
    close(ctx) {
      const assign = findFrame(ctx.stack, 'assign')
      if (!assign) return []
      if (assign.depth > 0) {
        const raw = rawCloseTag(ctx.tagName)
        const prefix = assign.pendingNewlines > 0
          ? (assign.body.length > 0 ? '\n'.repeat(assign.pendingNewlines) : '')
          : ''
        const full = prefix + raw
        return [
          replace({
            ...assign,
            depth: assign.depth - 1,
            body: assign.body + full,
            pendingNewlines: 0,
          }),
        ]
      }

      return [
        emit({
          _tag: 'TaskAssign',
          taskId: assign.taskId,
          role: assign.role,
          body: assign.body.replace(/^\n+|\n+$/g, ''),
        }),
        pop,
      ]
    },
    selfClose(ctx) {
      const task = findFrame(ctx.stack, 'task')
      if (!task) {
        return [
          ...endTopProse(ctx.stack),
          emit({
            _tag: 'ParseError',
            error: {
              _tag: 'InvalidAttributeValue',
              id: ctx.generateId(),
              tagName: 'assign',
              attribute: 'role',
              expected: 'non-empty string inside task frame',
              received: '',
              detail: 'Assign must be used inside a task',
            },
          }),
        ]
      }

      const role = ctx.attrs.get('role')?.trim() ?? ''
      if (!role) {
        return [
          ...endTopProse(ctx.stack),
          emit({
            _tag: 'ParseError',
            error: {
              _tag: 'InvalidAttributeValue',
              id: ctx.generateId(),
              tagName: 'assign',
              attribute: 'role',
              expected: 'non-empty string',
              received: '',
              detail: 'Assign role is required',
            },
          }),
        ]
      }

      return [emit({ _tag: 'TaskAssign', taskId: task.id, role, body: '' })]
    },
  }
}
