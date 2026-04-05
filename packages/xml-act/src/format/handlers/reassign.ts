import { emit, pop, push, replace } from '../ops'
import { endTopProse } from '../prose'
import { rawCloseTag, rawOpenTag } from '../raw'
import type { TagMap, TagHandler, XmlActEvent, XmlActFrame } from '../types'
import { findFrame } from '../types'

export function reassignHandler(
  _tags: TagMap,
): TagHandler<XmlActFrame, XmlActEvent> {
  const mutableTags = new Map<string, TagHandler<XmlActFrame, XmlActEvent>>()

  const handler: TagHandler<XmlActFrame, XmlActEvent> = {
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
              tagName: 'reassign',
              attribute: 'role',
              expected: 'non-empty string inside task frame',
              received: '',
              detail: 'Reassign must be used inside a task',
            },
          }),
        ]
      }

      const role = ctx.attrs.get('role')?.trim()
      if (!role || role.length === 0) {
        return [
          ...endTopProse(ctx.stack),
          emit({
            _tag: 'ParseError',
            error: {
              _tag: 'InvalidAttributeValue',
              id: ctx.generateId(),
              tagName: 'reassign',
              attribute: 'role',
              expected: 'non-empty string',
              received: '',
              detail: 'Reassign role is required',
            },
          }),
        ]
      }

      const existingReassign = findFrame(ctx.stack, 'reassign')
      if (existingReassign) {
        const raw = rawOpenTag('reassign', ctx.attrs)
        const prefix = existingReassign.pendingNewlines > 0
          ? (existingReassign.body.length > 0 ? '\n'.repeat(existingReassign.pendingNewlines) : '')
          : ''
        return [
          replace({
            ...existingReassign,
            depth: existingReassign.depth + 1,
            body: existingReassign.body + prefix + raw,
            pendingNewlines: 0,
          }),
        ]
      }

      return [
        ...endTopProse(ctx.stack),
        push({
          type: 'reassign',
          taskId: task.id,
          role,
          body: '',
          depth: 0,
          pendingNewlines: 0,
          tags: reassignBodyTags,
        }),
      ]
    },
    close(ctx) {
      const reassign = findFrame(ctx.stack, 'reassign')
      if (!reassign) return []
      if (reassign.depth > 0) {
        const raw = rawCloseTag(ctx.tagName)
        const prefix = reassign.pendingNewlines > 0
          ? (reassign.body.length > 0 ? '\n'.repeat(reassign.pendingNewlines) : '')
          : ''
        return [
          replace({
            ...reassign,
            depth: reassign.depth - 1,
            body: reassign.body + (prefix + raw),
            pendingNewlines: 0,
          }),
        ]
      }

      return [
        emit({
          _tag: 'TaskReassign',
          taskId: reassign.taskId,
          role: reassign.role,
          body: reassign.body.replace(/^\n+|\n+$/g, ''),
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
              tagName: 'reassign',
              attribute: 'role',
              expected: 'non-empty string inside task frame',
              received: '',
              detail: 'Reassign must be used inside a task',
            },
          }),
        ]
      }

      const role = ctx.attrs.get('role')?.trim()
      if (!role || role.length === 0) {
        return [
          ...endTopProse(ctx.stack),
          emit({
            _tag: 'ParseError',
            error: {
              _tag: 'InvalidAttributeValue',
              id: ctx.generateId(),
              tagName: 'reassign',
              attribute: 'role',
              expected: 'non-empty string',
              received: '',
              detail: 'Reassign role is required',
            },
          }),
        ]
      }

      const existingReassign = findFrame(ctx.stack, 'reassign')
      if (existingReassign) {
        const raw = rawOpenTag('reassign', ctx.attrs).replace(/>$/, '/>')
        const prefix = existingReassign.pendingNewlines > 0
          ? (existingReassign.body.length > 0 ? '\n'.repeat(existingReassign.pendingNewlines) : '')
          : ''
        return [replace({ ...existingReassign, body: existingReassign.body + prefix + raw, pendingNewlines: 0 })]
      }

      return [emit({ _tag: 'TaskReassign', taskId: task.id, role, body: '' })]
    },
  }

  mutableTags.set('reassign', handler)
  const reassignBodyTags: TagMap = mutableTags

  return handler
}
