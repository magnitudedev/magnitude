import { emit, pop, push, replace } from '../ops'
import { endTopProse } from '../prose'
import type { TagHandler, XmlActFrame, XmlActEvent } from '../types'
import { findFrame } from '../types'
import { rawCloseTag, rawOpenTag } from '../raw'

export function messageHandler(defaultDest: string): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      const existing = findFrame(ctx.stack, 'message')
      if (existing) {
        const raw = rawOpenTag('message', ctx.attrs)
        const prefix = existing.pendingNewlines > 0
          ? (existing.body.length > 0 ? '\n'.repeat(existing.pendingNewlines) : '')
          : ''
        const full = prefix + raw
        return [
          replace({
            ...existing,
            depth: existing.depth + 1,
            body: existing.body + full,
            pendingNewlines: 0,
          }),
          emit({ _tag: 'MessageChunk', id: existing.id, text: full }),
        ]
      }

      const id = ctx.generateId()
      const dest = ctx.attrs.get('to') ?? defaultDest
      const artifactsRaw = ctx.attrs.get('artifacts') ?? null
      return [
        ...endTopProse(ctx.stack),
        emit({ _tag: 'MessageStart', id, dest, artifactsRaw }),
        push({
          type: 'message',
          id,
          dest,
          artifactsRaw,
          body: '',
          depth: 0,
          pendingNewlines: 0,
        }),
      ]
    },
    close(ctx) {
      const msg = findFrame(ctx.stack, 'message')
      if (!msg) return []
      if (msg.depth > 0) {
        const raw = rawCloseTag(ctx.tagName)
        const prefix = msg.pendingNewlines > 0
          ? (msg.body.length > 0 ? '\n'.repeat(msg.pendingNewlines) : '')
          : ''
        const full = prefix + raw
        return [
          replace({
            ...msg,
            depth: msg.depth - 1,
            body: msg.body + full,
            pendingNewlines: 0,
          }),
          emit({ _tag: 'MessageChunk', id: msg.id, text: full }),
        ]
      }
      const content = msg.body.replace(/^\n+|\n+$/g, '')
      return [
        replace({ ...msg, body: content }),
        emit({ _tag: 'MessageEnd', id: msg.id }),
        pop,
      ]
    },
    selfClose(ctx) {
      const id = ctx.generateId()
      const dest = ctx.attrs.get('to') ?? defaultDest
      const artifactsRaw = ctx.attrs.get('artifacts') ?? null
      return [
        emit({ _tag: 'MessageStart', id, dest, artifactsRaw }),
        emit({ _tag: 'MessageEnd', id }),
      ]
    },
  }
}