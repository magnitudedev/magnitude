import { describe, expect, it } from 'bun:test'
import { xmlActContent } from '../format/content'
import { xmlActFlush } from '../format/flush'
import { containerHandler } from '../format/handlers/container'
import { finishHandler } from '../format/handlers/finish'
import { messageHandler } from '../format/handlers/message'
import { lensHandler, thinkHandler } from '../format/handlers/think'
import { childHandler, toolHandler } from '../format/handlers/tool'
import { turnControlHandler } from '../format/handlers/turn-control'
import type { TagMap, XmlActFrame } from '../format/types'

const id = () => 'abcdefgh1234'
const emptyTags: TagMap = new Map()

describe('format handlers', () => {
  it('container open emits container open', () => {
    const ops = containerHandler('actions', emptyTags).open({
      tagName: 'actions',
      attrs: new Map(),
      afterNewline: true,
      stack: [{ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }],
      generateId: id,
    })
    expect(ops.some((op) => op.type === 'emit' && (op as any).event._tag === 'ContainerOpen')).toBe(true)
  })

  it('think + lens lifecycle', () => {
    const betweenLensTags: TagMap = new Map()
    const insideLensTags: TagMap = new Map()
    const plainThinkTags: TagMap = new Map()
    const thinkOps = thinkHandler('lenses', betweenLensTags, plainThinkTags).open({
      tagName: 'lenses',
      attrs: new Map(),
      afterNewline: true,
      stack: [{ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }],
      generateId: id,
    })
    const think = (thinkOps.find((op) => op.type === 'push') as any).frame as XmlActFrame
    const lensOps = lensHandler(betweenLensTags, insideLensTags).open({
      tagName: 'lens',
      attrs: new Map([['name', 'task']]),
      afterNewline: true,
      stack: [{ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }, think],
      generateId: id,
    })
    expect(lensOps.some((op) => op.type === 'emit' && (op as any).event._tag === 'LensStart')).toBe(true)
  })

  it('message nested depth handled in open/close', () => {
    const msg = messageHandler('user', emptyTags)
    const opened = msg.open({
      tagName: 'message',
      attrs: new Map([['to', 'parent']]),
      afterNewline: true,
      stack: [{ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }],
      generateId: id,
    })
    const messageFrame = (opened.find((op) => op.type === 'push') as any).frame as XmlActFrame
    const nested = msg.open({
      tagName: 'message',
      attrs: new Map(),
      afterNewline: true,
      stack: [{ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }, messageFrame],
      generateId: id,
    })
    expect(nested.some((op) => op.type === 'replace')).toBe(true)
  })

  it('tool open emits TagOpened', () => {
    const ops = toolHandler('shell', new Set(), undefined, new Map()).open({
      tagName: 'shell',
      attrs: new Map([['cmd', 'ls']]),
      afterNewline: true,
      stack: [{ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }],
      generateId: id,
    })
    expect(ops.some((op) => op.type === 'emit' && (op as any).event._tag === 'TagOpened')).toBe(true)
  })

  it('child open emits ChildOpened', () => {
    const parent: XmlActFrame = {
      type: 'tool-body',
      tag: 'tool',
      id: 't1',
      attrs: new Map(),
      body: '',
      children: [],
      childCounts: new Map(),
      childTags: new Set(['arg']),
      schema: undefined,
      tags: emptyTags,
    }
    const ops = childHandler().open({
      tagName: 'arg',
      attrs: new Map([['name', 'x']]),
      afterNewline: true,
      stack: [{ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }, parent],
      generateId: id,
    })
    expect(ops.some((op) => op.type === 'emit' && (op as any).event._tag === 'ChildOpened')).toBe(true)
  })

  it('turn control self-close emits TurnControl + done', () => {
    const ops = turnControlHandler('continue').selfClose({
      tagName: 'next',
      attrs: new Map(),
      afterNewline: true,
      stack: [{ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }],
      generateId: id,
    })
    expect(ops.some((op) => op.type === 'emit' && (op as any).event._tag === 'TurnControl')).toBe(true)
    expect(ops.some((op) => op.type === 'done')).toBe(true)
  })

  it('finish self-close emits parse error only + done', () => {
    const ops = finishHandler().selfClose({
      tagName: 'finish',
      attrs: new Map(),
      afterNewline: true,
      stack: [{ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }],
      generateId: id,
    })
    expect(ops.some((op) => op.type === 'emit' && (op as any).event._tag === 'ParseError')).toBe(true)
    expect(ops.some((op) => op.type === 'emit' && (op as any).event._tag === 'TurnControl')).toBe(false)
    expect(ops.some((op) => op.type === 'done')).toBe(true)
  })

  it('content routing for prose + message + tool', () => {
    const proseOps = xmlActContent({ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }, 'hello')
    expect((proseOps[0] as any).event._tag).toBe('ProseChunk')

    const msgOps = xmlActContent(
      { type: 'message', id: 'm1', dest: 'user', artifactsRaw: null, body: '', depth: 0, pendingNewlines: 0, tags: emptyTags },
      'hey',
    )
    expect(msgOps.some((op) => op.type === 'emit' && (op as any).event._tag === 'MessageChunk')).toBe(true)

    const toolOps = xmlActContent(
      { type: 'tool-body', tag: 'shell', id: 't1', attrs: new Map(), body: '', children: [], childCounts: new Map(), childTags: new Set(), schema: undefined, tags: emptyTags },
      'ls',
    )
    expect(toolOps.some((op) => op.type === 'emit' && (op as any).event._tag === 'BodyChunk')).toBe(true)
  })

  it('flush emits unclosed container error', () => {
    const ops = xmlActFlush([{ type: 'prose', body: '', pendingNewlines: 0, tags: emptyTags }, { type: 'container', tag: 'actions', depth: 0, tags: emptyTags }])
    expect(ops.some((op) => op.type === 'emit' && (op as any).event._tag === 'ParseError')).toBe(true)
  })
})