import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

function parse(xml: string, knownTags = new Set(['read', 'grep'])): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, new Map())
  return [...parser.processChunk(xml), ...parser.flush()]
}

describe('task format parsing', () => {
  it('parses task open/close', () => {
    const events = parse('<task id="t1" type="scan" title="Scan"><read path="x" /></task><yield/>')
    expect(events.some(e => e._tag === 'TaskOpen')).toBe(true)
    expect(events.some(e => e._tag === 'TaskClose')).toBe(true)
  })

  it('parses self-closing task update when create attrs are absent', () => {
    const events = parse('<task id="t1" status="completed" /><yield/>')
    const update = events.find((e): e is Extract<ParseEvent, { _tag: 'TaskUpdate' }> => e._tag === 'TaskUpdate')
    expect(update).toBeDefined()
    expect(update?.id).toBe('t1')
    expect(update?.status).toBe('completed')
  })

  it('parses self-closing task create when type and title are present', () => {
    const events = parse('<task id="t1" type="review" title="Review changes" parent="p1" /><yield/>')
    const open = events.find((e): e is Extract<ParseEvent, { _tag: 'TaskOpen' }> => e._tag === 'TaskOpen')
    const close = events.find((e): e is Extract<ParseEvent, { _tag: 'TaskClose' }> => e._tag === 'TaskClose')

    expect(open).toBeDefined()
    expect(open?.id).toBe('t1')
    expect(open?.taskType).toBe('review')
    expect(open?.title).toBe('Review changes')
    expect(open?.parent).toBe('p1')
    expect(close).toBeDefined()
    expect(close?.id).toBe('t1')
  })

  it('assign valid only inside task', () => {
    const ok = parse('<task id="t1"><assign role="builder">do it</assign></task><yield/>')
    expect(ok.some(e => e._tag === 'TaskAssign')).toBe(true)

    const bad = parse('<assign role="builder">nope</assign><yield/>')
    expect(bad.some(e => e._tag === 'ParseError')).toBe(true)
  })

  it('nested task gets implicit parent', () => {
    const events = parse('<task id="p"><task id="c"></task></task><yield/>')
    const opens = events.filter((e): e is Extract<ParseEvent, { _tag: 'TaskOpen' }> => e._tag === 'TaskOpen')
    expect(opens).toHaveLength(2)
    expect(opens[1]?.parent).toBe('p')
  })

  it('message scope metadata top-level vs task', () => {
    const events = parse('<message>top</message><task id="t1"><message>inner</message></task><yield/>')
    const starts = events.filter((e): e is Extract<ParseEvent, { _tag: 'MessageStart' }> => e._tag === 'MessageStart')
    expect(starts[0]?.scope).toBe('top-level')
    expect(starts[1]?.scope).toBe('task')
    expect(starts[1]?.taskId).toBe('t1')
  })

  it('turn control auto-closes all open tasks', () => {
    const events = parse('<task id="a"><task id="b"><yield/>')
    const closes = events.filter((e): e is Extract<ParseEvent, { _tag: 'TaskClose' }> => e._tag === 'TaskClose')
    expect(closes.map(c => c.id)).toEqual(['b', 'a'])
  })

  it('flush emits UnclosedTask', () => {
    const events = parse('<task id="a">')
    const err = events.find((e): e is Extract<ParseEvent, { _tag: 'ParseError' }> => e._tag === 'ParseError')
    expect(err?.error._tag).toBe('UnclosedTask')
  })
})
