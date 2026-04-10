import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import type { XmlTagBinding } from '../types'
import { createStreamingXmlParser } from '../parser'
import { validateBinding } from '../execution/binding-validator'
import type { ParseEvent } from '../format/types'

/**
 * Repro for the attr/body swap bug.
 *
 * The bug comes from the actual runtime/binding path:
 * - validateBinding synthesizes child schemas for attribute names
 * - the runtime uses the resulting schema children to build childTagMap
 * - the parser then treats angle-bracket prose matching those attr names as child tags
 *
 * These cases isolate the bug:
 * - unknown / unrelated tag names remain raw BodyChunk text
 * - only tag names introduced through the attr/body swap path trigger child parsing
 */

const knownTags = new Set(['spawn-worker'])

const SpawnWorkerInput = Schema.Struct({
  workerId: Schema.String,
  role: Schema.String,
  message: Schema.String,
})

const binding: XmlTagBinding = {
  tag: 'spawn-worker',
  attributes: [
    { attr: 'id', field: 'workerId' },
    { attr: 'role', field: 'role' },
  ],
  body: 'message',
}

function parse(xml: string, overrideBinding?: XmlTagBinding): ParseEvent[] {
  const schema = validateBinding('spawn-worker', overrideBinding ?? binding, SpawnWorkerInput.ast)
  const parser = createStreamingXmlParser(
    knownTags,
    new Map([['spawn-worker', new Set(schema.children.keys())]]),
    new Map([['spawn-worker', schema]]),
  )
  return [...parser.processChunk(xml), ...parser.flush()]
}

function body(events: ParseEvent[]): string {
  return events
    .filter((event): event is Extract<ParseEvent, { _tag: 'BodyChunk' }> => event._tag === 'BodyChunk')
    .map((event) => event.text)
    .join('')
}

function childOpens(events: ParseEvent[]) {
  return events.filter((event): event is Extract<ParseEvent, { _tag: 'ChildOpened' }> => event._tag === 'ChildOpened')
}

function parseErrors(events: ParseEvent[]) {
  return events.filter((event): event is Extract<ParseEvent, { _tag: 'ParseError' }> => event._tag === 'ParseError')
}

describe('repro: attr/body swap parsing bug', () => {
  it('unknown tag in body with empty childTags passes through as BodyChunk', () => {
    const xml = '<spawn-worker>path `~/.magnitude/sessions/<placeholder>/events.jsonl`</spawn-worker>'

    const events = parse(xml, { tag: 'spawn-worker', body: 'message' })

    expect(body(events)).toContain('`~/.magnitude/sessions/<placeholder>/events.jsonl`')
    expect(childOpens(events)).toHaveLength(0)
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('unrelated tag in body alongside attr-derived childTags passes through', () => {
    const xml = '<spawn-worker>body with <div> marker in prose</spawn-worker>'

    const events = parse(xml)

    expect(body(events)).toContain('body with <div> marker in prose')
    expect(childOpens(events)).toHaveLength(0)
    expect(parseErrors(events)).toHaveLength(0)
  })

  it("attr-matching tag 'id' in body triggers ChildOpened instead of BodyChunk", () => {
    const xml = '<spawn-worker>path `~/.magnitude/sessions/<id>/events.jsonl`</spawn-worker>'

    const events = parse(xml, {
      tag: 'spawn-worker',
      attributes: [{ attr: 'id', field: 'workerId' }],
      body: 'message',
    })

    expect(body(events)).toContain('`~/.magnitude/sessions/<id>/events.jsonl`')
    expect(childOpens(events)).toHaveLength(0)
    expect(parseErrors(events)).toHaveLength(0)
  })

  it("attr-matching tag 'role' in body triggers same bug", () => {
    const xml = '<spawn-worker>choose <role> carefully</spawn-worker>'

    const events = parse(xml)

    expect(body(events)).toContain('choose <role> carefully')
    expect(childOpens(events)).toHaveLength(0)
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('full spawn-worker repro with id and role attrs', () => {
    const xml = [
      '<spawn-worker id="compiler-cli" role="planner">',
      'Sessions stored as JSONL events at `~/.magnitude/sessions/<id>/events.jsonl`',
      'Write transcript to `~/.magnitude/sessions/<id>/transcript.txt`.',
      '</spawn-worker>',
    ].join('\n')

    const events = parse(xml)

    expect(body(events)).toContain('`~/.magnitude/sessions/<id>/events.jsonl`')
    expect(body(events)).toContain('`~/.magnitude/sessions/<id>/transcript.txt`.')
    expect(childOpens(events)).toHaveLength(0)
    expect(parseErrors(events)).toHaveLength(0)
  })
})
