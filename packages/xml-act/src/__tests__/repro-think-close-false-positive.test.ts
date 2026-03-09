import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser/streaming-xml-parser'
import type { ParseEvent } from '../parser/types'

function collectParseEvents(input: string): ParseEvent[] {
  const parser = createStreamingXmlParser(new Set(), new Map())
  const events = parser.processChunk(input)
  const flushEvents = parser.flush()
  return [...events, ...flushEvents]
}

describe('think block close at stream end (no trailing newline)', () => {
  it('should not emit UnclosedThink for single-line think block without trailing newline', () => {
    // Exact case from the bug report: model outputs a properly closed think block
    // but </think> is not followed by a newline
    const input = '<think>Waiting for the planner.</think>'
    const events = collectParseEvents(input)

    const parseErrors = events.filter(e => e._tag === 'ParseError')
    console.log('Events:', events.map(e => e._tag))
    if (parseErrors.length > 0) {
      console.log('Parse errors:', JSON.stringify(parseErrors, null, 2))
    }
    expect(parseErrors).toEqual([])
  })

  it('should not emit UnclosedThink for multi-line think block (close after newline)', () => {
    const input = `<think>
Some thinking here.
</think>`
    const events = collectParseEvents(input)

    const parseErrors = events.filter(e => e._tag === 'ParseError')
    expect(parseErrors).toEqual([])
  })

  it('should emit UnclosedThink for actually unclosed think blocks', () => {
    const input = '<think>This is never closed'
    const events = collectParseEvents(input)

    const parseErrors = events.filter(e => e._tag === 'ParseError')
    expect(parseErrors.length).toBe(1)
    expect((parseErrors[0] as { error: { _tag: string } }).error._tag).toBe('UnclosedThink')
  })
})
