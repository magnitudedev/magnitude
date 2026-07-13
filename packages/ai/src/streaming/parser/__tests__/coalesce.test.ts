import { describe, it, expect } from 'vitest'
import { coalesce } from '../coalesce'
import type { FieldEvent } from '../../types'

describe('Coalescing', () => {
  it('merges adjacent field_delta events with same path', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_delta', path: ['a'], delta: 'hel' },
      { _tag: 'field_delta', path: ['a'], delta: 'lo' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_delta', path: ['a'], delta: 'hello' },
    ])
  })

  it('does not merge field_delta events with different paths', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_delta', path: ['a'], delta: 'hel' },
      { _tag: 'field_delta', path: ['b'], delta: 'lo' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_delta', path: ['a'], delta: 'hel' },
      { _tag: 'field_delta', path: ['b'], delta: 'lo' },
    ])
  })

  it('field_start breaks coalescing', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_delta', path: ['a'], delta: 'hel' },
      { _tag: 'field_start', path: ['b'] },
      { _tag: 'field_delta', path: ['a'], delta: 'lo' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_delta', path: ['a'], delta: 'hel' },
      { _tag: 'field_start', path: ['b'] },
      { _tag: 'field_delta', path: ['a'], delta: 'lo' },
    ])
  })

  it('field_end breaks coalescing', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_delta', path: ['a'], delta: 'hel' },
      { _tag: 'field_end', path: ['a'], value: 'hello' },
      { _tag: 'field_delta', path: ['a'], delta: 'lo' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_delta', path: ['a'], delta: 'hel' },
      { _tag: 'field_end', path: ['a'], value: 'hello' },
      { _tag: 'field_delta', path: ['a'], delta: 'lo' },
    ])
  })

  it('empty delta strings are merged', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_delta', path: ['a'], delta: '' },
      { _tag: 'field_delta', path: ['a'], delta: 'hello' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_delta', path: ['a'], delta: 'hello' },
    ])
  })

  it('three consecutive deltas same path', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_delta', path: ['a'], delta: 'h' },
      { _tag: 'field_delta', path: ['a'], delta: 'e' },
      { _tag: 'field_delta', path: ['a'], delta: 'llo' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_delta', path: ['a'], delta: 'hello' },
    ])
  })

  it('nested path coalescing', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_delta', path: ['a', 'b', '0'], delta: 'he' },
      { _tag: 'field_delta', path: ['a', 'b', '0'], delta: 'llo' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_delta', path: ['a', 'b', '0'], delta: 'hello' },
    ])
  })

  it('empty input returns empty', () => {
    expect(coalesce([])).toEqual([])
  })

  it('single event passes through', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_start', path: ['a'] },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_start', path: ['a'] },
    ])
  })

  it('alternating paths not merged', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_delta', path: ['a'], delta: '1' },
      { _tag: 'field_delta', path: ['b'], delta: '2' },
      { _tag: 'field_delta', path: ['a'], delta: '3' },
      { _tag: 'field_delta', path: ['b'], delta: '4' },
    ]
    expect(coalesce(events)).toEqual(events)
  })

  // =========================================================================
  // LONG CHAINS
  // =========================================================================
  it('10+ consecutive deltas same path', () => {
    const events: FieldEvent[] = Array.from({ length: 15 }, (_, i) => ({
      _tag: 'field_delta' as const,
      path: ['message'],
      delta: String.fromCharCode(97 + (i % 26)),
    }))
    const result = coalesce(events)
    expect(result.length).toBe(1)
    expect(result[0]._tag).toBe('field_delta')
    if (result[0]._tag === 'field_delta') {
      expect(result[0].delta).toBe(events.map(e => (e as any).delta).join(''))
    }
  })

  it('20 deltas same path', () => {
    const events: FieldEvent[] = Array.from({ length: 20 }, () => ({
      _tag: 'field_delta' as const,
      path: ['content'],
      delta: 'x',
    }))
    const result = coalesce(events)
    expect(result.length).toBe(1)
    if (result[0]._tag === 'field_delta') {
      expect(result[0].delta).toBe('x'.repeat(20))
    }
  })

  // =========================================================================
  // MIXED EVENT TYPES IN COMPLEX SEQUENCES
  // =========================================================================
  it('start, deltas, end, start, deltas, end', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_start', path: ['a'] },
      { _tag: 'field_delta', path: ['a'], delta: 'hel' },
      { _tag: 'field_delta', path: ['a'], delta: 'lo' },
      { _tag: 'field_end', path: ['a'], value: 'hello' },
      { _tag: 'field_start', path: ['b'] },
      { _tag: 'field_delta', path: ['b'], delta: 'wor' },
      { _tag: 'field_delta', path: ['b'], delta: 'ld' },
      { _tag: 'field_end', path: ['b'], value: 'world' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_start', path: ['a'] },
      { _tag: 'field_delta', path: ['a'], delta: 'hello' },
      { _tag: 'field_end', path: ['a'], value: 'hello' },
      { _tag: 'field_start', path: ['b'] },
      { _tag: 'field_delta', path: ['b'], delta: 'world' },
      { _tag: 'field_end', path: ['b'], value: 'world' },
    ])
  })

  it('multiple field lifecycles on same path', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_start', path: ['a'] },
      { _tag: 'field_delta', path: ['a'], delta: 'first' },
      { _tag: 'field_end', path: ['a'], value: 'first' },
      { _tag: 'field_start', path: ['a'] },
      { _tag: 'field_delta', path: ['a'], delta: 'sec' },
      { _tag: 'field_delta', path: ['a'], delta: 'ond' },
      { _tag: 'field_end', path: ['a'], value: 'second' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_start', path: ['a'] },
      { _tag: 'field_delta', path: ['a'], delta: 'first' },
      { _tag: 'field_end', path: ['a'], value: 'first' },
      { _tag: 'field_start', path: ['a'] },
      { _tag: 'field_delta', path: ['a'], delta: 'second' },
      { _tag: 'field_end', path: ['a'], value: 'second' },
    ])
  })

  // =========================================================================
  // REAL-WORLD EVENT SEQUENCES
  // =========================================================================
  it('real-world: parsing {"message": "hello world"}', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_start', path: ['message'] },
      { _tag: 'field_delta', path: ['message'], delta: 'h' },
      { _tag: 'field_delta', path: ['message'], delta: 'e' },
      { _tag: 'field_delta', path: ['message'], delta: 'l' },
      { _tag: 'field_delta', path: ['message'], delta: 'l' },
      { _tag: 'field_delta', path: ['message'], delta: 'o' },
      { _tag: 'field_delta', path: ['message'], delta: ' ' },
      { _tag: 'field_delta', path: ['message'], delta: 'w' },
      { _tag: 'field_delta', path: ['message'], delta: 'o' },
      { _tag: 'field_delta', path: ['message'], delta: 'r' },
      { _tag: 'field_delta', path: ['message'], delta: 'l' },
      { _tag: 'field_delta', path: ['message'], delta: 'd' },
      { _tag: 'field_end', path: ['message'], value: 'hello world' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_start', path: ['message'] },
      { _tag: 'field_delta', path: ['message'], delta: 'hello world' },
      { _tag: 'field_end', path: ['message'], value: 'hello world' },
    ])
  })

  it('interleaved fields from multi-key object', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_start', path: ['a'] },
      { _tag: 'field_delta', path: ['a'], delta: 'x' },
      { _tag: 'field_delta', path: ['a'], delta: 'y' },
      { _tag: 'field_end', path: ['a'], value: 'xy' },
      { _tag: 'field_start', path: ['b'] },
      { _tag: 'field_delta', path: ['b'], delta: '1' },
      { _tag: 'field_delta', path: ['b'], delta: '2' },
      { _tag: 'field_delta', path: ['b'], delta: '3' },
      { _tag: 'field_end', path: ['b'], value: '123' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_start', path: ['a'] },
      { _tag: 'field_delta', path: ['a'], delta: 'xy' },
      { _tag: 'field_end', path: ['a'], value: 'xy' },
      { _tag: 'field_start', path: ['b'] },
      { _tag: 'field_delta', path: ['b'], delta: '123' },
      { _tag: 'field_end', path: ['b'], value: '123' },
    ])
  })

  it('deeply nested path deltas', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_start', path: ['a', 'b', 'c', '0'] },
      { _tag: 'field_delta', path: ['a', 'b', 'c', '0'], delta: 'deep' },
      { _tag: 'field_delta', path: ['a', 'b', 'c', '0'], delta: 'value' },
      { _tag: 'field_end', path: ['a', 'b', 'c', '0'], value: 'deepvalue' },
    ]
    expect(coalesce(events)).toEqual([
      { _tag: 'field_start', path: ['a', 'b', 'c', '0'] },
      { _tag: 'field_delta', path: ['a', 'b', 'c', '0'], delta: 'deepvalue' },
      { _tag: 'field_end', path: ['a', 'b', 'c', '0'], value: 'deepvalue' },
    ])
  })

  it('only start and end events, no deltas', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_start', path: ['a'] },
      { _tag: 'field_end', path: ['a'], value: '' },
    ]
    expect(coalesce(events)).toEqual(events)
  })

  it('single delta event', () => {
    const events: FieldEvent[] = [
      { _tag: 'field_delta', path: ['a'], delta: 'hello' },
    ]
    expect(coalesce(events)).toEqual(events)
  })
})
