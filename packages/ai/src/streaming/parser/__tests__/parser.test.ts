import { describe, it, expect } from 'vitest'
import { createIncrementalJsonParser } from '../index'
import type { ParsedValue } from '../../types'

/** Parse a complete JSON string and return the final partial */
function parse(input: string): ParsedValue | undefined {
  const p = createIncrementalJsonParser()
  p.push(input)
  p.end()
  return p.partial
}

/** Parse incrementally (one chunk per arg) and return final partial */
function parseChunks(...chunks: string[]): ParsedValue | undefined {
  const p = createIncrementalJsonParser()
  for (const c of chunks) p.push(c)
  p.end()
  return p.partial
}

/** Parse incrementally and return partial after each push (before end) */
function parseSteps(...chunks: string[]): (ParsedValue | undefined)[] {
  const p = createIncrementalJsonParser()
  const results: (ParsedValue | undefined)[] = []
  for (const c of chunks) {
    p.push(c)
    results.push(structuredClone(p.partial))
  }
  return results
}

/** Parse and return currentPath at each step */
function parsePaths(...chunks: string[]): (readonly string[])[] {
  const p = createIncrementalJsonParser()
  const results: (readonly string[])[] = []
  for (const c of chunks) {
    p.push(c)
    results.push([...p.currentPath])
  }
  return results
}

describe('JsonParser', () => {
  // =========================================================================
  // SCALARS
  // =========================================================================
  describe('scalars', () => {
    it('string', () => {
      expect(parse('"hello"')).toEqual({
        _tag: 'string', value: 'hello', state: 'complete',
      })
    })

    it('number', () => {
      expect(parse('42')).toEqual({
        _tag: 'number', value: '42', state: 'complete',
      })
    })

    it('true', () => {
      expect(parse('true')).toEqual({
        _tag: 'boolean', value: true, state: 'complete',
      })
    })

    it('false', () => {
      expect(parse('false')).toEqual({
        _tag: 'boolean', value: false, state: 'complete',
      })
    })

    it('null', () => {
      expect(parse('null')).toEqual({
        _tag: 'null', state: 'complete',
      })
    })

    it('incomplete string at EOF', () => {
      expect(parse('"hello')).toEqual({
        _tag: 'string', value: 'hello', state: 'incomplete',
      })
    })

    it('incomplete number at EOF', () => {
      expect(parse('1.')).toEqual({
        _tag: 'number', value: '1.', state: 'incomplete',
      })
    })
  })

  // =========================================================================
  // OBJECTS
  // =========================================================================
  describe('objects', () => {
    it('empty object', () => {
      expect(parse('{}')).toEqual({
        _tag: 'object', entries: [], state: 'complete',
      })
    })

    it('single key-value', () => {
      expect(parse('{"a": 1}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'number', value: '1', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('multiple keys', () => {
      expect(parse('{"a": 1, "b": 2}')).toEqual({
        _tag: 'object',
        entries: [
          ['a', { _tag: 'number', value: '1', state: 'complete' }],
          ['b', { _tag: 'number', value: '2', state: 'complete' }],
        ],
        state: 'complete',
      })
    })

    it('nested objects', () => {
      expect(parse('{"a": {"b": 1}}')).toEqual({
        _tag: 'object',
        entries: [
          ['a', {
            _tag: 'object',
            entries: [['b', { _tag: 'number', value: '1', state: 'complete' }]],
            state: 'complete',
          }],
        ],
        state: 'complete',
      })
    })

    it('unquoted keys (permissive)', () => {
      expect(parse('{a: 1}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'number', value: '1', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('partial: missing value after colon', () => {
      const result = parse('{"a":')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      expect((result as any).state).toBe('incomplete')
    })

    it('partial: missing colon after key', () => {
      const result = parse('{"a"')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      expect((result as any).state).toBe('incomplete')
    })

    it('trailing comma', () => {
      const result = parse('{"a": 1,}')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      // Should still produce a valid object
      expect((result as any).entries).toEqual([
        ['a', { _tag: 'number', value: '1', state: 'complete' }],
      ])
    })

    it('string values', () => {
      expect(parse('{"a": "hello"}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'string', value: 'hello', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('boolean values', () => {
      expect(parse('{"a": true, "b": false}')).toEqual({
        _tag: 'object',
        entries: [
          ['a', { _tag: 'boolean', value: true, state: 'complete' }],
          ['b', { _tag: 'boolean', value: false, state: 'complete' }],
        ],
        state: 'complete',
      })
    })

    it('null value', () => {
      expect(parse('{"a": null}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'null', state: 'complete' }]],
        state: 'complete',
      })
    })
  })

  // =========================================================================
  // ARRAYS
  // =========================================================================
  describe('arrays', () => {
    it('empty array', () => {
      expect(parse('[]')).toEqual({
        _tag: 'array', items: [], state: 'complete',
      })
    })

    it('single element', () => {
      expect(parse('[1]')).toEqual({
        _tag: 'array',
        items: [{ _tag: 'number', value: '1', state: 'complete' }],
        state: 'complete',
      })
    })

    it('multiple elements', () => {
      expect(parse('[1, 2, 3]')).toEqual({
        _tag: 'array',
        items: [
          { _tag: 'number', value: '1', state: 'complete' },
          { _tag: 'number', value: '2', state: 'complete' },
          { _tag: 'number', value: '3', state: 'complete' },
        ],
        state: 'complete',
      })
    })

    it('mixed types', () => {
      expect(parse('[1, "two", true, null]')).toEqual({
        _tag: 'array',
        items: [
          { _tag: 'number', value: '1', state: 'complete' },
          { _tag: 'string', value: 'two', state: 'complete' },
          { _tag: 'boolean', value: true, state: 'complete' },
          { _tag: 'null', state: 'complete' },
        ],
        state: 'complete',
      })
    })

    it('nested arrays', () => {
      expect(parse('[[1, 2], [3, 4]]')).toEqual({
        _tag: 'array',
        items: [
          {
            _tag: 'array',
            items: [
              { _tag: 'number', value: '1', state: 'complete' },
              { _tag: 'number', value: '2', state: 'complete' },
            ],
            state: 'complete',
          },
          {
            _tag: 'array',
            items: [
              { _tag: 'number', value: '3', state: 'complete' },
              { _tag: 'number', value: '4', state: 'complete' },
            ],
            state: 'complete',
          },
        ],
        state: 'complete',
      })
    })

    it('array containing objects', () => {
      expect(parse('[{"a": 1}, {"b": 2}]')).toEqual({
        _tag: 'array',
        items: [
          {
            _tag: 'object',
            entries: [['a', { _tag: 'number', value: '1', state: 'complete' }]],
            state: 'complete',
          },
          {
            _tag: 'object',
            entries: [['b', { _tag: 'number', value: '2', state: 'complete' }]],
            state: 'complete',
          },
        ],
        state: 'complete',
      })
    })

    it('trailing comma', () => {
      const result = parse('[1,]')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('array')
      expect((result as any).items).toEqual([
        { _tag: 'number', value: '1', state: 'complete' },
      ])
    })
  })

  // =========================================================================
  // DEEP NESTING
  // =========================================================================
  describe('deep nesting', () => {
    it('10 levels of nested objects', () => {
      const input = '{"a":'.repeat(10) + '1' + '}'.repeat(10)
      const result = parse(input)
      expect(result).toBeDefined()
      // Walk down to the innermost value
      let current: any = result
      for (let i = 0; i < 10; i++) {
        expect(current._tag).toBe('object')
        expect(current.entries.length).toBe(1)
        expect(current.entries[0][0]).toBe('a')
        current = current.entries[0][1]
      }
      expect(current).toEqual({ _tag: 'number', value: '1', state: 'complete' })
    })

    it('mixed nesting: object > array > object', () => {
      expect(parse('{"a": [{"b": 1}]}')).toEqual({
        _tag: 'object',
        entries: [
          ['a', {
            _tag: 'array',
            items: [{
              _tag: 'object',
              entries: [['b', { _tag: 'number', value: '1', state: 'complete' }]],
              state: 'complete',
            }],
            state: 'complete',
          }],
        ],
        state: 'complete',
      })
    })

    it('deeply nested with multiple incomplete levels', () => {
      const result = parse('{"a": [{"b":')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      expect((result as any).state).toBe('incomplete')
    })
  })

  // =========================================================================
  // PARTIAL STATES (STREAMING)
  // =========================================================================
  describe('partial states', () => {
    it('object with incomplete string value shows partial', () => {
      const p = createIncrementalJsonParser()
      p.push('{"a": "hel')
      const partial = p.partial
      expect(partial).toBeDefined()
      expect(partial!._tag).toBe('object')
      if (partial!._tag === 'object') {
        expect(partial!.entries.length).toBe(1)
        expect(partial!.entries[0][0]).toBe('a')
        expect(partial!.entries[0][1]._tag).toBe('string')
        expect((partial!.entries[0][1] as any).value).toBe('hel')
        expect((partial!.entries[0][1] as any).state).toBe('incomplete')
      }
    })

    it('array with incomplete number shows partial', () => {
      const p = createIncrementalJsonParser()
      p.push('[42')
      const partial = p.partial
      expect(partial).toBeDefined()
      expect(partial!._tag).toBe('array')
      if (partial!._tag === 'array') {
        expect(partial!.items.length).toBe(1)
        expect(partial!.items[0]._tag).toBe('number')
        expect((partial!.items[0] as any).value).toBe('42')
        expect((partial!.items[0] as any).state).toBe('incomplete')
      }
    })

    it('partial grows as more data arrives', () => {
      const p = createIncrementalJsonParser()
      p.push('{"a": "h')
      const p1 = structuredClone(p.partial)
      p.push('ello')
      const p2 = structuredClone(p.partial)
      p.push('"')
      const p3 = structuredClone(p.partial)
      p.push('}')
      const p4 = structuredClone(p.partial)

      // p1: incomplete string "h"
      expect((p1 as any).entries[0][1].value).toBe('h')
      expect((p1 as any).entries[0][1].state).toBe('incomplete')

      // p2: incomplete string "hello"
      expect((p2 as any).entries[0][1].value).toBe('hello')
      expect((p2 as any).entries[0][1].state).toBe('incomplete')

      // p3: complete string "hello"
      expect((p3 as any).entries[0][1].value).toBe('hello')
      expect((p3 as any).entries[0][1].state).toBe('complete')

      // p4: complete object
      expect((p4 as any).state).toBe('complete')
    })
  })

  // =========================================================================
  // OBJECTS WITH MANY KEYS
  // =========================================================================
  describe('objects with many keys', () => {
    it('object with 10+ keys', () => {
      const entries = Array.from({ length: 12 }, (_, i) => `"k${i}": ${i}`)
      const input = '{' + entries.join(', ') + '}'
      const result = parse(input)
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      if (result!._tag === 'object') {
        expect(result!.entries.length).toBe(12)
        for (let i = 0; i < 12; i++) {
          expect(result!.entries[i][0]).toBe(`k${i}`)
          expect(result!.entries[i][1]).toEqual({ _tag: 'number', value: String(i), state: 'complete' })
        }
      }
    })
  })

  // =========================================================================
  // ARRAYS WITH MANY ELEMENTS
  // =========================================================================
  describe('arrays with many elements', () => {
    it('array with 20+ elements', () => {
      const elements = Array.from({ length: 25 }, (_, i) => String(i))
      const input = '[' + elements.join(',') + ']'
      const result = parse(input)
      expect(result).toBeDefined()
      expect(result!._tag).toBe('array')
      if (result!._tag === 'array') {
        expect(result!.items.length).toBe(25)
        for (let i = 0; i < 25; i++) {
          expect(result!.items[i]).toEqual({ _tag: 'number', value: String(i), state: 'complete' })
        }
      }
    })
  })

  // =========================================================================
  // EVERY VALUE TYPE AS OBJECT VALUE
  // =========================================================================
  describe('every value type as object value', () => {
    it('string value', () => {
      expect(parse('{"a": "hello"}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'string', value: 'hello', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('number value', () => {
      expect(parse('{"a": 42}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'number', value: '42', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('true value', () => {
      expect(parse('{"a": true}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'boolean', value: true, state: 'complete' }]],
        state: 'complete',
      })
    })

    it('false value', () => {
      expect(parse('{"a": false}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'boolean', value: false, state: 'complete' }]],
        state: 'complete',
      })
    })

    it('null value', () => {
      expect(parse('{"a": null}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'null', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('object value', () => {
      expect(parse('{"a": {"b": 1}}')).toEqual({
        _tag: 'object',
        entries: [['a', {
          _tag: 'object',
          entries: [['b', { _tag: 'number', value: '1', state: 'complete' }]],
          state: 'complete',
        }]],
        state: 'complete',
      })
    })

    it('array value', () => {
      expect(parse('{"a": [1, 2]}')).toEqual({
        _tag: 'object',
        entries: [['a', {
          _tag: 'array',
          items: [
            { _tag: 'number', value: '1', state: 'complete' },
            { _tag: 'number', value: '2', state: 'complete' },
          ],
          state: 'complete',
        }]],
        state: 'complete',
      })
    })
  })

  // =========================================================================
  // EVERY VALUE TYPE AS ARRAY ELEMENT
  // =========================================================================
  describe('every value type as array element', () => {
    it('string element', () => {
      expect(parse('["hello"]')).toEqual({
        _tag: 'array',
        items: [{ _tag: 'string', value: 'hello', state: 'complete' }],
        state: 'complete',
      })
    })

    it('number element', () => {
      expect(parse('[42]')).toEqual({
        _tag: 'array',
        items: [{ _tag: 'number', value: '42', state: 'complete' }],
        state: 'complete',
      })
    })

    it('true element', () => {
      expect(parse('[true]')).toEqual({
        _tag: 'array',
        items: [{ _tag: 'boolean', value: true, state: 'complete' }],
        state: 'complete',
      })
    })

    it('false element', () => {
      expect(parse('[false]')).toEqual({
        _tag: 'array',
        items: [{ _tag: 'boolean', value: false, state: 'complete' }],
        state: 'complete',
      })
    })

    it('null element', () => {
      expect(parse('[null]')).toEqual({
        _tag: 'array',
        items: [{ _tag: 'null', state: 'complete' }],
        state: 'complete',
      })
    })

    it('object element', () => {
      expect(parse('[{"a": 1}]')).toEqual({
        _tag: 'array',
        items: [{
          _tag: 'object',
          entries: [['a', { _tag: 'number', value: '1', state: 'complete' }]],
          state: 'complete',
        }],
        state: 'complete',
      })
    })

    it('array element', () => {
      expect(parse('[[1, 2]]')).toEqual({
        _tag: 'array',
        items: [{
          _tag: 'array',
          items: [
            { _tag: 'number', value: '1', state: 'complete' },
            { _tag: 'number', value: '2', state: 'complete' },
          ],
          state: 'complete',
        }],
        state: 'complete',
      })
    })
  })

  // =========================================================================
  // NESTED EMPTY CONTAINERS
  // =========================================================================
  describe('nested empty containers', () => {
    it('nested empty objects', () => {
      expect(parse('{"a": {"b": {}}}')).toEqual({
        _tag: 'object',
        entries: [['a', {
          _tag: 'object',
          entries: [['b', { _tag: 'object', entries: [], state: 'complete' }]],
          state: 'complete',
        }]],
        state: 'complete',
      })
    })

    it('nested empty arrays', () => {
      expect(parse('[[], []]')).toEqual({
        _tag: 'array',
        items: [
          { _tag: 'array', items: [], state: 'complete' },
          { _tag: 'array', items: [], state: 'complete' },
        ],
        state: 'complete',
      })
    })

    it('triple nested arrays', () => {
      expect(parse('[[[1]]]')).toEqual({
        _tag: 'array',
        items: [{
          _tag: 'array',
          items: [{
            _tag: 'array',
            items: [{ _tag: 'number', value: '1', state: 'complete' }],
            state: 'complete',
          }],
          state: 'complete',
        }],
        state: 'complete',
      })
    })
  })

  // =========================================================================
  // OBJECT WITH ARRAY OF OBJECTS
  // =========================================================================
  describe('object with array of objects', () => {
    it('items array pattern', () => {
      expect(parse('{"items": [{"id": 1}, {"id": 2}]}')).toEqual({
        _tag: 'object',
        entries: [['items', {
          _tag: 'array',
          items: [
            { _tag: 'object', entries: [['id', { _tag: 'number', value: '1', state: 'complete' }]], state: 'complete' },
            { _tag: 'object', entries: [['id', { _tag: 'number', value: '2', state: 'complete' }]], state: 'complete' },
          ],
          state: 'complete',
        }]],
        state: 'complete',
      })
    })
  })

  // =========================================================================
  // PARTIAL PROGRESSION: EVERY CHARACTER
  // =========================================================================
  describe('partial progression character by character', () => {
    it('{"a": true} — partial after every char', () => {
      const input = '{"a": true}'
      const p = createIncrementalJsonParser()
      const partials: (ParsedValue | undefined)[] = []
      for (const ch of input) {
        p.push(ch)
        partials.push(structuredClone(p.partial))
      }

      // After '{' — empty incomplete object
      expect(partials[0]?._tag).toBe('object')
      expect((partials[0] as any)?.state).toBe('incomplete')

      // After '{"a"' — object with key but no value yet
      expect(partials[3]?._tag).toBe('object')

      // After '{"a": true' (index 9) — "true" is still pending as keyword (no delimiter seen)
      // The keyword hasn't been finalized yet, so it may appear as incomplete string/unquoted
      expect(partials[9]?._tag).toBe('object')

      // After '{"a": true}' (index 10) — } terminates keyword, complete object
      expect(partials[10]?._tag).toBe('object')
      expect((partials[10] as any)?.state).toBe('complete')
      if (partials[10]?._tag === 'object') {
        expect(partials[10].entries.length).toBe(1)
        expect(partials[10].entries[0][1]._tag).toBe('boolean')
        expect((partials[10].entries[0][1] as any).value).toBe(true)
      }
    })

    it('[1, 2, 3] — array grows element by element', () => {
      const steps = parseSteps('[', '1', ',', ' ', '2', ',', ' ', '3', ']')

      // After '[' — empty incomplete array
      expect(steps[0]?._tag).toBe('array')
      expect((steps[0] as any)?.items.length).toBeLessThanOrEqual(1)

      // After '[1,' — array with 1 complete element
      expect(steps[2]?._tag).toBe('array')
      if (steps[2]?._tag === 'array') {
        expect(steps[2].items.length).toBe(1)
      }

      // After '[1, 2,' — array with 2 elements
      expect(steps[5]?._tag).toBe('array')
      if (steps[5]?._tag === 'array') {
        expect(steps[5].items.length).toBe(2)
      }

      // After '[1, 2, 3]' — complete array with 3 elements
      expect(steps[8]?._tag).toBe('array')
      if (steps[8]?._tag === 'array') {
        expect(steps[8].items.length).toBe(3)
        expect((steps[8] as any).state).toBe('complete')
      }
    })
  })

  // =========================================================================
  // PATH TRACKING FOR DEEPLY NESTED AND ARRAYS
  // =========================================================================
  describe('path tracking for deeply nested structures', () => {
    it('tracks numeric indices in arrays', () => {
      const paths = parsePaths('[', '"a"', ',', '"b"', ',', '"c"')
      // After first element
      expect(paths[1]).toEqual(['0'])
      // After second element
      expect(paths[3]).toEqual(['1'])
      // After third element
      expect(paths[5]).toEqual(['2'])
    })

    it('tracks nested object paths', () => {
      const p = createIncrementalJsonParser()
      p.push('{"a": {"b": {"c":')
      expect([...p.currentPath]).toEqual(['a', 'b', 'c'])
    })
  })

  // =========================================================================
  // UNQUOTED KEYS IN VARIOUS POSITIONS
  // =========================================================================
  describe('unquoted keys in various positions', () => {
    it('multiple unquoted keys', () => {
      expect(parse('{a: 1, b: 2}')).toEqual({
        _tag: 'object',
        entries: [
          ['a', { _tag: 'number', value: '1', state: 'complete' }],
          ['b', { _tag: 'number', value: '2', state: 'complete' }],
        ],
        state: 'complete',
      })
    })

    it('unquoted key with nested object', () => {
      expect(parse('{a: {b: 1}}')).toEqual({
        _tag: 'object',
        entries: [['a', {
          _tag: 'object',
          entries: [['b', { _tag: 'number', value: '1', state: 'complete' }]],
          state: 'complete',
        }]],
        state: 'complete',
      })
    })
  })

  // =========================================================================
  // TRAILING COMMAS
  // =========================================================================
  describe('trailing commas', () => {
    it('trailing comma in nested object', () => {
      const result = parse('{"a": {"b": 1,}}')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
    })

    it('trailing comma in nested array', () => {
      const result = parse('{"a": [1, 2,]}')
      expect(result).toBeDefined()
      if (result!._tag === 'object') {
        const arr = result!.entries[0][1]
        expect(arr._tag).toBe('array')
        if (arr._tag === 'array') {
          expect(arr.items.length).toBe(2)
        }
      }
    })
  })

  // =========================================================================
  // WHITESPACE-HEAVY INPUTS
  // =========================================================================
  describe('whitespace-heavy inputs', () => {
    it('lots of whitespace around everything', () => {
      const input = '  {  "a"  :  1  ,  "b"  :  2  }  '
      const result = parse(input)
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      if (result!._tag === 'object') {
        expect(result!.entries.length).toBe(2)
      }
    })

    it('newlines and tabs', () => {
      const input = '{\n\t"a":\n\t\t1,\n\t"b":\n\t\t2\n}'
      expect(parse(input)).toEqual({
        _tag: 'object',
        entries: [
          ['a', { _tag: 'number', value: '1', state: 'complete' }],
          ['b', { _tag: 'number', value: '2', state: 'complete' }],
        ],
        state: 'complete',
      })
    })
  })

  // =========================================================================
  // MIXED NESTING 5+ LEVELS DEEP
  // =========================================================================
  describe('mixed nesting 5+ levels', () => {
    it('object > array > object > array > object > value', () => {
      const input = '{"a": [{"b": [{"c": 42}]}]}'
      const result = parse(input)
      expect(result).toBeDefined()
      // Walk down
      let current: any = result
      expect(current._tag).toBe('object')
      current = current.entries[0][1] // array
      expect(current._tag).toBe('array')
      current = current.items[0] // object
      expect(current._tag).toBe('object')
      current = current.entries[0][1] // array
      expect(current._tag).toBe('array')
      current = current.items[0] // object
      expect(current._tag).toBe('object')
      current = current.entries[0][1] // number
      expect(current).toEqual({ _tag: 'number', value: '42', state: 'complete' })
    })

    it('6 levels of alternating nesting', () => {
      const input = '{"a": [{"b": [{"c": [1]}]}]}'
      const result = parse(input)
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
    })
  })

  // =========================================================================
  // CHUNKED PARSING
  // =========================================================================
  describe('chunked parsing', () => {
    it('object split across many chunks', () => {
      expect(parseChunks('{', '"a"', ':', ' ', '1', ',', '"b"', ':', '2', '}')).toEqual({
        _tag: 'object',
        entries: [
          ['a', { _tag: 'number', value: '1', state: 'complete' }],
          ['b', { _tag: 'number', value: '2', state: 'complete' }],
        ],
        state: 'complete',
      })
    })

    it('nested structure char by char', () => {
      const input = '{"a":[1]}'
      const result = parseChunks(...input.split(''))
      expect(result).toEqual(parse(input))
    })
  })

  // =========================================================================
  // PATH TRACKING
  // =========================================================================
  describe('path tracking', () => {
    it('tracks path through nested structure', () => {
      const p = createIncrementalJsonParser()

      p.push('{')
      expect([...p.currentPath]).toEqual([])

      p.push('"a"')
      expect([...p.currentPath]).toEqual(['a'])

      p.push(':')
      expect([...p.currentPath]).toEqual(['a'])

      p.push('{')
      expect([...p.currentPath]).toEqual(['a'])

      p.push('"b"')
      expect([...p.currentPath]).toEqual(['a', 'b'])

      p.push(':')
      expect([...p.currentPath]).toEqual(['a', 'b'])

      p.push('[')
      expect([...p.currentPath]).toEqual(['a', 'b'])

      p.push('1')
      expect([...p.currentPath]).toEqual(['a', 'b', '0'])

      p.push(',')
      // After comma in array, index advances
      p.push('2')
      expect([...p.currentPath]).toEqual(['a', 'b', '1'])
    })
  })

  // =========================================================================
  // EMPTY / WHITESPACE INPUT
  // =========================================================================
  describe('empty and whitespace input', () => {
    it('empty input: partial should be undefined', () => {
      expect(parse('')).toBeUndefined()
    })

    it('whitespace-only input: partial should be undefined', () => {
      expect(parse('   ')).toBeUndefined()
    })
  })

  // =========================================================================
  // JUST OPENING BRACE / BRACKET
  // =========================================================================
  describe('just opening delimiters', () => {
    it('just opening brace: incomplete object', () => {
      const result = parse('{')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      expect((result as any).state).toBe('incomplete')
    })

    it('just opening bracket: incomplete array', () => {
      const result = parse('[')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('array')
      expect((result as any).state).toBe('incomplete')
    })
  })

  // =========================================================================
  // DEEPLY INCOMPLETE
  // =========================================================================
  describe('deeply incomplete structures', () => {
    it('{"a": {"b": [ — nested incomplete', () => {
      const result = parse('{"a": {"b": [')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      expect((result as any).state).toBe('incomplete')
      if (result!._tag === 'object') {
        expect(result!.entries.length).toBe(1)
        const inner = result!.entries[0][1]
        expect(inner._tag).toBe('object')
        if (inner._tag === 'object') {
          expect(inner.entries.length).toBe(1)
          const arr = inner.entries[0][1]
          expect(arr._tag).toBe('array')
          expect((arr as any).state).toBe('incomplete')
        }
      }
    })
  })

  // =========================================================================
  // DUPLICATE KEYS
  // =========================================================================
  describe('duplicate keys', () => {
    it('second value wins for duplicate keys', () => {
      const result = parse('{"a": 1, "a": 2}')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      if (result!._tag === 'object') {
        // Both entries present (JSON parser preserves all entries)
        // or second wins — check actual behavior
        const aEntries = result!.entries.filter(([k]) => k === 'a')
        const lastValue = aEntries[aEntries.length - 1][1]
        expect(lastValue).toEqual({ _tag: 'number', value: '2', state: 'complete' })
      }
    })
  })

  // =========================================================================
  // PARTIAL KEY / VALUE VISIBILITY
  // =========================================================================
  describe('partial key and value visibility', () => {
    it('partial key visibility: push {"he — shows incomplete object', () => {
      const p = createIncrementalJsonParser()
      p.push('{"he')
      const partial = p.partial
      expect(partial).toBeDefined()
      expect(partial!._tag).toBe('object')
      expect((partial as any).state).toBe('incomplete')
    })

    it('partial value with pending number: push {"a": 12', () => {
      const p = createIncrementalJsonParser()
      p.push('{"a": 12')
      const partial = p.partial
      expect(partial).toBeDefined()
      expect(partial!._tag).toBe('object')
      if (partial!._tag === 'object') {
        expect(partial!.entries.length).toBe(1)
        expect(partial!.entries[0][0]).toBe('a')
        expect(partial!.entries[0][1]._tag).toBe('number')
        expect((partial!.entries[0][1] as any).value).toBe('12')
        expect((partial!.entries[0][1] as any).state).toBe('incomplete')
      }
    })
  })

  // =========================================================================
  // MULTIPLE ROOT VALUES
  // =========================================================================
  describe('multiple root values', () => {
    it('push 1 then 2: second replaces first', () => {
      const result = parseChunks('1', '2')
      // 1 then 2 without separator — tokenizer sees "12" as one number
      expect(result).toBeDefined()
      expect(result!._tag).toBe('number')
    })
  })

  // =========================================================================
  // BOOLEAN / NULL AS ROOT VALUE CHUNKED
  // =========================================================================
  describe('boolean and null as root value chunked', () => {
    it('boolean true chunked: tr + ue', () => {
      expect(parseChunks('tr', 'ue')).toEqual({
        _tag: 'boolean', value: true, state: 'complete',
      })
    })

    it('null chunked: nu + ll', () => {
      expect(parseChunks('nu', 'll')).toEqual({
        _tag: 'null', state: 'complete',
      })
    })
  })

  // =========================================================================
  // ARRAY WITH NESTED INCOMPLETE OBJECT
  // =========================================================================
  describe('array with nested incomplete structures', () => {
    it('[{"a": 1}, {"b": — second object incomplete', () => {
      const result = parse('[{"a": 1}, {"b":')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('array')
      if (result!._tag === 'array') {
        expect(result!.items.length).toBe(2)
        expect(result!.items[0]._tag).toBe('object')
        expect((result!.items[0] as any).state).toBe('complete')
        expect(result!.items[1]._tag).toBe('object')
        expect((result!.items[1] as any).state).toBe('incomplete')
      }
    })

    it('object value is incomplete array: {"items": [1, 2', () => {
      const result = parse('{"items": [1, 2')
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      if (result!._tag === 'object') {
        const arr = result!.entries[0][1]
        expect(arr._tag).toBe('array')
        if (arr._tag === 'array') {
          expect(arr.items.length).toBe(2)
          expect((arr as any).state).toBe('incomplete')
        }
      }
    })
  })

  // =========================================================================
  // PATH TRACKING FOR ARRAY OF OBJECTS
  // =========================================================================
  describe('path tracking for array of objects', () => {
    it('tracks indices and keys in array of objects', () => {
      const p = createIncrementalJsonParser()
      p.push('[{"a":')
      expect([...p.currentPath]).toEqual(['a'])
      p.push('1},{"b":')
      expect([...p.currentPath]).toEqual(['1', 'b'])
    })

    it('path tracking during key parsing', () => {
      const p = createIncrementalJsonParser()
      p.push('{"ke')
      // Key is being parsed, path may show partial or empty
      expect(p.currentPath).toBeDefined()
    })
  })

  // =========================================================================
  // DONE GETTER
  // =========================================================================
  describe('done getter', () => {
    it('false before end(), true after', () => {
      const p = createIncrementalJsonParser()
      p.push('{"a": 1}')
      expect(p.done).toBe(false)
      p.end()
      expect(p.done).toBe(true)
    })
  })

  // =========================================================================
  // VERY DEEPLY NESTED (20 LEVELS) OBJECTS
  // =========================================================================
  describe('very deeply nested structures', () => {
    it('20 levels of nested objects', () => {
      const input = '{"a":'.repeat(20) + '1' + '}'.repeat(20)
      const result = parse(input)
      expect(result).toBeDefined()
      let current: any = result
      for (let i = 0; i < 20; i++) {
        expect(current._tag).toBe('object')
        expect(current.entries.length).toBe(1)
        current = current.entries[0][1]
      }
      expect(current).toEqual({ _tag: 'number', value: '1', state: 'complete' })
    })
  })

  // =========================================================================
  // OBJECT WITH 50 KEYS
  // =========================================================================
  describe('object with 50 keys', () => {
    it('parses 50-key object correctly', () => {
      const entries = Array.from({ length: 50 }, (_, i) => `"k${i}": ${i}`)
      const input = '{' + entries.join(', ') + '}'
      const result = parse(input)
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      if (result!._tag === 'object') {
        expect(result!.entries.length).toBe(50)
        for (let i = 0; i < 50; i++) {
          expect(result!.entries[i][0]).toBe(`k${i}`)
          expect(result!.entries[i][1]).toEqual({ _tag: 'number', value: String(i), state: 'complete' })
        }
      }
    })
  })

  // =========================================================================
  // ARRAY WITH 100 ELEMENTS
  // =========================================================================
  describe('array with 100 elements', () => {
    it('parses 100-element array correctly', () => {
      const elements = Array.from({ length: 100 }, (_, i) => String(i))
      const input = '[' + elements.join(',') + ']'
      const result = parse(input)
      expect(result).toBeDefined()
      expect(result!._tag).toBe('array')
      if (result!._tag === 'array') {
        expect(result!.items.length).toBe(100)
        for (let i = 0; i < 100; i++) {
          expect(result!.items[i]).toEqual({ _tag: 'number', value: String(i), state: 'complete' })
        }
      }
    })
  })

  // =========================================================================
  // NESTED ARRAYS 5 DEEP
  // =========================================================================
  describe('nested arrays 5 deep', () => {
    it('[[[[[1]]]]]', () => {
      const result = parse('[[[[[1]]]]]')
      expect(result).toBeDefined()
      let current: any = result
      for (let i = 0; i < 5; i++) {
        expect(current._tag).toBe('array')
        expect(current.items.length).toBe(1)
        current = current.items[0]
      }
      expect(current).toEqual({ _tag: 'number', value: '1', state: 'complete' })
    })
  })

  // =========================================================================
  // PARTIAL PROGRESSION: ARRAY OF OBJECTS TOKEN BY TOKEN
  // =========================================================================
  describe('partial progression: array of objects', () => {
    it('[{"a": 1}, {"b": 2}] fed token by token', () => {
      const p = createIncrementalJsonParser()

      p.push('[')
      expect(p.partial?._tag).toBe('array')
      expect((p.partial as any).state).toBe('incomplete')

      p.push('{"a": 1}')
      expect(p.partial?._tag).toBe('array')
      if (p.partial?._tag === 'array') {
        expect(p.partial.items.length).toBe(1)
        expect(p.partial.items[0]._tag).toBe('object')
        expect((p.partial.items[0] as any).state).toBe('complete')
      }

      p.push(',')
      if (p.partial?._tag === 'array') {
        expect(p.partial.items.length).toBe(1)
      }

      p.push('{"b": 2}')
      if (p.partial?._tag === 'array') {
        expect(p.partial.items.length).toBe(2)
        expect(p.partial.items[1]._tag).toBe('object')
      }

      p.push(']')
      expect(p.partial?._tag).toBe('array')
      expect((p.partial as any).state).toBe('complete')
      if (p.partial?._tag === 'array') {
        expect(p.partial.items.length).toBe(2)
      }
    })
  })

  // =========================================================================
  // PARTIAL PROGRESSION: OBJECT WITH GROWING ARRAY CHAR BY CHAR
  // =========================================================================
  describe('partial progression: object with growing array', () => {
    it('{"x": [1, 2, 3]} char by char shows growing array', () => {
      const input = '{"x": [1, 2, 3]}'
      const p = createIncrementalJsonParser()
      const snapshots: (ParsedValue | undefined)[] = []

      for (const ch of input) {
        p.push(ch)
        snapshots.push(structuredClone(p.partial))
      }

      // After '{"x": [1' (index 8) — array with 1 incomplete element
      const afterFirstElem = snapshots[8] as any
      expect(afterFirstElem?._tag).toBe('object')
      if (afterFirstElem?._tag === 'object') {
        const arr = afterFirstElem.entries[0]?.[1]
        expect(arr?._tag).toBe('array')
        if (arr?._tag === 'array') {
          expect(arr.items.length).toBeGreaterThanOrEqual(1)
        }
      }

      // After '{"x": [1, 2' (index 11) — array with 2 elements
      const afterSecondElem = snapshots[11] as any
      if (afterSecondElem?._tag === 'object') {
        const arr = afterSecondElem.entries[0]?.[1]
        if (arr?._tag === 'array') {
          expect(arr.items.length).toBeGreaterThanOrEqual(2)
        }
      }

      // After '{"x": [1, 2, 3]}' (last) — complete
      const final = snapshots[snapshots.length - 1] as any
      expect(final?._tag).toBe('object')
      expect(final?.state).toBe('complete')
      if (final?._tag === 'object') {
        const arr = final.entries[0][1]
        expect(arr._tag).toBe('array')
        if (arr._tag === 'array') {
          expect(arr.items.length).toBe(3)
          expect(arr.state).toBe('complete')
        }
      }
    })
  })

  // =========================================================================
  // ESCAPE HANDLING IN PARSER
  // =========================================================================
  describe('escape handling in parsed values', () => {
    it('newline escape in string value', () => {
      expect(parse('{"a": "line1\\nline2"}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'string', value: 'line1\nline2', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('tab escape in string value', () => {
      expect(parse('{"a": "col1\\tcol2"}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'string', value: 'col1\tcol2', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('escaped quotes in string value', () => {
      expect(parse('{"a": "say \\"hi\\""}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'string', value: 'say "hi"', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('unicode escape in string value', () => {
      expect(parse('{"a": "\\u0041"}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'string', value: 'A', state: 'complete' }]],
        state: 'complete',
      })
    })
  })

  // =========================================================================
  // COMPLEX REAL-WORLD STRUCTURES
  // =========================================================================
  describe('complex real-world structures', () => {
    it('tool call with nested args', () => {
      const input = '{"name": "shell", "args": {"command": "ls -la", "timeout": 30}}'
      const result = parse(input)
      expect(result).toBeDefined()
      expect(result!._tag).toBe('object')
      if (result!._tag === 'object') {
        expect(result!.entries.length).toBe(2)
        expect(result!.entries[0][0]).toBe('name')
        expect(result!.entries[1][0]).toBe('args')
        const args = result!.entries[1][1]
        expect(args._tag).toBe('object')
        if (args._tag === 'object') {
          expect(args.entries.length).toBe(2)
        }
      }
    })

    it('array of heterogeneous objects', () => {
      const input = '[{"type": "text", "value": "hello"}, {"type": "number", "value": 42}, {"type": "bool", "value": true}]'
      const result = parse(input)
      expect(result!._tag).toBe('array')
      if (result!._tag === 'array') {
        expect(result!.items.length).toBe(3)
        for (const item of result!.items) {
          expect(item._tag).toBe('object')
        }
      }
    })

    it('empty containers at every level', () => {
      expect(parse('{"a": {}, "b": [], "c": {"d": {}, "e": []}}')).toBeDefined()
    })
  })

  // =========================================================================
  // NEGATIVE NUMBER PARSING
  // =========================================================================
  describe('negative numbers', () => {
    it('-0 as value', () => {
      expect(parse('{"a": -0}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'number', value: '-0', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('-3.14 as value', () => {
      expect(parse('{"a": -3.14}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'number', value: '-3.14', state: 'complete' }]],
        state: 'complete',
      })
    })

    it('scientific notation as value', () => {
      expect(parse('{"a": 1.5e-3}')).toEqual({
        _tag: 'object',
        entries: [['a', { _tag: 'number', value: '1.5e-3', state: 'complete' }]],
        state: 'complete',
      })
    })
  })

  // =========================================================================
  // INCREMENTAL PUSH THEN END
  // =========================================================================
  describe('incremental push then end', () => {
    it('partial object completed by end()', () => {
      const p = createIncrementalJsonParser()
      p.push('{"a": 1')
      expect((p.partial as any)?.state).toBe('incomplete')
      p.end()
      expect((p.partial as any)?.state).toBe('incomplete') // still incomplete — no closing }
    })

    it('complete object before end()', () => {
      const p = createIncrementalJsonParser()
      p.push('{"a": 1}')
      expect((p.partial as any)?.state).toBe('complete')
      p.end()
      expect((p.partial as any)?.state).toBe('complete')
    })
  })
})
