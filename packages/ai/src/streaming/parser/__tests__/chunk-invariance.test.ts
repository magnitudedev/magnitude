import { describe, it, expect } from 'vitest'
import { createIncrementalJsonParser } from '../index'
import type { ParsedValue } from '../../types'

// =========================================================================
// HELPERS
// =========================================================================

/** Parse as single chunk */
function parseSingle(input: string): ParsedValue | undefined {
  const p = createIncrementalJsonParser()
  p.push(input)
  p.end()
  return p.partial
}

/** Parse char-by-char */
function parseCharByChar(input: string): ParsedValue | undefined {
  const p = createIncrementalJsonParser()
  for (const ch of input) p.push(ch)
  p.end()
  return p.partial
}

/** Parse with a specific 2-split at position i */
function parseSplit(input: string, i: number): ParsedValue | undefined {
  const p = createIncrementalJsonParser()
  p.push(input.slice(0, i))
  p.push(input.slice(i))
  p.end()
  return p.partial
}

/** Parse with random chunking */
function parseRandom(input: string, seed: number): ParsedValue | undefined {
  const p = createIncrementalJsonParser()
  let pos = 0
  let s = seed
  while (pos < input.length) {
    // Simple LCG for deterministic "random"
    s = (s * 1103515245 + 12345) & 0x7fffffff
    const chunkSize = Math.max(1, (s % 5) + 1)
    const end = Math.min(pos + chunkSize, input.length)
    p.push(input.slice(pos, end))
    pos = end
  }
  p.end()
  return p.partial
}

/** Assert chunk invariance for a given input */
function assertChunkInvariant(input: string, label?: string) {
  const desc = label || input
  const reference = parseSingle(input)

  // Char-by-char
  expect(parseCharByChar(input), `char-by-char failed for: ${desc}`).toEqual(reference)

  // All 2-splits
  for (let i = 1; i < input.length; i++) {
    expect(parseSplit(input, i), `2-split at ${i} failed for: ${desc}`).toEqual(reference)
  }

  // Random chunking (5 seeds)
  for (let seed = 0; seed < 5; seed++) {
    expect(parseRandom(input, seed), `random seed ${seed} failed for: ${desc}`).toEqual(reference)
  }
}

// =========================================================================
// TESTS
// =========================================================================

describe('Chunk Invariance', () => {
  // The original bug case
  it('{"a": true}', () => assertChunkInvariant('{"a": true}'))
  it('{"a": false}', () => assertChunkInvariant('{"a": false}'))
  it('{"a": null}', () => assertChunkInvariant('{"a": null}'))
  it('{"a": 123}', () => assertChunkInvariant('{"a": 123}'))
  it('{"a": "hello"}', () => assertChunkInvariant('{"a": "hello"}'))

  // Escapes
  it('escape in value', () => assertChunkInvariant('{"a": "he\\nllo"}'))
  it('unicode escape', () => assertChunkInvariant('{"a": "\\u0041"}'))
  it('all escape types', () => assertChunkInvariant('{"a": "\\n\\t\\r\\b\\f\\\\\\"\\/"}'  ))

  // Multiple keys
  it('multiple booleans and null', () => assertChunkInvariant('{"a": true, "b": false, "c": null}'))

  // Arrays
  it('array of mixed types', () => assertChunkInvariant('[true, false, null, 123, "hi"]'))

  // Nesting
  it('nested objects', () => assertChunkInvariant('{"a": {"b": {"c": 1}}}'))
  it('object with array value', () => assertChunkInvariant('{"a": [1, 2, 3]}'))
  it('array of objects', () => assertChunkInvariant('[{"a": 1}, {"b": 2}]'))

  // Keys with spaces
  it('key with spaces', () => assertChunkInvariant('{"key with spaces": "value"}'))

  // String with multiple escapes
  it('string with escapes', () => assertChunkInvariant('{"a": "line1\\nline2\\ttab"}'))
  it('escaped quotes', () => assertChunkInvariant('{"a": "quote: \\"hello\\""}'))

  // Numbers
  it('scientific notation', () => assertChunkInvariant('{"a": 1.23e+10}'))
  it('negative float', () => assertChunkInvariant('{"a": -0.5}'))

  // Empty containers
  it('empty string', () => assertChunkInvariant('""'))
  it('empty object', () => assertChunkInvariant('{}'))
  it('empty array', () => assertChunkInvariant('[]'))
  it('empty string value', () => assertChunkInvariant('{"a": ""}'))
  it('empty array value', () => assertChunkInvariant('{"a": []}'))
  it('empty object value', () => assertChunkInvariant('{"a": {}}'))

  // Multiple consecutive escapes
  it('multiple backslashes', () => assertChunkInvariant('{"a": "\\\\\\\\"}'))

  // Unicode surrogate pair
  it('surrogate pair', () => assertChunkInvariant('{"a": "\\uD83D\\uDE00"}'))

  // Deeply nested
  it('10 levels deep', () => {
    const input = '{"a":'.repeat(10) + '1' + '}'.repeat(10)
    assertChunkInvariant(input, '10 levels deep')
  })

  // Wide object
  it('wide object (20 keys)', () => {
    const entries = Array.from({ length: 20 }, (_, i) => `"k${i}": ${i}`)
    const input = '{' + entries.join(', ') + '}'
    assertChunkInvariant(input, 'wide object')
  })

  // Large array
  it('array of 15 mixed elements', () => {
    const input = '[1, "two", true, false, null, 3.14, "hello", -42, 0, "\\n", [], {}, [1], {"a": 1}, "end"]'
    assertChunkInvariant(input, 'large mixed array')
  })

  // Realistic tool call payload
  it('realistic tool call payload', () => {
    const input = JSON.stringify({
      workerId: "get-date-demo",
      message: "Please run `date` in the shell and report the output back to me.",
      yield: true,
    })
    assertChunkInvariant(input, 'tool call payload')
  })

  // Every JSON number format
  describe('number formats', () => {
    it('0', () => assertChunkInvariant('{"a": 0}'))
    it('-0', () => assertChunkInvariant('{"a": -0}'))
    it('0.0', () => assertChunkInvariant('{"a": 0.0}'))
    it('-0.0', () => assertChunkInvariant('{"a": -0.0}'))
    it('1e0', () => assertChunkInvariant('{"a": 1e0}'))
    it('1E0', () => assertChunkInvariant('{"a": 1E0}'))
    it('1e+0', () => assertChunkInvariant('{"a": 1e+0}'))
    it('1e-0', () => assertChunkInvariant('{"a": 1e-0}'))
    it('1.0e1', () => assertChunkInvariant('{"a": 1.0e1}'))
    it('-1.5e-10', () => assertChunkInvariant('{"a": -1.5e-10}'))
    it('999', () => assertChunkInvariant('{"a": 999}'))
    it('-999.999', () => assertChunkInvariant('{"a": -999.999}'))
    it('1.23456789e+100', () => assertChunkInvariant('{"a": 1.23456789e+100}'))
  })

  // Objects with every value type
  describe('objects with every value type', () => {
    it('all types in one object', () => {
      assertChunkInvariant('{"s":"hello","n":42,"t":true,"f":false,"z":null,"o":{"x":1},"a":[1,2]}')
    })
  })

  // Deeply nested arrays
  describe('deeply nested arrays', () => {
    it('4 levels of nested arrays', () => {
      assertChunkInvariant('[[[[1, 2], [3]], [[4]]], [[[5]]]]')
    })
  })

  // Large payload
  describe('large payloads', () => {
    it('50+ key object', () => {
      const entries = Array.from({ length: 55 }, (_, i) => `"key${i}": ${i}`)
      const input = '{' + entries.join(', ') + '}'
      assertChunkInvariant(input, '55-key object')
    })
  })

  // Every 2-char escape sequence
  describe('all escape sequences', () => {
    it('every 2-char escape', () => {
      assertChunkInvariant('{"a": "\\n\\t\\r\\b\\f\\\\\\/\\""}')
    })
  })

  // Unicode escapes at every split point
  describe('unicode escape split points', () => {
    it('\\u0041 chunk invariance', () => assertChunkInvariant('{"a": "\\u0041"}'))
    it('\\uD83D\\uDE00 chunk invariance', () => assertChunkInvariant('{"a": "\\uD83D\\uDE00"}'))
    it('multiple unicode escapes', () => assertChunkInvariant('{"a": "\\u0041\\u0042\\u0043"}'))
  })

  // Real tool call payloads
  describe('real-world payloads', () => {
    it('shell command with JSON', () => {
      const input = JSON.stringify({
        command: 'echo \'{"key": "value"}\'',
        timeout: 30,
        cwd: '/home/user',
      })
      assertChunkInvariant(input, 'shell command payload')
    })

    it('file edit with code content', () => {
      const input = JSON.stringify({
        path: 'src/index.ts',
        old: 'const x = 1;\nconst y = 2;',
        new: 'const x = 10;\nconst y = 20;\nconst z = 30;',
      })
      assertChunkInvariant(input, 'file edit payload')
    })

    it('message with special chars', () => {
      const input = JSON.stringify({
        workerId: "worker-1",
        message: "Line 1\nLine 2\tTabbed\r\nWindows line\b\fSpecial",
        yield: true,
      })
      assertChunkInvariant(input, 'message with special chars')
    })
  })

  // Payloads with newlines/tabs/special chars
  describe('payloads with special chars in values', () => {
    it('newlines in string value', () => assertChunkInvariant('{"a": "line1\\nline2\\nline3"}'))
    it('tabs in string value', () => assertChunkInvariant('{"a": "col1\\tcol2\\tcol3"}'))
    it('mixed special chars', () => assertChunkInvariant('{"a": "\\n\\t\\r\\b\\f"}'))
  })

  // Adjacent keywords
  describe('adjacent keywords', () => {
    it('[true,false,null,true]', () => assertChunkInvariant('[true,false,null,true]'))
    it('[false,true,null,false,true]', () => assertChunkInvariant('[false,true,null,false,true]'))
  })

  // Adjacent numbers
  describe('adjacent numbers', () => {
    it('[1,2,3,4,5,6,7,8,9,10]', () => assertChunkInvariant('[1,2,3,4,5,6,7,8,9,10]'))
    it('[-1,0,1,-0,0.5,-0.5]', () => assertChunkInvariant('[-1,0,1,-0,0.5,-0.5]'))
  })

  // Mixed complex
  describe('mixed complex', () => {
    it('mixed types object', () => {
      assertChunkInvariant('{"a":1,"b":"two","c":true,"d":null,"e":[1,2],"f":{"g":3}}')
    })

    it('array of mixed objects', () => {
      assertChunkInvariant('[{"a":1,"b":"x"},{"a":2,"b":"y"},{"a":3,"b":"z"}]')
    })

    it('deeply nested mixed', () => {
      assertChunkInvariant('{"a":{"b":[{"c":true},{"d":[1,2,3]},{"e":{"f":"g"}}]}}')
    })
  })

  // The exact regression cases
  describe('regression: original bug', () => {
    it('true then } in separate chunks produces correct result', () => {
      const ref = parseSingle('{"a": true}')
      const p = createIncrementalJsonParser()
      p.push('{"a": true')
      p.push('}')
      p.end()
      expect(p.partial).toEqual(ref)
    })

    it('false then } in separate chunks', () => {
      const ref = parseSingle('{"a": false}')
      const p = createIncrementalJsonParser()
      p.push('{"a": false')
      p.push('}')
      p.end()
      expect(p.partial).toEqual(ref)
    })

    it('null then } in separate chunks', () => {
      const ref = parseSingle('{"a": null}')
      const p = createIncrementalJsonParser()
      p.push('{"a": null')
      p.push('}')
      p.end()
      expect(p.partial).toEqual(ref)
    })

    it('number then } in separate chunks', () => {
      const ref = parseSingle('{"a": 123}')
      const p = createIncrementalJsonParser()
      p.push('{"a": 123')
      p.push('}')
      p.end()
      expect(p.partial).toEqual(ref)
    })

    it('number then ] in separate chunks', () => {
      const ref = parseSingle('[123]')
      const p = createIncrementalJsonParser()
      p.push('[123')
      p.push(']')
      p.end()
      expect(p.partial).toEqual(ref)
    })

    it('true is not concatenated with }', () => {
      // The exact bug: true + } should NOT produce "true}" string
      const p = createIncrementalJsonParser()
      p.push('{"a": true')
      p.push('}')
      p.end()
      const result = p.partial as any
      expect(result._tag).toBe('object')
      expect(result.entries[0][1]._tag).toBe('boolean')
      expect(result.entries[0][1].value).toBe(true)
    })
  })

  // =========================================================================
  // STANDALONE SCALARS
  // =========================================================================
  describe('standalone scalars', () => {
    it('true', () => assertChunkInvariant('true'))
    it('false', () => assertChunkInvariant('false'))
    it('null', () => assertChunkInvariant('null'))
    it('42', () => assertChunkInvariant('42'))
    it('-3.14', () => assertChunkInvariant('-3.14'))
    it('"hello"', () => assertChunkInvariant('"hello"'))
    it('""', () => assertChunkInvariant('""'))
  })

  // =========================================================================
  // INCOMPLETE INPUTS
  // =========================================================================
  describe('incomplete inputs', () => {
    it('{"a":', () => assertChunkInvariant('{"a":'))
    it('[1, 2,', () => assertChunkInvariant('[1, 2,'))
    it('{"a": "hel', () => assertChunkInvariant('{"a": "hel'))
    it('[true, fal', () => assertChunkInvariant('[true, fal'))
  })

  // =========================================================================
  // UNQUOTED KEY OBJECTS
  // =========================================================================
  describe('unquoted key objects', () => {
    it('{a: 1}', () => assertChunkInvariant('{a: 1}'))
    it('{a: 1, b: "two"}', () => assertChunkInvariant('{a: 1, b: "two"}'))
  })

  // =========================================================================
  // WHITESPACE VARIATIONS
  // =========================================================================
  describe('whitespace variations', () => {
    it('{ "a" : 1 }', () => assertChunkInvariant('{ "a" : 1 }'))
    it('[\\n1\\n,\\n2\\n]', () => assertChunkInvariant('[\n1\n,\n2\n]'))
  })

  // =========================================================================
  // CONSECUTIVE EMPTY CONTAINERS
  // =========================================================================
  describe('consecutive empty containers', () => {
    it('[{}, {}, []]', () => assertChunkInvariant('[{}, {}, []]'))
  })

  // =========================================================================
  // NESTED INCOMPLETE
  // =========================================================================
  describe('nested incomplete', () => {
    it('{"a": {"b": {"c":', () => assertChunkInvariant('{"a": {"b": {"c":'))
  })

  // =========================================================================
  // LONG STRING WITH MANY ESCAPES
  // =========================================================================
  describe('long string with many escapes', () => {
    it('string with escapes throughout', () => {
      assertChunkInvariant('{"a": "line1\\nline2\\ttab\\rret\\\\back\\"quote\\/slash\\u0041end"}')
    })
  })

  // =========================================================================
  // OBJECT WHERE EVERY VALUE IS A DIFFERENT TYPE
  // =========================================================================
  describe('every value type in one object', () => {
    it('all types', () => {
      assertChunkInvariant('{"s":"str","n":42,"f":3.14,"t":true,"b":false,"z":null,"o":{},"a":[]}')
    })
  })

  // =========================================================================
  // ALL NULLS ARRAY
  // =========================================================================
  describe('all nulls array', () => {
    it('[null, null, null, null]', () => assertChunkInvariant('[null, null, null, null]'))
  })

  // =========================================================================
  // EMPTY KEY
  // =========================================================================
  describe('empty key', () => {
    it('{"": "empty key"}', () => assertChunkInvariant('{"": "empty key"}'))
  })

  // =========================================================================
  // SINGLE CHARACTER INPUTS
  // =========================================================================
  describe('single character inputs', () => {
    it('{', () => assertChunkInvariant('{'))
    it('[', () => assertChunkInvariant('['))
    it('"', () => assertChunkInvariant('"'))
    it('1', () => assertChunkInvariant('1'))
    it('t', () => assertChunkInvariant('t'))
  })

  // =========================================================================
  // JUST WHITESPACE
  // =========================================================================
  describe('just whitespace', () => {
    it('   ', () => assertChunkInvariant('   '))
  })

  // =========================================================================
  // MORE COMPLEX STRUCTURES
  // =========================================================================
  describe('complex structures', () => {
    it('{"a": [true, false, null]}', () => assertChunkInvariant('{"a": [true, false, null]}'))
    it('[{"a": 1, "b": 2}, {"c": 3}]', () => assertChunkInvariant('[{"a": 1, "b": 2}, {"c": 3}]'))
    it('nested object with mixed array', () => assertChunkInvariant('{"nested": {"array": [1, "two", true, null, {"deep": false}]}}'))
    it('deeply nested arrays: [1, [2, [3, [4, [5]]]]]', () => assertChunkInvariant('[1, [2, [3, [4, [5]]]]]'))
    it('mixed escapes and types', () => assertChunkInvariant('{"a": "hello\\nworld", "b": 42, "c": true}'))
    it('escaped quotes and unicode in array', () => assertChunkInvariant('["\\"quoted\\"", "\\u0041\\u0042"]'))
    it('long string value', () => assertChunkInvariant('{"long": "abcdefghijklmnopqrstuvwxyz0123456789"}'))
    it('multi-level mixed nesting', () => assertChunkInvariant('{"multi": {"a": 1, "b": [2, 3], "c": {"d": 4}}}'))
    it('all number formats in array', () => assertChunkInvariant('[0, -0, 0.0, -0.0, 1e0, 1E0, 1e+0, 1e-0]'))
    it('emoji surrogate pairs', () => assertChunkInvariant('{"emoji": "\\uD83D\\uDE00\\uD83D\\uDE01"}'))
  })

  // =========================================================================
  // EDGE CASE SIMPLE INPUTS
  // =========================================================================
  describe('edge case simple inputs', () => {
    it('just 0', () => assertChunkInvariant('0'))
    it('just ""', () => assertChunkInvariant('""'))
    it('just {}', () => assertChunkInvariant('{}'))
    it('just []', () => assertChunkInvariant('[]'))
    it('[[], [], []]', () => assertChunkInvariant('[[], [], []]'))
    it('5 levels of nested objects', () => assertChunkInvariant('{"a": {"b": {"c": {"d": {"e": 1}}}}}'))
  })

  // =========================================================================
  // ARRAYS OF ARRAYS
  // =========================================================================
  describe('arrays of arrays', () => {
    it('[[], [], []]', () => assertChunkInvariant('[[], [], []]'))
    it('[[1], [2], [3]]', () => assertChunkInvariant('[[1], [2], [3]]'))
    it('[[1, 2], [3, 4], [5, 6]]', () => assertChunkInvariant('[[1, 2], [3, 4], [5, 6]]'))
    it('[[[]], [[]]]', () => assertChunkInvariant('[[[]], [[]]]'))
  })

  // =========================================================================
  // OBJECTS WITH VARIOUS VALUE COMBINATIONS
  // =========================================================================
  describe('objects with various value combos', () => {
    it('string + number', () => assertChunkInvariant('{"name": "alice", "age": 30}'))
    it('bool + null + array', () => assertChunkInvariant('{"ok": true, "err": null, "data": [1, 2]}'))
    it('nested object + string', () => assertChunkInvariant('{"meta": {"v": 1}, "id": "abc"}'))
    it('array of strings', () => assertChunkInvariant('{"tags": ["a", "b", "c"]}'))
    it('array of booleans', () => assertChunkInvariant('{"flags": [true, false, true, false]}'))
    it('array of nulls', () => assertChunkInvariant('{"empty": [null, null]}'))
    it('mixed nested', () => assertChunkInvariant('{"a": [{"b": [1, 2]}, {"c": [3, 4]}]}'))
  })

  // =========================================================================
  // STRINGS WITH SPECIAL CONTENT
  // =========================================================================
  describe('strings with special content', () => {
    it('string with only spaces', () => assertChunkInvariant('{"a": "   "}'))
    it('string with newlines', () => assertChunkInvariant('{"a": "\\n\\n\\n"}'))
    it('string with tabs', () => assertChunkInvariant('{"a": "\\t\\t"}'))
    it('string with mixed whitespace escapes', () => assertChunkInvariant('{"a": "\\n\\t\\r\\n"}'))
    it('string with numbers', () => assertChunkInvariant('{"a": "12345"}'))
    it('string with braces', () => assertChunkInvariant('{"a": "{[]}"}'))
    it('string with colons and commas', () => assertChunkInvariant('{"a": "k:v,k:v"}'))
    it('string that looks like JSON', () => assertChunkInvariant('{"a": "{\\"b\\": 1}"}'))
  })

  // =========================================================================
  // NUMBERS IN VARIOUS CONTEXTS
  // =========================================================================
  describe('numbers in various contexts', () => {
    it('negative in array', () => assertChunkInvariant('[-1, -2, -3]'))
    it('floats in array', () => assertChunkInvariant('[0.1, 0.2, 0.3]'))
    it('exponents in array', () => assertChunkInvariant('[1e5, 2E10, 3e-1]'))
    it('mixed number formats', () => assertChunkInvariant('[0, -0, 1, -1, 0.5, -0.5, 1e2, 1.5e-3]'))
    it('number as only object value', () => assertChunkInvariant('{"x": 999999}'))
  })

  // =========================================================================
  // TRAILING COMMAS
  // =========================================================================
  describe('trailing commas chunk invariance', () => {
    it('trailing comma in object', () => assertChunkInvariant('{"a": 1,}'))
    it('trailing comma in array', () => assertChunkInvariant('[1, 2,]'))
    it('trailing comma in nested', () => assertChunkInvariant('{"a": [1,],}'))
  })

  // =========================================================================
  // REALISTIC PAYLOADS
  // =========================================================================
  describe('more realistic payloads', () => {
    it('spawn worker payload', () => {
      assertChunkInvariant(JSON.stringify({
        workerId: "build-feature",
        profile: "engineer",
        task: "Implement the login page",
      }), 'spawn worker')
    })

    it('file read payload', () => {
      assertChunkInvariant(JSON.stringify({
        path: "src/components/App.tsx",
        offset: 1,
        limit: 100,
      }), 'file read')
    })

    it('grep payload', () => {
      assertChunkInvariant(JSON.stringify({
        pattern: "TODO|FIXME|HACK",
        glob: "*.ts",
        path: "src/",
      }), 'grep')
    })

    it('complex nested response', () => {
      assertChunkInvariant(JSON.stringify({
        status: "ok",
        data: {
          items: [
            { id: 1, name: "first", tags: ["a", "b"] },
            { id: 2, name: "second", tags: [] },
          ],
          total: 2,
          hasMore: false,
        },
        meta: null,
      }), 'complex response')
    })

    it('deeply nested config', () => {
      assertChunkInvariant(JSON.stringify({
        server: {
          host: "localhost",
          port: 3000,
          ssl: {
            enabled: true,
            cert: "/path/to/cert.pem",
            key: "/path/to/key.pem",
          },
        },
        database: {
          url: "postgres://localhost:5432/db",
          pool: { min: 2, max: 10 },
        },
      }), 'config payload')
    })
  })

  // =========================================================================
  // LARGE ARRAYS
  // =========================================================================
  describe('large arrays chunk invariance', () => {
    it('array of 30 numbers', () => {
      const input = '[' + Array.from({ length: 30 }, (_, i) => String(i)).join(',') + ']'
      assertChunkInvariant(input, '30 numbers')
    })

    it('array of 20 strings', () => {
      const input = '[' + Array.from({ length: 20 }, (_, i) => `"item${i}"`).join(',') + ']'
      assertChunkInvariant(input, '20 strings')
    })

    it('array of 10 objects', () => {
      const items = Array.from({ length: 10 }, (_, i) => `{"id":${i},"v":"${String.fromCharCode(97 + i)}"}`)
      const input = '[' + items.join(',') + ']'
      assertChunkInvariant(input, '10 objects')
    })
  })

  // =========================================================================
  // ADDITIONAL EDGE CASES
  // =========================================================================
  describe('additional edge cases', () => {
    it('object with empty string key and value', () => assertChunkInvariant('{"": ""}'))
    it('array with single true', () => assertChunkInvariant('[true]'))
    it('array with single false', () => assertChunkInvariant('[false]'))
    it('array with single null', () => assertChunkInvariant('[null]'))
    it('array with single string', () => assertChunkInvariant('["hello"]'))
    it('array with single number', () => assertChunkInvariant('[42]'))
    it('array with single object', () => assertChunkInvariant('[{"a": 1}]'))
    it('array with single array', () => assertChunkInvariant('[[1]]'))
    it('object with all falsy values', () => assertChunkInvariant('{"a": false, "b": null, "c": 0, "d": ""}'))
    it('very long key', () => assertChunkInvariant('{"abcdefghijklmnopqrstuvwxyz": 1}'))
  })
})
