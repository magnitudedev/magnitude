import { describe, it, expect } from 'vitest'
import { createJsonTokenizer } from '../tokenizer'
import type { JsonToken } from '../types'

/** Collect all tokens from pushing chunks through a fresh tokenizer */
function tokenize(...chunks: string[]): JsonToken[] {
  const tokens: JsonToken[] = []
  const t = createJsonTokenizer((tok) => tokens.push(tok))
  for (const chunk of chunks) t.push(chunk)
  t.end()
  return tokens
}

/** Collect tokens without calling end() */
function tokenizeNoEnd(...chunks: string[]): JsonToken[] {
  const tokens: JsonToken[] = []
  const t = createJsonTokenizer((tok) => tokens.push(tok))
  for (const chunk of chunks) t.push(chunk)
  return tokens
}

/** Get pending state after pushing chunks (no end) */
function getPending(...chunks: string[]) {
  const t = createJsonTokenizer(() => {})
  for (const chunk of chunks) t.push(chunk)
  return t.pending
}

describe('JsonTokenizer', () => {
  // =========================================================================
  // STRING TOKENS
  // =========================================================================
  describe('strings', () => {
    it('simple string', () => {
      expect(tokenize('"hello"')).toEqual([
        { _tag: 'string', value: 'hello', complete: true },
      ])
    })

    it('empty string', () => {
      expect(tokenize('""')).toEqual([
        { _tag: 'string', value: '', complete: true },
      ])
    })

    it('string with newline escape', () => {
      expect(tokenize('"he\\nllo"')).toEqual([
        { _tag: 'string', value: 'he\nllo', complete: true },
      ])
    })

    it('string with tab escape', () => {
      expect(tokenize('"a\\tb"')).toEqual([
        { _tag: 'string', value: 'a\tb', complete: true },
      ])
    })

    it('string with carriage return escape', () => {
      expect(tokenize('"a\\rb"')).toEqual([
        { _tag: 'string', value: 'a\rb', complete: true },
      ])
    })

    it('string with backspace escape', () => {
      expect(tokenize('"a\\bb"')).toEqual([
        { _tag: 'string', value: 'a\bb', complete: true },
      ])
    })

    it('string with form feed escape', () => {
      expect(tokenize('"a\\fb"')).toEqual([
        { _tag: 'string', value: 'a\fb', complete: true },
      ])
    })

    it('string with escaped backslash', () => {
      expect(tokenize('"a\\\\b"')).toEqual([
        { _tag: 'string', value: 'a\\b', complete: true },
      ])
    })

    it('string with escaped quote', () => {
      expect(tokenize('"he said \\"hi\\""')).toEqual([
        { _tag: 'string', value: 'he said "hi"', complete: true },
      ])
    })

    it('string with escaped forward slash', () => {
      expect(tokenize('"a\\/b"')).toEqual([
        { _tag: 'string', value: 'a/b', complete: true },
      ])
    })

    it('unicode escape \\u0041 → A', () => {
      expect(tokenize('"\\u0041"')).toEqual([
        { _tag: 'string', value: 'A', complete: true },
      ])
    })

    it('unicode surrogate pair \\uD83D\\uDE00', () => {
      expect(tokenize('"\\uD83D\\uDE00"')).toEqual([
        { _tag: 'string', value: '😀', complete: true },
      ])
    })

    it('invalid unicode \\uZZZZ preserved', () => {
      expect(tokenize('"\\uZZZZ"')).toEqual([
        { _tag: 'string', value: '\\uZZZZ', complete: true },
      ])
    })

    it('string with all escape types combined', () => {
      expect(tokenize('"\\n\\t\\r\\b\\f\\\\\\"\\/"')).toEqual([
        { _tag: 'string', value: '\n\t\r\b\f\\"/'.toString(), complete: true },
      ])
    })

    // Chunk boundary tests for strings
    it('backslash at chunk boundary', () => {
      expect(tokenize('"a\\', 'nb"')).toEqual([
        { _tag: 'string', value: 'a\nb', complete: true },
      ])
    })

    it('\\u at chunk boundary', () => {
      expect(tokenize('"\\u', '0041"')).toEqual([
        { _tag: 'string', value: 'A', complete: true },
      ])
    })

    it('\\u00 at chunk boundary', () => {
      expect(tokenize('"\\u00', '41"')).toEqual([
        { _tag: 'string', value: 'A', complete: true },
      ])
    })

    it('closing quote at chunk boundary', () => {
      expect(tokenize('"hello', '"')).toEqual([
        { _tag: 'string', value: 'hello', complete: true },
      ])
    })

    it('opening quote alone in chunk', () => {
      expect(tokenize('"', 'hello"')).toEqual([
        { _tag: 'string', value: 'hello', complete: true },
      ])
    })

    it('each char in separate chunk', () => {
      expect(tokenize('"', 'a', 'b', '"')).toEqual([
        { _tag: 'string', value: 'ab', complete: true },
      ])
    })

    it('incomplete string at end', () => {
      expect(tokenize('"hello')).toEqual([
        { _tag: 'string', value: 'hello', complete: false },
      ])
    })

    it('pending state for incomplete string', () => {
      expect(getPending('"hello')).toEqual({ _tag: 'string', content: 'hello' })
    })

    it('pending state for string with pending escape', () => {
      expect(getPending('"hello\\')).toEqual({ _tag: 'string', content: 'hello' })
    })

    it('pending state for string with pending unicode', () => {
      expect(getPending('"\\u00')).toEqual({ _tag: 'string', content: '' })
    })

    it('double backslash at chunk boundary', () => {
      expect(tokenize('"\\\\', '"')).toEqual([
        { _tag: 'string', value: '\\', complete: true },
      ])
    })

    it('escaped quote does not close string', () => {
      expect(tokenize('"\\"', '"')).toEqual([
        { _tag: 'string', value: '"', complete: true },
      ])
    })

    it('backslash at end of string with chunk split', () => {
      // "a\" + "b" should parse as string containing a\b? No — \" is escaped quote
      // "a\\" + "b" should parse as string "a\" then leftover b
      expect(tokenize('"a\\\\', 'b"')).toEqual([
        { _tag: 'string', value: 'a\\b', complete: true },
      ])
    })
  })

  // =========================================================================
  // NUMBER TOKENS
  // =========================================================================
  describe('numbers', () => {
    it('integer', () => {
      expect(tokenize('42')).toEqual([
        { _tag: 'number', value: '42', complete: true },
      ])
    })

    it('negative integer', () => {
      expect(tokenize('-42')).toEqual([
        { _tag: 'number', value: '-42', complete: true },
      ])
    })

    it('float', () => {
      expect(tokenize('3.14')).toEqual([
        { _tag: 'number', value: '3.14', complete: true },
      ])
    })

    it('exponent lowercase', () => {
      expect(tokenize('1e10')).toEqual([
        { _tag: 'number', value: '1e10', complete: true },
      ])
    })

    it('exponent uppercase', () => {
      expect(tokenize('1E10')).toEqual([
        { _tag: 'number', value: '1E10', complete: true },
      ])
    })

    it('exponent with plus', () => {
      expect(tokenize('1e+10')).toEqual([
        { _tag: 'number', value: '1e+10', complete: true },
      ])
    })

    it('exponent with minus', () => {
      expect(tokenize('1e-10')).toEqual([
        { _tag: 'number', value: '1e-10', complete: true },
      ])
    })

    it('leading zero float', () => {
      expect(tokenize('0.5')).toEqual([
        { _tag: 'number', value: '0.5', complete: true },
      ])
    })

    it('zero', () => {
      expect(tokenize('0')).toEqual([
        { _tag: 'number', value: '0', complete: true },
      ])
    })

    it('negative zero', () => {
      expect(tokenize('-0')).toEqual([
        { _tag: 'number', value: '-0', complete: true },
      ])
    })

    it('complex float', () => {
      expect(tokenize('1.5e-3')).toEqual([
        { _tag: 'number', value: '1.5e-3', complete: true },
      ])
    })

    // Incomplete numbers
    it('incomplete: minus alone', () => {
      expect(tokenize('-')).toEqual([
        { _tag: 'number', value: '-', complete: false },
      ])
    })

    it('incomplete: trailing decimal point', () => {
      expect(tokenize('1.')).toEqual([
        { _tag: 'number', value: '1.', complete: false },
      ])
    })

    it('incomplete: trailing exponent marker', () => {
      expect(tokenize('1e')).toEqual([
        { _tag: 'number', value: '1e', complete: false },
      ])
    })

    it('incomplete: trailing exponent sign', () => {
      expect(tokenize('1e+')).toEqual([
        { _tag: 'number', value: '1e+', complete: false },
      ])
    })

    it('incomplete: trailing exponent sign minus', () => {
      expect(tokenize('1e-')).toEqual([
        { _tag: 'number', value: '1e-', complete: false },
      ])
    })

    // Chunk boundary tests for numbers
    it('chunk split: minus then digits', () => {
      expect(tokenize('-', '42')).toEqual([
        { _tag: 'number', value: '-42', complete: true },
      ])
    })

    it('chunk split: digits then decimal then digits', () => {
      expect(tokenize('3.', '14')).toEqual([
        { _tag: 'number', value: '3.14', complete: true },
      ])
    })

    it('chunk split: digits then exponent then digits', () => {
      expect(tokenize('1e', '10')).toEqual([
        { _tag: 'number', value: '1e10', complete: true },
      ])
    })

    it('chunk split: mid-digit', () => {
      expect(tokenize('4', '2')).toEqual([
        { _tag: 'number', value: '42', complete: true },
      ])
    })

    it('number terminated by structural char', () => {
      expect(tokenize('42}')).toEqual([
        { _tag: 'number', value: '42', complete: true },
        { _tag: 'objectClose' },
      ])
    })

    it('number terminated by structural char across chunks', () => {
      expect(tokenize('42', '}')).toEqual([
        { _tag: 'number', value: '42', complete: true },
        { _tag: 'objectClose' },
      ])
    })

    it('number terminated by comma', () => {
      expect(tokenize('42,')).toEqual([
        { _tag: 'number', value: '42', complete: true },
        { _tag: 'comma' },
      ])
    })

    it('number terminated by whitespace then structural', () => {
      expect(tokenize('42 }')).toEqual([
        { _tag: 'number', value: '42', complete: true },
        { _tag: 'objectClose' },
      ])
    })

    it('pending state for incomplete number', () => {
      expect(getPending('42')).toEqual({ _tag: 'number', content: '42' })
    })
  })

  // =========================================================================
  // KEYWORD TOKENS
  // =========================================================================
  describe('keywords', () => {
    it('true', () => {
      expect(tokenize('true')).toEqual([{ _tag: 'true' }])
    })

    it('false', () => {
      expect(tokenize('false')).toEqual([{ _tag: 'false' }])
    })

    it('null', () => {
      expect(tokenize('null')).toEqual([{ _tag: 'null' }])
    })

    // Every chunk split for true
    it('true: t + rue', () => {
      expect(tokenize('t', 'rue')).toEqual([{ _tag: 'true' }])
    })

    it('true: tr + ue', () => {
      expect(tokenize('tr', 'ue')).toEqual([{ _tag: 'true' }])
    })

    it('true: tru + e', () => {
      expect(tokenize('tru', 'e')).toEqual([{ _tag: 'true' }])
    })

    // Every chunk split for false
    it('false: f + alse', () => {
      expect(tokenize('f', 'alse')).toEqual([{ _tag: 'false' }])
    })

    it('false: fa + lse', () => {
      expect(tokenize('fa', 'lse')).toEqual([{ _tag: 'false' }])
    })

    it('false: fal + se', () => {
      expect(tokenize('fal', 'se')).toEqual([{ _tag: 'false' }])
    })

    it('false: fals + e', () => {
      expect(tokenize('fals', 'e')).toEqual([{ _tag: 'false' }])
    })

    // Every chunk split for null
    it('null: n + ull', () => {
      expect(tokenize('n', 'ull')).toEqual([{ _tag: 'null' }])
    })

    it('null: nu + ll', () => {
      expect(tokenize('nu', 'll')).toEqual([{ _tag: 'null' }])
    })

    it('null: nul + l', () => {
      expect(tokenize('nul', 'l')).toEqual([{ _tag: 'null' }])
    })

    // Keyword terminated by structural char
    it('true terminated by }', () => {
      expect(tokenize('true}')).toEqual([
        { _tag: 'true' },
        { _tag: 'objectClose' },
      ])
    })

    it('true terminated by } across chunks', () => {
      expect(tokenize('true', '}')).toEqual([
        { _tag: 'true' },
        { _tag: 'objectClose' },
      ])
    })

    it('false terminated by } across chunks', () => {
      expect(tokenize('false', '}')).toEqual([
        { _tag: 'false' },
        { _tag: 'objectClose' },
      ])
    })

    it('null terminated by } across chunks', () => {
      expect(tokenize('null', '}')).toEqual([
        { _tag: 'null' },
        { _tag: 'objectClose' },
      ])
    })

    it('true terminated by ]', () => {
      expect(tokenize('true]')).toEqual([
        { _tag: 'true' },
        { _tag: 'arrayClose' },
      ])
    })

    it('true terminated by ] across chunks', () => {
      expect(tokenize('true', ']')).toEqual([
        { _tag: 'true' },
        { _tag: 'arrayClose' },
      ])
    })

    it('true terminated by comma', () => {
      expect(tokenize('true,')).toEqual([
        { _tag: 'true' },
        { _tag: 'comma' },
      ])
    })

    it('true terminated by comma across chunks', () => {
      expect(tokenize('true', ',')).toEqual([
        { _tag: 'true' },
        { _tag: 'comma' },
      ])
    })

    it('true terminated by whitespace', () => {
      expect(tokenize('true ')).toEqual([{ _tag: 'true' }])
    })

    // Keyword prefix becomes unquoted
    it('truthy becomes unquoted string', () => {
      expect(tokenize('truthy')).toEqual([
        { _tag: 'unquotedString', value: 'truthy', complete: true },
      ])
    })

    it('falsey becomes unquoted string', () => {
      expect(tokenize('falsey')).toEqual([
        { _tag: 'unquotedString', value: 'falsey', complete: true },
      ])
    })

    it('nullable becomes unquoted string', () => {
      expect(tokenize('nullable')).toEqual([
        { _tag: 'unquotedString', value: 'nullable', complete: true },
      ])
    })

    it('truethy across chunks: true + thy', () => {
      expect(tokenize('true', 'thy')).toEqual([
        { _tag: 'unquotedString', value: 'truethy', complete: true },
      ])
    })

    // Pending state
    it('pending state for partial keyword tru', () => {
      expect(getPending('tru')).toEqual({ _tag: 'keyword', content: 'tru' })
    })

    it('pending state for completed keyword waiting for delimiter', () => {
      // "true" fully matched but no delimiter yet — still pending
      expect(getPending('true')).toEqual({ _tag: 'keyword', content: 'true' })
    })
  })

  // =========================================================================
  // STRUCTURAL TOKENS
  // =========================================================================
  describe('structural tokens', () => {
    it('object open', () => {
      expect(tokenize('{')).toEqual([{ _tag: 'objectOpen' }])
    })

    it('object close', () => {
      expect(tokenize('}')).toEqual([{ _tag: 'objectClose' }])
    })

    it('array open', () => {
      expect(tokenize('[')).toEqual([{ _tag: 'arrayOpen' }])
    })

    it('array close', () => {
      expect(tokenize(']')).toEqual([{ _tag: 'arrayClose' }])
    })

    it('colon', () => {
      expect(tokenize(':')).toEqual([{ _tag: 'colon' }])
    })

    it('comma', () => {
      expect(tokenize(',')).toEqual([{ _tag: 'comma' }])
    })

    it('all structural in one chunk', () => {
      expect(tokenize('{[:,]}')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'arrayOpen' },
        { _tag: 'colon' },
        { _tag: 'comma' },
        { _tag: 'arrayClose' },
        { _tag: 'objectClose' },
      ])
    })

    it('each structural char in separate chunk', () => {
      expect(tokenize('{', '[', ':', ',', ']', '}')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'arrayOpen' },
        { _tag: 'colon' },
        { _tag: 'comma' },
        { _tag: 'arrayClose' },
        { _tag: 'objectClose' },
      ])
    })
  })

  // =========================================================================
  // UNQUOTED STRINGS (PERMISSIVE)
  // =========================================================================
  describe('unquoted strings', () => {
    it('bare identifier', () => {
      expect(tokenize('hello')).toEqual([
        { _tag: 'unquotedString', value: 'hello', complete: true },
      ])
    })

    it('unquoted terminated by colon', () => {
      expect(tokenize('key:')).toEqual([
        { _tag: 'unquotedString', value: 'key', complete: true },
        { _tag: 'colon' },
      ])
    })

    it('unquoted terminated by colon across chunks', () => {
      expect(tokenize('key', ':')).toEqual([
        { _tag: 'unquotedString', value: 'key', complete: true },
        { _tag: 'colon' },
      ])
    })

    it('unquoted at EOF is incomplete', () => {
      expect(tokenize('hello')).toEqual([
        { _tag: 'unquotedString', value: 'hello', complete: true },
      ])
    })

    it('pending state for unquoted', () => {
      expect(getPending('hello')).toEqual({ _tag: 'unquoted', content: 'hello' })
    })

    it('unquoted with mixed chars', () => {
      expect(tokenize('abc_123')).toEqual([
        { _tag: 'unquotedString', value: 'abc_123', complete: true },
      ])
    })
  })

  // =========================================================================
  // WHITESPACE
  // =========================================================================
  describe('whitespace', () => {
    it('spaces between tokens', () => {
      expect(tokenize('{ "a" : 1 }')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'colon' },
        { _tag: 'number', value: '1', complete: true },
        { _tag: 'objectClose' },
      ])
    })

    it('tabs and newlines between tokens', () => {
      expect(tokenize('{\n\t"a"\n:\n1\n}')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'colon' },
        { _tag: 'number', value: '1', complete: true },
        { _tag: 'objectClose' },
      ])
    })

    it('whitespace-only chunk produces no tokens', () => {
      expect(tokenize('   \t\n\r  ')).toEqual([])
    })

    it('whitespace-only chunks between value chunks', () => {
      expect(tokenize('{', '  ', '"a"', '  ', ':', '  ', '1', '  ', '}')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'colon' },
        { _tag: 'number', value: '1', complete: true },
        { _tag: 'objectClose' },
      ])
    })
  })

  // =========================================================================
  // MULTI-CHUNK (3+) SPLITS
  // =========================================================================
  describe('multi-chunk splits', () => {
    it('string split into 3 chunks', () => {
      expect(tokenize('"he', 'll', 'o"')).toEqual([
        { _tag: 'string', value: 'hello', complete: true },
      ])
    })

    it('string split into 4 chunks', () => {
      expect(tokenize('"', 'ab', 'cd', '"')).toEqual([
        { _tag: 'string', value: 'abcd', complete: true },
      ])
    })

    it('number split into 3 chunks: -3.14', () => {
      expect(tokenize('-', '3.', '14')).toEqual([
        { _tag: 'number', value: '-3.14', complete: true },
      ])
    })

    it('number split into 4 chunks: 1.5e-3', () => {
      expect(tokenize('1', '.5', 'e', '-3')).toEqual([
        { _tag: 'number', value: '1.5e-3', complete: true },
      ])
    })

    it('number split into 5 chunks: -1.5e+10', () => {
      expect(tokenize('-', '1', '.5', 'e+', '10')).toEqual([
        { _tag: 'number', value: '-1.5e+10', complete: true },
      ])
    })

    it('keyword true split into 4 chunks', () => {
      expect(tokenize('t', 'r', 'u', 'e')).toEqual([{ _tag: 'true' }])
    })

    it('keyword false split into 5 chunks', () => {
      expect(tokenize('f', 'a', 'l', 's', 'e')).toEqual([{ _tag: 'false' }])
    })

    it('keyword null split into 4 chunks', () => {
      expect(tokenize('n', 'u', 'l', 'l')).toEqual([{ _tag: 'null' }])
    })

    it('unquoted split into 3 chunks terminated by colon', () => {
      expect(tokenize('ke', 'y_', 'name:')).toEqual([
        { _tag: 'unquotedString', value: 'key_name', complete: true },
        { _tag: 'colon' },
      ])
    })
  })

  // =========================================================================
  // ESCAPE BOUNDARY SPLITS
  // =========================================================================
  describe('escape boundary splits', () => {
    it('\\u0041 split as \\u + 00 + 41', () => {
      expect(tokenize('"\\u', '00', '41"')).toEqual([
        { _tag: 'string', value: 'A', complete: true },
      ])
    })

    it('\\u0041 split as \\u0 + 041', () => {
      expect(tokenize('"\\u0', '041"')).toEqual([
        { _tag: 'string', value: 'A', complete: true },
      ])
    })

    it('\\u0041 split as \\u004 + 1', () => {
      expect(tokenize('"\\u004', '1"')).toEqual([
        { _tag: 'string', value: 'A', complete: true },
      ])
    })

    it('\\u0041 split char by char', () => {
      expect(tokenize('"', '\\', 'u', '0', '0', '4', '1', '"')).toEqual([
        { _tag: 'string', value: 'A', complete: true },
      ])
    })

    it('\\n split at backslash', () => {
      expect(tokenize('"a\\', 'nb"')).toEqual([
        { _tag: 'string', value: 'a\nb', complete: true },
      ])
    })

    it('\\t split at backslash', () => {
      expect(tokenize('"a\\', 'tb"')).toEqual([
        { _tag: 'string', value: 'a\tb', complete: true },
      ])
    })

    it('\\r split at backslash', () => {
      expect(tokenize('"a\\', 'rb"')).toEqual([
        { _tag: 'string', value: 'a\rb', complete: true },
      ])
    })

    it('\\b split at backslash', () => {
      expect(tokenize('"a\\', 'bb"')).toEqual([
        { _tag: 'string', value: 'a\bb', complete: true },
      ])
    })

    it('\\f split at backslash', () => {
      expect(tokenize('"a\\', 'fb"')).toEqual([
        { _tag: 'string', value: 'a\fb', complete: true },
      ])
    })

    it('\\\\ split at first backslash', () => {
      expect(tokenize('"a\\', '\\b"')).toEqual([
        { _tag: 'string', value: 'a\\b', complete: true },
      ])
    })

    it('\\" split at backslash', () => {
      expect(tokenize('"a\\', '"b"')).toEqual([
        { _tag: 'string', value: 'a"b', complete: true },
      ])
    })

    it('\\/ split at backslash', () => {
      expect(tokenize('"a\\', '/b"')).toEqual([
        { _tag: 'string', value: 'a/b', complete: true },
      ])
    })

    it('surrogate pair split between surrogates', () => {
      expect(tokenize('"\\uD83D', '\\uDE00"')).toEqual([
        { _tag: 'string', value: '😀', complete: true },
      ])
    })

    it('surrogate pair split mid-second-surrogate', () => {
      expect(tokenize('"\\uD83D\\uDE', '00"')).toEqual([
        { _tag: 'string', value: '😀', complete: true },
      ])
    })
  })

  // =========================================================================
  // STRINGS WITH STRUCTURAL CHARS
  // =========================================================================
  describe('strings containing structural chars', () => {
    it('string containing {', () => {
      expect(tokenize('"{"')).toEqual([
        { _tag: 'string', value: '{', complete: true },
      ])
    })

    it('string containing }', () => {
      expect(tokenize('"}"')).toEqual([
        { _tag: 'string', value: '}', complete: true },
      ])
    })

    it('string containing [', () => {
      expect(tokenize('"["')).toEqual([
        { _tag: 'string', value: '[', complete: true },
      ])
    })

    it('string containing ]', () => {
      expect(tokenize('"]"')).toEqual([
        { _tag: 'string', value: ']', complete: true },
      ])
    })

    it('string containing :', () => {
      expect(tokenize('":"')).toEqual([
        { _tag: 'string', value: ':', complete: true },
      ])
    })

    it('string containing ,', () => {
      expect(tokenize('","')).toEqual([
        { _tag: 'string', value: ',', complete: true },
      ])
    })

    it('string containing all structural chars', () => {
      expect(tokenize('"{[]:,}"')).toEqual([
        { _tag: 'string', value: '{[]:,}', complete: true },
      ])
    })
  })

  // =========================================================================
  // STRINGS WITH KEYWORD-LIKE CONTENT
  // =========================================================================
  describe('strings containing keyword-like content', () => {
    it('string "true"', () => {
      expect(tokenize('"true"')).toEqual([
        { _tag: 'string', value: 'true', complete: true },
      ])
    })

    it('string "false"', () => {
      expect(tokenize('"false"')).toEqual([
        { _tag: 'string', value: 'false', complete: true },
      ])
    })

    it('string "null"', () => {
      expect(tokenize('"null"')).toEqual([
        { _tag: 'string', value: 'null', complete: true },
      ])
    })

    it('string "123"', () => {
      expect(tokenize('"123"')).toEqual([
        { _tag: 'string', value: '123', complete: true },
      ])
    })
  })

  // =========================================================================
  // NUMBERS IN VARIOUS POSITIONS
  // =========================================================================
  describe('numbers in various positions', () => {
    it('number after colon in object', () => {
      expect(tokenize('{"a":42}')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'colon' },
        { _tag: 'number', value: '42', complete: true },
        { _tag: 'objectClose' },
      ])
    })

    it('number in array', () => {
      expect(tokenize('[42]')).toEqual([
        { _tag: 'arrayOpen' },
        { _tag: 'number', value: '42', complete: true },
        { _tag: 'arrayClose' },
      ])
    })

    it('number at root', () => {
      expect(tokenize('42')).toEqual([
        { _tag: 'number', value: '42', complete: true },
      ])
    })

    it('negative number split at - boundary', () => {
      expect(tokenize('-', '42')).toEqual([
        { _tag: 'number', value: '-42', complete: true },
      ])
    })

    it('exponent split at e boundary', () => {
      expect(tokenize('1', 'e10')).toEqual([
        { _tag: 'number', value: '1e10', complete: true },
      ])
    })

    it('exponent split at E boundary', () => {
      expect(tokenize('1', 'E10')).toEqual([
        { _tag: 'number', value: '1E10', complete: true },
      ])
    })

    it('exponent split at + boundary', () => {
      expect(tokenize('1e', '+10')).toEqual([
        { _tag: 'number', value: '1e+10', complete: true },
      ])
    })

    it('exponent split at - boundary', () => {
      expect(tokenize('1e', '-10')).toEqual([
        { _tag: 'number', value: '1e-10', complete: true },
      ])
    })

    it('negative zero', () => {
      expect(tokenize('-0')).toEqual([
        { _tag: 'number', value: '-0', complete: true },
      ])
    })

    it('-0.0', () => {
      expect(tokenize('-0.0')).toEqual([
        { _tag: 'number', value: '-0.0', complete: true },
      ])
    })

    it('0.0', () => {
      expect(tokenize('0.0')).toEqual([
        { _tag: 'number', value: '0.0', complete: true },
      ])
    })

    it('1e0', () => {
      expect(tokenize('1e0')).toEqual([
        { _tag: 'number', value: '1e0', complete: true },
      ])
    })

    it('1E0', () => {
      expect(tokenize('1E0')).toEqual([
        { _tag: 'number', value: '1E0', complete: true },
      ])
    })

    it('1e+0', () => {
      expect(tokenize('1e+0')).toEqual([
        { _tag: 'number', value: '1e+0', complete: true },
      ])
    })

    it('1e-0', () => {
      expect(tokenize('1e-0')).toEqual([
        { _tag: 'number', value: '1e-0', complete: true },
      ])
    })

    it('1.0e1', () => {
      expect(tokenize('1.0e1')).toEqual([
        { _tag: 'number', value: '1.0e1', complete: true },
      ])
    })

    it('-1.5e-10', () => {
      expect(tokenize('-1.5e-10')).toEqual([
        { _tag: 'number', value: '-1.5e-10', complete: true },
      ])
    })
  })

  // =========================================================================
  // CONSECUTIVE TOKENS NO WHITESPACE
  // =========================================================================
  describe('consecutive tokens no whitespace', () => {
    it('compact object', () => {
      expect(tokenize('{"a":1,"b":2}')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'colon' },
        { _tag: 'number', value: '1', complete: true },
        { _tag: 'comma' },
        { _tag: 'string', value: 'b', complete: true },
        { _tag: 'colon' },
        { _tag: 'number', value: '2', complete: true },
        { _tag: 'objectClose' },
      ])
    })

    it('compact array', () => {
      expect(tokenize('[1,2,3]')).toEqual([
        { _tag: 'arrayOpen' },
        { _tag: 'number', value: '1', complete: true },
        { _tag: 'comma' },
        { _tag: 'number', value: '2', complete: true },
        { _tag: 'comma' },
        { _tag: 'number', value: '3', complete: true },
        { _tag: 'arrayClose' },
      ])
    })

    it('keywords packed together', () => {
      expect(tokenize('[true,false,null]')).toEqual([
        { _tag: 'arrayOpen' },
        { _tag: 'true' },
        { _tag: 'comma' },
        { _tag: 'false' },
        { _tag: 'comma' },
        { _tag: 'null' },
        { _tag: 'arrayClose' },
      ])
    })
  })

  // =========================================================================
  // LOTS OF WHITESPACE
  // =========================================================================
  describe('lots of whitespace', () => {
    it('heavily spaced object', () => {
      expect(tokenize('{   "a"   :   1   ,   "b"   :   2   }')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'colon' },
        { _tag: 'number', value: '1', complete: true },
        { _tag: 'comma' },
        { _tag: 'string', value: 'b', complete: true },
        { _tag: 'colon' },
        { _tag: 'number', value: '2', complete: true },
        { _tag: 'objectClose' },
      ])
    })
  })

  // =========================================================================
  // EMPTY CHUNKS INTERSPERSED
  // =========================================================================
  describe('empty chunks', () => {
    it('empty chunks between structural tokens', () => {
      expect(tokenize('', '{', '', '}', '')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'objectClose' },
      ])
    })

    it('empty chunks around string', () => {
      expect(tokenize('', '"hello"', '')).toEqual([
        { _tag: 'string', value: 'hello', complete: true },
      ])
    })

    it('empty chunks mid-string', () => {
      expect(tokenize('"he', '', 'llo"')).toEqual([
        { _tag: 'string', value: 'hello', complete: true },
      ])
    })

    it('empty chunks around keyword', () => {
      expect(tokenize('', 'true', '', ',', '')).toEqual([
        { _tag: 'true' },
        { _tag: 'comma' },
      ])
    })

    it('all empty chunks', () => {
      expect(tokenize('', '', '')).toEqual([])
    })
  })

  // =========================================================================
  // VERY LONG STRINGS
  // =========================================================================
  describe('very long strings', () => {
    it('100+ char string single chunk', () => {
      const content = 'a'.repeat(150)
      expect(tokenize(`"${content}"`)).toEqual([
        { _tag: 'string', value: content, complete: true },
      ])
    })

    it('100+ char string in multiple chunks', () => {
      const content = 'abcdefghij'.repeat(15)
      const chunks = ['"']
      for (let i = 0; i < content.length; i += 10) {
        chunks.push(content.slice(i, i + 10))
      }
      chunks.push('"')
      expect(tokenize(...chunks)).toEqual([
        { _tag: 'string', value: content, complete: true },
      ])
    })
  })

  // =========================================================================
  // MULTIPLE STRINGS IN SEQUENCE
  // =========================================================================
  describe('multiple strings in sequence', () => {
    it('two strings separated by comma', () => {
      expect(tokenize('"a","b"')).toEqual([
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'comma' },
        { _tag: 'string', value: 'b', complete: true },
      ])
    })

    it('three strings as array elements', () => {
      expect(tokenize('["a","b","c"]')).toEqual([
        { _tag: 'arrayOpen' },
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'comma' },
        { _tag: 'string', value: 'b', complete: true },
        { _tag: 'comma' },
        { _tag: 'string', value: 'c', complete: true },
        { _tag: 'arrayClose' },
      ])
    })
  })

  // =========================================================================
  // WHITESPACE-ONLY CHUNKS BETWEEN TOKENS
  // =========================================================================
  describe('whitespace-only chunks between tokens', () => {
    it('whitespace chunks between object tokens', () => {
      expect(tokenize('{', '  \n  ', '"a"', '  \t  ', ':', '  ', '1', '\n', '}')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'colon' },
        { _tag: 'number', value: '1', complete: true },
        { _tag: 'objectClose' },
      ])
    })
  })

  // =========================================================================
  // PENDING STATE VERIFICATION
  // =========================================================================
  describe('pending state at various progress points', () => {
    it('inString: after opening quote', () => {
      expect(getPending('"')).toEqual({ _tag: 'string', content: '' })
    })

    it('inString: mid content', () => {
      expect(getPending('"abc')).toEqual({ _tag: 'string', content: 'abc' })
    })

    it('inString: after escape backslash', () => {
      expect(getPending('"abc\\')).toEqual({ _tag: 'string', content: 'abc' })
    })

    it('inString: mid unicode escape \\u0', () => {
      expect(getPending('"\\u0')).toEqual({ _tag: 'string', content: '' })
    })

    it('inNumber: single digit', () => {
      expect(getPending('1')).toEqual({ _tag: 'number', content: '1' })
    })

    it('inNumber: after decimal point', () => {
      expect(getPending('1.')).toEqual({ _tag: 'number', content: '1.' })
    })

    it('inNumber: after exponent', () => {
      expect(getPending('1e')).toEqual({ _tag: 'number', content: '1e' })
    })

    it('inNumber: after exponent sign', () => {
      expect(getPending('1e+')).toEqual({ _tag: 'number', content: '1e+' })
    })

    it('inNumber: negative sign only', () => {
      expect(getPending('-')).toEqual({ _tag: 'number', content: '-' })
    })

    it('inKeyword: t', () => {
      expect(getPending('t')).toEqual({ _tag: 'keyword', content: 't' })
    })

    it('inKeyword: tr', () => {
      expect(getPending('tr')).toEqual({ _tag: 'keyword', content: 'tr' })
    })

    it('inKeyword: fals', () => {
      expect(getPending('fals')).toEqual({ _tag: 'keyword', content: 'fals' })
    })

    it('inKeyword: nul', () => {
      expect(getPending('nul')).toEqual({ _tag: 'keyword', content: 'nul' })
    })

    it('inUnquoted: bare word', () => {
      expect(getPending('mykey')).toEqual({ _tag: 'unquoted', content: 'mykey' })
    })

    it('no pending after structural token', () => {
      expect(getPending('{')).toBeNull()
    })

    it('no pending after complete string', () => {
      expect(getPending('"hello"')).toBeNull()
    })

    it('no pending after whitespace', () => {
      expect(getPending('   ')).toBeNull()
    })
  })

  // =========================================================================
  // COMPLEX SEQUENCES
  // =========================================================================
  describe('complex sequences', () => {
    it('full object with multiple keys', () => {
      expect(tokenize('{"a": true, "b": false, "c": null}')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'colon' },
        { _tag: 'true' },
        { _tag: 'comma' },
        { _tag: 'string', value: 'b', complete: true },
        { _tag: 'colon' },
        { _tag: 'false' },
        { _tag: 'comma' },
        { _tag: 'string', value: 'c', complete: true },
        { _tag: 'colon' },
        { _tag: 'null' },
        { _tag: 'objectClose' },
      ])
    })

    it('array of mixed types', () => {
      expect(tokenize('[1, "two", true, null]')).toEqual([
        { _tag: 'arrayOpen' },
        { _tag: 'number', value: '1', complete: true },
        { _tag: 'comma' },
        { _tag: 'string', value: 'two', complete: true },
        { _tag: 'comma' },
        { _tag: 'true' },
        { _tag: 'comma' },
        { _tag: 'null' },
        { _tag: 'arrayClose' },
      ])
    })

    it('nested structure', () => {
      expect(tokenize('{"a": [1, {"b": 2}]}')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'colon' },
        { _tag: 'arrayOpen' },
        { _tag: 'number', value: '1', complete: true },
        { _tag: 'comma' },
        { _tag: 'objectOpen' },
        { _tag: 'string', value: 'b', complete: true },
        { _tag: 'colon' },
        { _tag: 'number', value: '2', complete: true },
        { _tag: 'objectClose' },
        { _tag: 'arrayClose' },
        { _tag: 'objectClose' },
      ])
    })
  })

  // =========================================================================
  // EOF WITH PENDING ESCAPE / UNICODE
  // =========================================================================
  describe('EOF with pending escape and unicode', () => {
    it('EOF with pending escape: "hello\\ then end()', () => {
      expect(tokenize('"hello\\')).toEqual([
        { _tag: 'string', value: 'hello', complete: false },
      ])
    })

    it('EOF with pending unicode (1 hex char): "a\\u0 then end()', () => {
      expect(tokenize('"a\\u0')).toEqual([
        { _tag: 'string', value: 'a', complete: false },
      ])
    })

    it('EOF with pending unicode (2 hex chars): "a\\u00 then end()', () => {
      expect(tokenize('"a\\u00')).toEqual([
        { _tag: 'string', value: 'a', complete: false },
      ])
    })

    it('EOF with pending unicode (3 hex chars): "a\\u004 then end()', () => {
      expect(tokenize('"a\\u004')).toEqual([
        { _tag: 'string', value: 'a', complete: false },
      ])
    })

    it('backslash at very end of input (no close quote)', () => {
      expect(tokenize('"\\\\' )).toEqual([
        { _tag: 'string', value: '\\', complete: false },
      ])
    })

    it('unicode escape at very end: "\\u0041 (no close quote)', () => {
      expect(tokenize('"\\u0041')).toEqual([
        { _tag: 'string', value: 'A', complete: false },
      ])
    })
  })

  // =========================================================================
  // MULTIPLE CONSECUTIVE ESCAPES
  // =========================================================================
  describe('multiple consecutive escapes', () => {
    it('three backslash pairs + newline escape: "\\\\\\\\\\\\\\n"', () => {
      expect(tokenize('"\\\\\\\\\\\\\\n"')).toEqual([
        { _tag: 'string', value: '\\\\\\\n', complete: true },
      ])
    })

    it('multiple escapes across chunk boundaries: "\\\\ + \\\\ + \\\\n"', () => {
      expect(tokenize('"\\\\', '\\\\', '\\n"')).toEqual([
        { _tag: 'string', value: '\\\\\n', complete: true },
      ])
    })

    it('string containing only escapes: "\\n\\t\\r"', () => {
      expect(tokenize('"\\n\\t\\r"')).toEqual([
        { _tag: 'string', value: '\n\t\r', complete: true },
      ])
    })

    it('escaped quote followed by real close: "say \\"hi\\""', () => {
      expect(tokenize('"say \\"hi\\""')).toEqual([
        { _tag: 'string', value: 'say "hi"', complete: true },
      ])
    })
  })

  // =========================================================================
  // TWO STRINGS BACK TO BACK
  // =========================================================================
  describe('two strings back to back', () => {
    it('"a""b" (no separator)', () => {
      expect(tokenize('"a""b"')).toEqual([
        { _tag: 'string', value: 'a', complete: true },
        { _tag: 'string', value: 'b', complete: true },
      ])
    })
  })

  // =========================================================================
  // NUMBER EDGE CASES
  // =========================================================================
  describe('number edge cases', () => {
    it('0 followed immediately by comma: 0,', () => {
      expect(tokenize('0,')).toEqual([
        { _tag: 'number', value: '0', complete: true },
        { _tag: 'comma' },
      ])
    })

    it('very large number: 99999999999999999', () => {
      expect(tokenize('99999999999999999')).toEqual([
        { _tag: 'number', value: '99999999999999999', complete: true },
      ])
    })

    it('number with many decimal places: 3.141592653589793', () => {
      expect(tokenize('3.141592653589793')).toEqual([
        { _tag: 'number', value: '3.141592653589793', complete: true },
      ])
    })
  })

  // =========================================================================
  // KEYWORD AT EOF WITH NO DELIMITER
  // =========================================================================
  describe('keyword at EOF with no delimiter', () => {
    it('true at EOF', () => {
      expect(tokenize('true')).toEqual([{ _tag: 'true' }])
    })

    it('false at EOF', () => {
      expect(tokenize('false')).toEqual([{ _tag: 'false' }])
    })

    it('null at EOF', () => {
      expect(tokenize('null')).toEqual([{ _tag: 'null' }])
    })
  })

  // =========================================================================
  // EMPTY STRING AT CHUNK BOUNDARY
  // =========================================================================
  describe('empty string at chunk boundary', () => {
    it('"" split as " + "', () => {
      expect(tokenize('"', '"')).toEqual([
        { _tag: 'string', value: '', complete: true },
      ])
    })
  })

  // =========================================================================
  // WHITESPACE CHARACTERS IN DEFAULT MODE
  // =========================================================================
  describe('whitespace characters in default mode', () => {
    it('tab character is whitespace', () => {
      expect(tokenize('\t42')).toEqual([
        { _tag: 'number', value: '42', complete: true },
      ])
    })

    it('carriage return is whitespace', () => {
      expect(tokenize('\r42')).toEqual([
        { _tag: 'number', value: '42', complete: true },
      ])
    })

    it('newline is whitespace', () => {
      expect(tokenize('\n42')).toEqual([
        { _tag: 'number', value: '42', complete: true },
      ])
    })

    it('multiple whitespace types between tokens: { \\t\\n\\r }', () => {
      expect(tokenize('{ \t\n\r }')).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'objectClose' },
      ])
    })
  })

  // =========================================================================
  // STRESS TESTS
  // =========================================================================
  describe('stress tests', () => {
    it('1000-char string in 10-char chunks', () => {
      const content = 'a'.repeat(1000)
      const chunks: string[] = ['"']
      for (let i = 0; i < content.length; i += 10) {
        chunks.push(content.slice(i, i + 10))
      }
      chunks.push('"')
      expect(tokenize(...chunks)).toEqual([
        { _tag: 'string', value: content, complete: true },
      ])
    })

    it('[1,2,3,...,100] char by char', () => {
      const input = '[' + Array.from({ length: 100 }, (_, i) => String(i)).join(',') + ']'
      const charByChar = tokenize(...input.split(''))
      const singleChunk = tokenize(input)
      expect(charByChar).toEqual(singleChunk)
    })

    it('alternating empty and non-empty chunks for complex payload', () => {
      const input = '{"a": [1, "two", true, null], "b": {"c": 3.14}}'
      const chunks: string[] = []
      for (const ch of input) {
        chunks.push('')
        chunks.push(ch)
        chunks.push('')
      }
      expect(tokenize(...chunks)).toEqual(tokenize(input))
    })

    it('1000-char string char by char', () => {
      const content = 'x'.repeat(1000)
      const input = `"${content}"`
      const charByChar = tokenize(...input.split(''))
      expect(charByChar).toEqual([
        { _tag: 'string', value: content, complete: true },
      ])
    })

    it('deeply nested braces char by char', () => {
      const depth = 50
      const input = '{'.repeat(depth) + '}'.repeat(depth)
      const charByChar = tokenize(...input.split(''))
      const singleChunk = tokenize(input)
      expect(charByChar).toEqual(singleChunk)
    })

    it('many keywords in array char by char', () => {
      const input = '[true,false,null,true,false,null,true,false,null,true]'
      const charByChar = tokenize(...input.split(''))
      const singleChunk = tokenize(input)
      expect(charByChar).toEqual(singleChunk)
    })

    it('many strings in array char by char', () => {
      const input = '["a","b","c","d","e","f","g","h","i","j"]'
      const charByChar = tokenize(...input.split(''))
      const singleChunk = tokenize(input)
      expect(charByChar).toEqual(singleChunk)
    })

    it('mixed types char by char', () => {
      const input = '{"a":1,"b":"two","c":true,"d":null,"e":[1,2]}'
      const charByChar = tokenize(...input.split(''))
      const singleChunk = tokenize(input)
      expect(charByChar).toEqual(singleChunk)
    })
  })

  // =========================================================================
  // TOKENIZE NO END BEHAVIOR
  // =========================================================================
  describe('tokenizeNoEnd behavior', () => {
    it('keyword not emitted without end() or delimiter', () => {
      const tokens = tokenizeNoEnd('true')
      expect(tokens).toEqual([])
    })

    it('number not emitted without end() or delimiter', () => {
      const tokens = tokenizeNoEnd('42')
      expect(tokens).toEqual([])
    })

    it('unquoted not emitted without end() or delimiter', () => {
      const tokens = tokenizeNoEnd('hello')
      expect(tokens).toEqual([])
    })

    it('string emitted on close quote even without end()', () => {
      const tokens = tokenizeNoEnd('"hello"')
      expect(tokens).toEqual([
        { _tag: 'string', value: 'hello', complete: true },
      ])
    })

    it('structural tokens emitted immediately without end()', () => {
      const tokens = tokenizeNoEnd('{[]}')
      expect(tokens).toEqual([
        { _tag: 'objectOpen' },
        { _tag: 'arrayOpen' },
        { _tag: 'arrayClose' },
        { _tag: 'objectClose' },
      ])
    })

    it('keyword emitted when followed by delimiter without end()', () => {
      const tokens = tokenizeNoEnd('true,')
      expect(tokens).toEqual([
        { _tag: 'true' },
        { _tag: 'comma' },
      ])
    })

    it('number emitted when followed by delimiter without end()', () => {
      const tokens = tokenizeNoEnd('42}')
      expect(tokens).toEqual([
        { _tag: 'number', value: '42', complete: true },
        { _tag: 'objectClose' },
      ])
    })
  })

  // =========================================================================
  // UNICODE EDGE CASES
  // =========================================================================
  describe('unicode edge cases', () => {
    it('multiple unicode escapes in sequence', () => {
      expect(tokenize('"\\u0041\\u0042\\u0043"')).toEqual([
        { _tag: 'string', value: 'ABC', complete: true },
      ])
    })

    it('unicode null char \\u0000', () => {
      expect(tokenize('"\\u0000"')).toEqual([
        { _tag: 'string', value: '\u0000', complete: true },
      ])
    })

    it('unicode max BMP \\uFFFF', () => {
      expect(tokenize('"\\uFFFF"')).toEqual([
        { _tag: 'string', value: '\uFFFF', complete: true },
      ])
    })

    it('unicode lowercase hex \\u00ff', () => {
      expect(tokenize('"\\u00ff"')).toEqual([
        { _tag: 'string', value: '\u00ff', complete: true },
      ])
    })

    it('unicode mixed case \\u00Ff', () => {
      expect(tokenize('"\\u00Ff"')).toEqual([
        { _tag: 'string', value: '\u00Ff', complete: true },
      ])
    })
  })
})
