import { describe, test, expect } from 'bun:test'
import { validateAndApply } from '../edit'

describe('existing behavior - no virtual matching needed', () => {
  test('Inline match in middle of file', async () => {
    const result = validateAndApply('alpha\nfoo\nomega', 'foo', 'bar', false)
    expect(result.result).toBe('alpha\nbar\nomega')
  })

  test("Inline match at SOF (old doesn't start with \\n)", async () => {
    const result = validateAndApply('foo\nbar', 'foo', 'baz', false)
    expect(result.result).toBe('baz\nbar')
  })

  test("Inline match at EOF (old doesn't end with \\n)", async () => {
    const result = validateAndApply('bar\nfoo', 'foo', 'baz', false)
    expect(result.result).toBe('bar\nbaz')
  })

  test('Inline match entire file', async () => {
    const result = validateAndApply('content', 'content', 'replacement', false)
    expect(result.result).toBe('replacement')
  })

  test('replaceAll with multiple matches', async () => {
    const result = validateAndApply('foo\nfoo\nfoo', 'foo', 'bar', true)
    expect(result.result).toBe('bar\nbar\nbar')
  })

  test('Not found error', async () => {
    expect(() => validateAndApply('alpha\nbeta', 'gamma', 'delta', false)).toThrow('not found')
  })

  test('Ambiguous match error', async () => {
    expect(() => validateAndApply('foo\nbar\nfoo', 'foo', 'baz', false)).toThrow('2 locations')
  })
})

describe('virtual leading newline - SOF tolerance', () => {
  test('old = "\\nfoo", file starts with "foo" → should match', async () => {
    const result = validateAndApply('foo\nbar', '\nfoo', 'baz', false)
    expect(result.result).toBe('baz\nbar')
  })

  test('old = "\\nfoo\\n", file = "foo\\nbar" → should match, clip leading \\n from new', async () => {
    const result = validateAndApply('foo\nbar', '\nfoo\n', '\nbaz\n', false)
    expect(result.result).toBe('baz\nbar')
  })

  test('Block style old/new at SOF: old = "\\nfoo\\n", new = "\\nbaz\\n", file = "foo\\nbar" → result = "baz\\nbar"', async () => {
    const result = validateAndApply('foo\nbar', '\nfoo\n', '\nbaz\n', false)
    expect(result.result).toBe('baz\nbar')
  })
})

describe('virtual trailing newline - EOF tolerance', () => {
  test('old = "foo\\n", file ends with "foo" (no trailing \\n) → should match', async () => {
    const result = validateAndApply('bar\nfoo', 'foo\n', 'baz', false)
    expect(result.result).toBe('bar\nbaz')
  })

  test('old = "\\nfoo\\n", file = "bar\\nfoo" → should match, clip trailing \\n from new', async () => {
    const result = validateAndApply('bar\nfoo', '\nfoo\n', '\nbaz\n', false)
    expect(result.result).toBe('bar\nbaz')
  })

  test('Block style old/new at EOF: old = "\\nfoo\\n", new = "\\nbaz\\n", file = "bar\\nfoo" → result = "bar\\nbaz"', async () => {
    const result = validateAndApply('bar\nfoo', '\nfoo\n', '\nbaz\n', false)
    expect(result.result).toBe('bar\nbaz')
  })
})

describe('virtual both boundaries - entire file tolerance', () => {
  test('old = "\\ncontent\\n", file = "content" → should match entire file', async () => {
    const result = validateAndApply('content', '\ncontent\n', 'replacement', false)
    expect(result.result).toBe('replacement')
  })

  test('Block style replace entire file: old = "\\nold content\\n", new = "\\nnew content\\n", file = "old content" → result = "new content"', async () => {
    const result = validateAndApply('old content', '\nold content\n', '\nnew content\n', false)
    expect(result.result).toBe('new content')
  })

  test('old = "\\nx\\n", file = "x" → result with new = "\\ny\\n" should be "y"', async () => {
    const result = validateAndApply('x', '\nx\n', '\ny\n', false)
    expect(result.result).toBe('y')
  })
})

describe('virtual matching is fallback only', () => {
  test('old = "\\nfoo\\n" should match REAL newlines first when they exist in the file', async () => {
    const result = validateAndApply('aaa\nfoo\nbar', '\nfoo\n', '\nbaz\n', false)
    expect(result.result).toBe('aaa\nbaz\nbar')
  })

  test('file = "aaa\\nfoo\\nbar", old = "\\nfoo\\n" → should match the real "\\nfoo\\n", NOT use virtual', async () => {
    const result = validateAndApply('aaa\nfoo\nbar', '\nfoo\n', '\nbaz\n', false)
    expect(result.result).toBe('aaa\nbaz\nbar')
  })

  test('file = "\\nfoo\\n", old = "\\nfoo\\n" → should match real content, not virtual', async () => {
    const result = validateAndApply('\nfoo\n', '\nfoo\n', '\nbaz\n', false)
    expect(result.result).toBe('\nbaz\n')
  })
})

describe('newString clipping', () => {
  test('Virtual leading consumed → clip leading \\n from new', async () => {
    const result = validateAndApply('foo\nbar', '\nfoo', '\nbaz', false)
    expect(result.result).toBe('baz\nbar')
  })

  test('Virtual trailing consumed → clip trailing \\n from new', async () => {
    const result = validateAndApply('bar\nfoo', 'foo\n', 'baz\n', false)
    expect(result.result).toBe('bar\nbaz')
  })

  test('Both virtual consumed → clip both from new', async () => {
    const result = validateAndApply('foo', '\nfoo\n', '\nbaz\n', false)
    expect(result.result).toBe('baz')
  })

  test("New doesn't start with \\n when leading consumed → no clip (new preserved as-is)", async () => {
    const result = validateAndApply('foo\nbar', '\nfoo', 'baz', false)
    expect(result.result).toBe('baz\nbar')
  })

  test("New doesn't end with \\n when trailing consumed → no clip (new preserved as-is)", async () => {
    const result = validateAndApply('bar\nfoo', 'foo\n', 'baz', false)
    expect(result.result).toBe('bar\nbaz')
  })
})

describe('intentional newline changes', () => {
  test('Add trailing newline at EOF: old = "world", new = "world\\n", file = "hello\\nworld" → "hello\\nworld\\n"', async () => {
    const result = validateAndApply('hello\nworld', 'world', 'world\n', false)
    expect(result.result).toBe('hello\nworld\n')
  })

  test('Remove trailing newline at EOF: old = "world\\n", new = "world", file = "hello\\nworld\\n" → "hello\\nworld"', async () => {
    const result = validateAndApply('hello\nworld\n', 'world\n', 'world', false)
    expect(result.result).toBe('hello\nworld')
  })

  test('Add trailing newline with block style: old = "\\nworld\\n", new = "\\nworld\\n\\n", file = "hello\\nworld" → should add one real trailing \\n', async () => {
    const result = validateAndApply('hello\nworld', '\nworld\n', '\nworld\n\n', false)
    expect(result.result).toBe('hello\nworld\n')
  })

  test('Remove trailing newline with block style: old = "\\nworld\\n\\n", new = "\\nworld\\n", file = "hello\\nworld\\n" → should remove trailing \\n', async () => {
    const result = validateAndApply('hello\nworld\n', '\nworld\n\n', '\nworld\n', false)
    expect(result.result).toBe('hello\nworld')
  })
})

describe('replaceAll with virtual matching', () => {
  test('replaceAll with real matches only → normal behavior, no virtual', async () => {
    const result = validateAndApply('foo\nfoo\nfoo', 'foo', 'bar', true)
    expect(result.result).toBe('bar\nbar\nbar')
  })

  test('replaceAll with no real matches, one virtual match → apply with clipping', async () => {
    const result = validateAndApply('foo\nbar', '\nfoo', 'baz', true)
    expect(result.result).toBe('baz\nbar')
  })

  test('replaceAll with no real matches, two virtual matches (SOF + EOF) → apply both with independent clipping', async () => {
    const result = validateAndApply('foo\nmiddle\nfoo', '\nfoo\n', '\nbaz\n', true)
    expect(result.result).toBe('baz\nmiddle\nbaz')
  })

  test('replaceAll with real matches existing → virtual matches ignored', async () => {
    const result = validateAndApply('aaa\nfoo\nbbb\nfoo\nccc', '\nfoo\n', '\nbaz\n', true)
    expect(result.result).toBe('aaa\nbaz\nbbb\nbaz\nccc')
  })
})

describe('edge cases', () => {
  test('Empty old string (should error)', async () => {
    expect(() => validateAndApply('foo', '', 'bar', false)).toThrow()
  })

  test('old = "\\n" only, no real match → virtual match rejected (zero-length real region)', async () => {
    expect(() => validateAndApply('foo', '\n', 'bar', false)).toThrow()
  })

  test('old = "\\n\\n" only → similar rejection', async () => {
    expect(() => validateAndApply('foo', '\n\n', 'bar', false)).toThrow()
  })

  test('File with real leading/trailing newlines: file = "\\nfoo\\n", old = "\\nfoo\\n" → should match REAL content, not virtual', async () => {
    const result = validateAndApply('\nfoo\n', '\nfoo\n', '\nbar\n', false)
    expect(result.result).toBe('\nbar\n')
  })

  test('Very large file with content at boundaries', async () => {
    const prefix = 'a'.repeat(5000)
    const suffix = 'z'.repeat(5000)
    const result = validateAndApply(`foo\n${prefix}\n${suffix}\nbar`, '\nfoo', 'baz', false)
    expect(result.result).toBe(`baz\n${prefix}\n${suffix}\nbar`)
  })
})

describe('no trimming - literal content', () => {
  test('old with leading spaces preserved: old = "  foo", file has "  foo" → matches', async () => {
    const result = validateAndApply('  foo\nbar', '  foo', '  baz', false)
    expect(result.result).toBe('  baz\nbar')
  })

  test('old with trailing spaces preserved: old = "foo  ", file has "foo  " → matches', async () => {
    const result = validateAndApply('foo  \nbar', 'foo  ', 'baz  ', false)
    expect(result.result).toBe('baz  \nbar')
  })

  test('Multiple leading newlines: old = "\\n\\nfoo", file = "\\nfoo" → virtual absorbs one \\n, real \\n matches the other', async () => {
    // Virtual boundary provides one \n, the real file starts with \n, so \n\nfoo matches
    const result = validateAndApply('\nfoo', '\n\nfoo', 'bar', false)
    expect(result.result).toBe('bar')
  })
})