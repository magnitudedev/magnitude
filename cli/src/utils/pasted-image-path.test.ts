import { describe, expect, test } from 'bun:test'
import { extractPastedPathCandidates } from './pasted-image-path'

describe('extractPastedPathCandidates', () => {
  test('parses text/uri-list payload and ignores comments/blank lines', () => {
    const payload = [
      '# exported from drag source',
      '',
      'file:///Users/me/a.png',
      'file:///Users/me/b.png',
    ].join('\n')

    expect(extractPastedPathCandidates(payload)).toEqual([
      '/Users/me/a.png',
      '/Users/me/b.png',
    ])
  })

  test('parses line-delimited paths', () => {
    const payload = '/Users/me/a.png\r\n/Users/me/b.png\r/Users/me/c.png\n/Users/me/d.png'
    expect(extractPastedPathCandidates(payload)).toEqual([
      '/Users/me/a.png',
      '/Users/me/b.png',
      '/Users/me/c.png',
      '/Users/me/d.png',
    ])
  })

  test('parses shell-word space-delimited paths with quotes and escapes', () => {
    const payload = "/Users/me/a.png '/Users/me/b b.png' /Users/me/c\\ c.png"
    expect(extractPastedPathCandidates(payload)).toEqual([
      '/Users/me/a.png',
      '/Users/me/b b.png',
      '/Users/me/c c.png',
    ])
  })

  test('returns single-path fallback', () => {
    expect(extractPastedPathCandidates('/Users/me/a.png')).toEqual(['/Users/me/a.png'])
  })

  test('filters malformed and empty tokens and dedupes preserving order', () => {
    const payload = "   '/Users/me/a.png'   /Users/me/a.png    ''   "
    expect(extractPastedPathCandidates(payload)).toEqual(['/Users/me/a.png'])
  })
})
