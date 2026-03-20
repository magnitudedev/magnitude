import { describe, expect, test } from 'vitest'
import { normalizeReferencedPath, scanFileRefs } from './file-refs'

describe('scanFileRefs', () => {
  test('matches basic ref', () => {
    const text = 'Use [plan](plan.md) next.'
    const refs = scanFileRefs(text)
    expect(refs).toEqual([
      {
        path: 'plan.md',
        start: 4,
        end: 19,
        raw: '[plan](plan.md)',
      },
    ])
  })

  test('matches section ref', () => {
    const refs = scanFileRefs('See [plan](plan.md#Approach)')
    expect(refs).toEqual([
      {
        path: 'plan.md',
        section: 'Approach',
        start: 4,
        end: 28,
        raw: '[plan](plan.md#Approach)',
      },
    ])
  })

  test('matches slash path', () => {
    const refs = scanFileRefs('[auth](src/auth.ts)')
    expect(refs).toEqual([
      {
        path: 'src/auth.ts',
        start: 0,
        end: 19,
        raw: '[auth](src/auth.ts)',
      },
    ])
  })

  test('matches explicit workspace path', () => {
    const refs = scanFileRefs('Use [plan]($M/plan.md)')
    expect(refs).toEqual([
      {
        path: '$M/plan.md',
        start: 4,
        end: 22,
        raw: '[plan]($M/plan.md)',
      },
    ])
  })

  test('skips fenced code blocks', () => {
    const text = 'Before [plan](plan.md)\n```ts\nconst x = "[inside](inside.ts)"\n```\nAfter [after](after.md)'
    const refs = scanFileRefs(text)
    expect(refs.map((r) => r.path)).toEqual(['plan.md', 'after.md'])
  })

  test('skips inline code', () => {
    const text = 'Use `[inside](inside.ts)` and [outside](outside.ts)'
    const refs = scanFileRefs(text)
    expect(refs.map((r) => r.path)).toEqual(['outside.ts'])
  })

  test('skips urls', () => {
    const refs = scanFileRefs('[docs](https://example.com/x.md)')
    expect(refs).toEqual([])
  })

  test('normalizes leading dot path', () => {
    const refs = scanFileRefs('Use [plan](./plan.md)')
    expect(refs).toEqual([
      {
        path: 'plan.md',
        start: 4,
        end: 21,
        raw: '[plan](./plan.md)',
      },
    ])
  })

  test('rejects escaping paths', () => {
    const refs = scanFileRefs('Do not use [bad](../../etc/passwd)')
    expect(refs).toEqual([])
  })

  test('matches multiple refs in one text', () => {
    const refs = scanFileRefs('Use [one](one.md) and [two](two.ts) now')
    expect(refs.map((r) => r.path)).toEqual(['one.md', 'two.ts'])
  })
})

describe('normalizeReferencedPath', () => {
  test('normalizes current directory prefix', () => {
    expect(normalizeReferencedPath('./plan.md')).toBe('plan.md')
  })

  test('rejects parent escape', () => {
    expect(normalizeReferencedPath('../../etc/passwd')).toBeNull()
  })
})
