import { describe, expect, test } from 'bun:test'

import { deriveVisualLineStarts } from './multiline-input'

describe('deriveVisualLineStarts', () => {
  test('falls back to logical line starts when lineInfo is unavailable', () => {
    expect(deriveVisualLineStarts('', null)).toEqual([0])
    expect(deriveVisualLineStarts('one\ntwo\nthree', null)).toEqual([0, 4, 8])
  })

  test('falls back to logical line starts when lineInfo arrays are missing/empty', () => {
    expect(
      deriveVisualLineStarts('a\nb', { lineSources: [] as number[], lineStartCols: [0] as number[] } as any),
    ).toEqual([0, 2])

    expect(
      deriveVisualLineStarts('a\nb', { lineSources: [0] as number[], lineStartCols: [] as number[] } as any),
    ).toEqual([0, 2])
  })
})
