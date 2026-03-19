import { describe, expect, it } from 'bun:test'
import { buildMergedPalette } from './theme'

describe('markdown/theme', () => {
  it('returns full default palette with no overrides', () => {
    const palette = buildMergedPalette()
    expect(palette.inlineCodeFg).toBeString()
    expect(palette.syntax.keyword).toBeString()
    expect(palette.headingFg[1]).toBeString()
  })

  it('merges partial top-level overrides', () => {
    const palette = buildMergedPalette({ linkFg: '#ff00ff' })
    expect(palette.linkFg).toBe('#ff00ff')
    expect(palette.syntax.keyword).toBeString()
  })

  it('deep-merges headingFg overrides', () => {
    const base = buildMergedPalette()
    const palette = buildMergedPalette({ headingFg: { 2: '#123456' } })
    expect(palette.headingFg[2]).toBe('#123456')
    expect(palette.headingFg[1]).toBe(base.headingFg[1])
  })

  it('deep-merges syntax overrides', () => {
    const base = buildMergedPalette()
    const palette = buildMergedPalette({ syntax: { keyword: '#abcdef' } as any })
    expect(palette.syntax.keyword).toBe('#abcdef')
    expect(palette.syntax.string).toBe(base.syntax.string)
  })

  it("ignores unknown keys safely", () => {
    const palette = buildMergedPalette({ unknownThing: 'x' } as any)
    expect(palette.syntax.default).toBeString()
    expect((palette as any).unknownThing).toBe('x')
  })
})
