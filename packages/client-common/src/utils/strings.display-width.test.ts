import { describe, expect, test } from 'bun:test'
import {
  getDisplayWidth,
  padEndToDisplayWidth,
  truncateToDisplayWidth,
} from './strings'

describe('display-width string helpers', () => {
  test('getDisplayWidth handles ascii and wide glyphs', () => {
    expect(getDisplayWidth('abc')).toBe(3)
    expect(getDisplayWidth('界')).toBe(2)
  })

  test('truncateToDisplayWidth keeps text when within width', () => {
    expect(truncateToDisplayWidth('agent', 10)).toBe('agent')
  })

  test('truncateToDisplayWidth truncates ascii and appends ellipsis', () => {
    const output = truncateToDisplayWidth('abcdefghijklmnopqrstuvwxyz', 8)
    expect(output).toBe('abcdefg…')
    expect(getDisplayWidth(output)).toBeLessThanOrEqual(8)
  })

  test('truncateToDisplayWidth respects display width for emoji/cjk', () => {
    const output = truncateToDisplayWidth('agent🚀界status', 8)
    expect(getDisplayWidth(output)).toBeLessThanOrEqual(8)
  })

  test('truncateToDisplayWidth respects display width for combining graphemes', () => {
    const output = truncateToDisplayWidth('e\u0301e\u0301e\u0301e\u0301', 4)
    expect(getDisplayWidth(output)).toBeLessThanOrEqual(4)
  })

  test('padEndToDisplayWidth pads ascii to exact target width', () => {
    const output = padEndToDisplayWidth('abc', 6)
    expect(output).toBe('abc   ')
    expect(getDisplayWidth(output)).toBe(6)
  })

  test('padEndToDisplayWidth pads mixed-width text to exact target width', () => {
    const output = padEndToDisplayWidth('界a', 6)
    expect(getDisplayWidth(output)).toBe(6)
  })
})
