import { test, expect, describe } from 'bun:test'
import { measureBounded, truncate } from './truncation'
import type { JsonValue } from './truncation'

describe('truncation handles undefined values', () => {
  test('measureBounded with top-level undefined', () => {
    expect(() => measureBounded(undefined as unknown as JsonValue, 100)).not.toThrow()
  })

  test('truncate top-level undefined renders as "undefined"', () => {
    expect(truncate(undefined as unknown as JsonValue, 100)).toBe('undefined')
  })

  test('measureBounded with nested undefined in object', () => {
    const val = { forkId: 'abc', taskId: undefined } as unknown as JsonValue
    expect(() => measureBounded(val, 100)).not.toThrow()
  })

  test('truncate omits undefined keys from objects', () => {
    const val = { forkId: 'abc', taskId: undefined } as unknown as JsonValue
    const result = truncate(val, 100)
    expect(result).not.toContain('taskId')
    expect(result).toContain('forkId')
  })

  test('measureBounded with deeply nested undefined', () => {
    const val = { a: { b: { c: undefined } } } as unknown as JsonValue
    expect(() => measureBounded(val, 100)).not.toThrow()
  })

  test('truncate with deeply nested undefined', () => {
    const val = { a: { b: { c: undefined } } } as unknown as JsonValue
    expect(() => truncate(val, 100)).not.toThrow()
  })

  test('measureBounded with undefined in array', () => {
    const val = [1, undefined, 'hello'] as unknown as JsonValue
    expect(() => measureBounded(val, 100)).not.toThrow()
  })

  test('truncate with undefined in array', () => {
    const val = [1, undefined, 'hello'] as unknown as JsonValue
    expect(() => truncate(val, 100)).not.toThrow()
  })

  test('truncate null still renders as "null"', () => {
    expect(truncate(null, 100)).toBe('null')
  })

  // Tight budgets force truncateObject/truncateArray paths (not renderFull),
  // which call flatMinCost/minMeaningfulCost on each value — the actual crash site.
  test('truncate object with undefined value under tight budget', () => {
    const val = { a: undefined, b: 'hello', c: 123 } as unknown as JsonValue
    expect(() => truncate(val, 3)).not.toThrow()
  })

  test('truncate object with null value under tight budget', () => {
    const val = { a: null, b: 'hello', c: 123 } as unknown as JsonValue
    expect(() => truncate(val, 3)).not.toThrow()
  })

  test('truncate deeply nested undefined under tight budget', () => {
    const val = { a: { b: { c: undefined } }, d: 'x' } as unknown as JsonValue
    expect(() => truncate(val, 3)).not.toThrow()
  })
})
