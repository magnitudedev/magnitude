import { describe, it, expect } from 'vitest'
import { describeShape } from '../src/truncation/describe-shape'

describe('describeShape', () => {
  describe('primitives', () => {
    it('renders null', () => {
      expect(describeShape(null)).toBe('null')
    })

    it('renders booleans', () => {
      expect(describeShape(true)).toBe('true')
      expect(describeShape(false)).toBe('false')
    })

    it('renders numbers', () => {
      expect(describeShape(42)).toBe('42')
      expect(describeShape(3.14)).toBe('3.14')
      expect(describeShape(0)).toBe('0')
      expect(describeShape(-1)).toBe('-1')
    })

    it('renders undefined', () => {
      expect(describeShape(undefined)).toBe('undefined')
    })
  })

  describe('strings', () => {
    it('shows short strings as-is', () => {
      expect(describeShape('hello')).toBe('"hello"')
    })

    it('shows empty string', () => {
      expect(describeShape('')).toBe('""')
    })

    it('shows descriptor for very large strings', () => {
      const big = 'x'.repeat(100_000)
      const result = describeShape(big)
      expect(result).toMatch(/^<string, 100000 chars, ~\S+ tokens>$/)
    })

    it('shows truncated prefix for medium strings with tight budget', () => {
      // 100 chars, budget=20 tokens (80 chars capacity). 100 < 80*4=320, so truncated prefix
      const medium = 'abcdefghij'.repeat(10) // 100 chars
      const result = describeShape(medium, 20)
      expect(result).toMatch(/^"abcdef.*\.\.\."$/)
    })
  })

  describe('objects', () => {
    it('renders empty object', () => {
      expect(describeShape({})).toBe('{}')
    })

    it('renders small object as full JSON', () => {
      const result = describeShape({ name: 'John', age: 30 })
      expect(result).toContain('"name": "John"')
      expect(result).toContain('"age": 30')
    })

    it('renders shell-like result with large stdout', () => {
      const obj = {
        stdout: 'x'.repeat(100_000),
        stderr: '',
        exitCode: 0,
        mode: 'sync',
      }
      const result = describeShape(obj)
      expect(result).toContain('<string, 100000 chars')
      expect(result).toContain('"stderr": ""')
      expect(result).toContain('"exitCode": 0')
      expect(result).toContain('"mode": "sync"')
    })

    it('collapses when budget is extremely tight', () => {
      const obj = { a: 1, b: 2, c: 3, d: 4, e: 5 }
      const result = describeShape(obj, 1)
      expect(result).toMatch(/\{<5 keys>\.\.\.}/)
    })

    it('shows ...N more for excess keys', () => {
      const obj: Record<string, string> = {}
      for (let i = 0; i < 20; i++) obj[`key${i}`] = 'x'.repeat(500)
      const result = describeShape(obj, 50)
      expect(result).toMatch(/\.\.\.\d+ more/)
    })
  })

  describe('arrays', () => {
    it('renders empty array', () => {
      expect(describeShape([])).toBe('[]')
    })

    it('renders small array as full JSON', () => {
      const result = describeShape([1, 2, 3])
      expect(result).toContain('1')
      expect(result).toContain('2')
      expect(result).toContain('3')
    })

    it('renders homogeneous array with item count and samples', () => {
      const items = Array.from({ length: 150 }, (_, i) => ({
        file: `src/file${i}.ts`,
        match: `import something${i}`,
      }))
      const result = describeShape(items)
      expect(result).toContain('<150 items>')
      expect(result).toContain('"file":')
      expect(result).toContain('"match":')
      expect(result).toMatch(/\.\.\.1\d+ more/)
    })

    it('collapses when budget is extremely tight', () => {
      const items = Array.from({ length: 50 }, () => ({ a: 1 }))
      const result = describeShape(items, 1)
      expect(result).toBe('[<50 items>...]')
    })
  })

  describe('nested structures', () => {
    it('handles nested object with mixed sizes', () => {
      const obj = {
        results: Array.from({ length: 50 }, (_, i) => ({
          id: i,
          name: `item${i}`,
          data: 'y'.repeat(3000),
        })),
        total: 50,
        page: 1,
      }
      const result = describeShape(obj)
      expect(result).toContain('"total": 50')
      expect(result).toContain('"page": 1')
      expect(result).toContain('"results":')
      expect(result).toContain('<50 items>')
    })

    it('handles deeply nested within budget', () => {
      const obj = { a: { b: { c: { d: 1 } } } }
      const result = describeShape(obj)
      expect(result).toContain('"d": 1')
    })
  })

  describe('edge cases', () => {
    it('handles large array of primitives', () => {
      const arr = Array.from({ length: 1000 }, (_, i) => i)
      const result = describeShape(arr)
      expect(result).toContain('<1000 items>')
    })

    it('renders small array of primitives fully', () => {
      const arr = Array.from({ length: 5 }, (_, i) => i)
      const result = describeShape(arr)
      expect(result).toContain('0')
      expect(result).toContain('4')
    })

    it('handles object with undefined values (filtered)', () => {
      const obj = { a: 1, b: undefined, c: 3 }
      const result = describeShape(obj)
      expect(result).toContain('"a": 1')
      expect(result).toContain('"c": 3')
      expect(result).not.toContain('"b"')
    })
  })
})
