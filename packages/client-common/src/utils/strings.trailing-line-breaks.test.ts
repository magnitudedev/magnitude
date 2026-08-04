import { describe, expect, test } from 'vitest'
import { stripTrailingLineBreaks } from './strings'

describe('stripTrailingLineBreaks', () => {
  test('removes trailing line breaks', () => {
    expect(stripTrailingLineBreaks('hello\n')).toBe('hello')
    expect(stripTrailingLineBreaks('hello\n\n')).toBe('hello')
    expect(stripTrailingLineBreaks('hello\r\n')).toBe('hello')
  })

  test('preserves internal line breaks', () => {
    expect(stripTrailingLineBreaks('hello\nworld')).toBe('hello\nworld')
  })

  test('does not trim other trailing whitespace', () => {
    expect(stripTrailingLineBreaks('hello  ')).toBe('hello  ')
  })
})
