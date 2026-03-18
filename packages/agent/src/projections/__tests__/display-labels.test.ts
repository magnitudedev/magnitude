import { describe, expect, test } from 'bun:test'
import { generateToolLabel } from '../display'

describe('display tool label generation', () => {
  test('formats shell labels in `$ cmd` form', () => {
    expect(generateToolLabel('shell', { command: 'bun test cli/src/utils/live-activity.test.ts' }))
      .toBe('$ bun test cli/src/utils/live-activity.test.ts')
  })

  test('truncates long shell command labels while preserving `$ ` prefix', () => {
    const longCommand = 'x'.repeat(80)
    const label = generateToolLabel('shell', { command: longCommand })
    expect(label.startsWith('$ ')).toBe(true)
    expect(label.length).toBe(52)
    expect(label.endsWith('...')).toBe(true)
  })
})