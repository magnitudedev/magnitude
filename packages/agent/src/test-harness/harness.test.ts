import { describe, test, expect } from 'bun:test'
import { createAgentTestHarness } from './harness'
import type { AppEvent } from '../events'

describe('Agent test harness integration', () => {
  test('files seed available on harness', async () => {
    const harness = await createAgentTestHarness({
      files: { 'README.md': 'hello\nworld' },
    })
    try {
      expect(harness.files.get('README.md')).toBe('hello\nworld')
      harness.files.set('notes.txt', 'ok')
      expect(harness.files.get('notes.txt')).toBe('ok')
    } finally {
      await harness.dispose()
    }
  })

  test('harness creates without tool overrides', async () => {
    const harness = await createAgentTestHarness()
    try {
      expect(harness).toBeTruthy()
    } finally {
      await harness.dispose()
    }
  })
})