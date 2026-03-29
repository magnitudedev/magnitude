import { describe, test, expect } from 'bun:test'
import { Effect } from 'effect'
import { createAgentTestHarness } from './harness'
import type { AppEvent } from '../events'
import { InMemoryChatPersistenceTag } from './in-memory-persistence'
import { MockTurnScriptTag } from './turn-script'

describe('Agent test harness integration', () => {
  test('basic harness lifecycle: boots and disposes cleanly', async () => {
    const harness = await createAgentTestHarness()
    try {
      const initEvent = await harness.wait.event('session_initialized')
      expect(initEvent.type).toBe('session_initialized')
      expect(initEvent.forkId).toBeNull()
      expect(harness.events().some((e: AppEvent) => e.type === 'session_initialized')).toBe(true)
    } finally {
      await harness.dispose()
    }
  })

  test('simple mock turn: scripted xml produces expected turn completion', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (script) =>
          script.enqueue({ xml: '<comms><message to="user">hello</message></comms><yield/>' }, null)
        )
      )

      await harness.user('run scripted turn')
      const completed = await harness.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_completed')
      expect(completed.result.success).toBe(true)

      const textPart = completed.responseParts.find((p) => p.type === 'text')
      expect(textPart?.type).toBe('text')
      if (textPart?.type === 'text') {
        expect(textPart.content).toContain('hello')
        expect(textPart.content).toContain('<yield/>')
      }
    } finally {
      await harness.dispose()
    }
  })

  test('persists emitted events in in-memory persistence', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (script) =>
          script.enqueue({ xml: '<comms><message to="user">persist check</message></comms><yield/>' }, null)
        )
      )

      await harness.user('persist check prompt')
      await harness.wait.idle(null)
      await new Promise((resolve) => setTimeout(resolve, 200))

      const persisted = await harness.runEffect(
        Effect.flatMap(InMemoryChatPersistenceTag, (persistence) => persistence.inspectEvents())
      )

      expect(persisted.length).toBeGreaterThan(0)
      expect(persisted.some((e: AppEvent) => e.type === 'session_initialized')).toBe(true)
      expect(persisted.some((e: AppEvent) => e.type === 'turn_started' && e.forkId === null)).toBe(true)
      expect(persisted.some((e: AppEvent) => e.type === 'turn_completed' && e.forkId === null)).toBe(true)
    } finally {
      await harness.dispose()
    }
  })

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