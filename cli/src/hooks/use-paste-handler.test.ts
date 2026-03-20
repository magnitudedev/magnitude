import { describe, expect, test } from 'bun:test'
import type { KeyEvent } from '@opentui/core'
import { createPasteFallbackController, isPasteFallbackKey } from './use-paste-handler'

function createKey(overrides: Partial<KeyEvent> = {}): KeyEvent {
  return {
    name: '',
    sequence: '',
    ctrl: false,
    meta: false,
    option: false,
    shift: false,
    ...overrides,
  } as KeyEvent
}

describe('isPasteFallbackKey', () => {
  test('matches Ctrl+V fallback', () => {
    expect(isPasteFallbackKey(createKey({ name: 'v', ctrl: true }))).toBe(true)
    expect(isPasteFallbackKey(createKey({ name: 'V', ctrl: true }))).toBe(true)
  })

  test('does not blanket-intercept Cmd+V on macOS', () => {
    expect(isPasteFallbackKey(createKey({ name: 'v', meta: true }))).toBe(false)
  })

  test('rejects modified or non-v keys', () => {
    expect(isPasteFallbackKey(createKey({ name: 'v', ctrl: true, option: true }))).toBe(false)
    expect(isPasteFallbackKey(createKey({ name: 'v', ctrl: true, meta: true }))).toBe(false)
    expect(isPasteFallbackKey(createKey({ name: 'c', ctrl: true }))).toBe(false)
  })
})

describe('createPasteFallbackController', () => {
  test('prevents duplicate insertion when fallback key is followed by native paste event', async () => {
    const calls: Array<string | undefined> = []
    const controller = createPasteFallbackController((text) => calls.push(text), 25)

    const handled = controller.handlePasteKey(createKey({ name: 'v', ctrl: true }))
    expect(handled).toBe(true)

    controller.handlePasteEvent({ text: 'native-text' })
    await new Promise((resolve) => setTimeout(resolve, 40))

    expect(calls).toEqual(['native-text'])
    controller.dispose()
  })

  test('fires fallback paste when no native paste event arrives', async () => {
    const calls: Array<string | undefined> = []
    const controller = createPasteFallbackController((text) => calls.push(text), 10)

    const handled = controller.handlePasteKey(createKey({ name: 'v', ctrl: true }))
    expect(handled).toBe(true)

    await new Promise((resolve) => setTimeout(resolve, 25))
    expect(calls).toEqual([undefined])
    controller.dispose()
  })

  test('dispose cancels pending fallback timer', async () => {
    const calls: Array<string | undefined> = []
    const controller = createPasteFallbackController((text) => calls.push(text), 10)

    controller.handlePasteKey(createKey({ name: 'v', ctrl: true }))
    controller.dispose()

    await new Promise((resolve) => setTimeout(resolve, 25))
    expect(calls).toEqual([])
  })
})
