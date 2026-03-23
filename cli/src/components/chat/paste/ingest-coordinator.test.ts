import { describe, expect, test } from 'bun:test'
import type { KeyEvent } from '@opentui/core'
import { createPasteIngestCoordinator, isPasteFallbackKey } from './ingest-coordinator'
import type { PasteIngestOutcome } from './types'

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

function defer<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((r) => {
    resolve = r
  })
  return { promise, resolve }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

describe('isPasteFallbackKey', () => {
  test('matches Ctrl+V and Cmd+V fallback shortcuts', () => {
    expect(isPasteFallbackKey(createKey({ name: 'v', ctrl: true }))).toBe(true)
    expect(isPasteFallbackKey(createKey({ name: 'V', ctrl: true }))).toBe(true)
    expect(isPasteFallbackKey(createKey({ name: 'v', meta: true }))).toBe(true)
  })
})

describe('createPasteIngestCoordinator', () => {
  test('native before fallback emits native event once', async () => {
    const outcomes: PasteIngestOutcome[] = []
    const coordinator = createPasteIngestCoordinator({
      fallbackDelayMs: 20,
      requestFallbackPaste: () => true,
      onOutcome: (outcome) => outcomes.push(outcome),
    })

    coordinator.handleKey(createKey({ name: 'v', ctrl: true }))
    await sleep(10)
    coordinator.handleNativeEvent({ text: 'native' })
    await sleep(30)

    expect(outcomes).toEqual([{ kind: 'native-event', text: 'native', replayedFromDeferred: false }])
    coordinator.dispose()
  })

  test('fallback then native in-flight replays when fallback empty', async () => {
    const outcomes: PasteIngestOutcome[] = []
    const fallback = defer<boolean>()
    const coordinator = createPasteIngestCoordinator({
      fallbackDelayMs: 10,
      requestFallbackPaste: () => fallback.promise,
      onOutcome: (outcome) => outcomes.push(outcome),
    })

    coordinator.handleKey(createKey({ name: 'v', ctrl: true }))
    await sleep(20)
    coordinator.handleNativeEvent({ text: 'native-late' })
    fallback.resolve(false)
    await Promise.resolve()
    await Promise.resolve()

    expect(outcomes).toEqual([
      { kind: 'fallback-requested' },
      { kind: 'native-event', text: 'native-late', replayedFromDeferred: true },
    ])
    coordinator.dispose()
  })

  test('fallback success drops late native duplicate', async () => {
    const outcomes: PasteIngestOutcome[] = []
    const coordinator = createPasteIngestCoordinator({
      fallbackDelayMs: 10,
      requestFallbackPaste: () => true,
      onOutcome: (outcome) => outcomes.push(outcome),
    })

    coordinator.handleKey(createKey({ name: 'v', ctrl: true }))
    await sleep(20)
    coordinator.handleNativeEvent({ text: 'native-late' })
    await Promise.resolve()

    expect(outcomes).toEqual([
      { kind: 'fallback-requested' },
      { kind: 'dropped', reason: 'native-duplicate-after-fallback-success' },
    ])
    coordinator.dispose()
  })

  test('native without active attempt is deterministic', () => {
    const outcomes: PasteIngestOutcome[] = []
    const coordinator = createPasteIngestCoordinator({
      requestFallbackPaste: () => true,
      onOutcome: (outcome) => outcomes.push(outcome),
    })

    coordinator.handleNativeEvent({ text: 'native-only' })
    expect(outcomes).toEqual([{ kind: 'native-event', text: 'native-only', replayedFromDeferred: false }])
    coordinator.dispose()
  })

  test('dispose cancels pending fallback', async () => {
    const outcomes: PasteIngestOutcome[] = []
    const coordinator = createPasteIngestCoordinator({
      fallbackDelayMs: 10,
      requestFallbackPaste: () => true,
      onOutcome: (outcome) => outcomes.push(outcome),
    })

    coordinator.handleKey(createKey({ name: 'v', ctrl: true }))
    coordinator.dispose()
    await sleep(25)

    expect(outcomes).toEqual([])
  })
})
