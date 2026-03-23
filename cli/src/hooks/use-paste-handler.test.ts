import { describe, expect, test } from 'bun:test'
import type { KeyEvent } from '@opentui/core'
import { createPasteFallbackController } from './use-paste-handler'
import { isPasteFallbackKey } from '../components/chat/paste/ingest-coordinator'

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

  test('rejects ctrl+meta+v and option-modified variants', () => {
    expect(isPasteFallbackKey(createKey({ name: 'v', ctrl: true, meta: true }))).toBe(false)
    expect(isPasteFallbackKey(createKey({ name: 'v', ctrl: true, option: true }))).toBe(false)
    expect(isPasteFallbackKey(createKey({ name: 'v', meta: true, option: true }))).toBe(false)
  })

  test('rejects non-v keys', () => {
    expect(isPasteFallbackKey(createKey({ name: 'c', ctrl: true }))).toBe(false)
    expect(isPasteFallbackKey(createKey({ name: 'c', meta: true }))).toBe(false)
  })
})

describe('createPasteFallbackController', () => {
  test('fallback-empty then late native inserts once from native', async () => {
    const calls: Array<string | undefined> = []
    const controller = createPasteFallbackController((text) => {
      calls.push(text)
      return Boolean(text)
    }, 10)

    controller.handlePasteKey(createKey({ name: 'v', ctrl: true }))
    await sleep(20)
    controller.handlePasteEvent({ text: 'native-late' })
    await Promise.resolve()

    expect(calls).toEqual([undefined, 'native-late'])
    controller.dispose()
  })

  test('fallback-success then late native is dropped', async () => {
    const calls: Array<string | undefined> = []
    const controller = createPasteFallbackController((text) => {
      calls.push(text)
      return true
    }, 10)

    controller.handlePasteKey(createKey({ name: 'v', ctrl: true }))
    await sleep(20)
    controller.handlePasteEvent({ text: 'native-late' })
    await Promise.resolve()

    expect(calls).toEqual([undefined])
    controller.dispose()
  })

  test('native while fallback in-flight is dropped when fallback later succeeds', async () => {
    const calls: Array<string | undefined> = []
    const fallback = defer<boolean>()
    const controller = createPasteFallbackController((text) => {
      calls.push(text)
      if (text === undefined) return fallback.promise
      return true
    }, 10)

    controller.handlePasteKey(createKey({ name: 'v', ctrl: true }))
    await sleep(20)
    controller.handlePasteEvent({ text: 'native-during-inflight' })
    await Promise.resolve()

    fallback.resolve(true)
    await Promise.resolve()
    await Promise.resolve()

    expect(calls).toEqual([undefined])
    controller.dispose()
  })

  test('native while fallback in-flight is replayed when fallback later resolves empty', async () => {
    const calls: Array<string | undefined> = []
    const fallback = defer<boolean>()
    const controller = createPasteFallbackController((text) => {
      calls.push(text)
      if (text === undefined) return fallback.promise
      return Boolean(text)
    }, 10)

    controller.handlePasteKey(createKey({ name: 'v', ctrl: true }))
    await sleep(20)
    controller.handlePasteEvent({ text: 'native-during-inflight-empty' })
    await Promise.resolve()

    fallback.resolve(false)
    await Promise.resolve()
    await Promise.resolve()

    expect(calls).toEqual([undefined, 'native-during-inflight-empty'])
    controller.dispose()
  })

  test('near-deadline native paste still yields single insertion', async () => {
    const calls: Array<string | undefined> = []
    const controller = createPasteFallbackController((text) => {
      calls.push(text)
      return Boolean(text)
    }, 20)

    controller.handlePasteKey(createKey({ name: 'v', ctrl: true }))
    await sleep(18)
    controller.handlePasteEvent({ text: 'native-near-deadline' })
    await sleep(30)

    expect(calls).toEqual(['native-near-deadline'])
    controller.dispose()
  })

  test('native paste without key path still inserts', async () => {
    const calls: Array<string | undefined> = []
    const controller = createPasteFallbackController((text) => {
      calls.push(text)
      return true
    }, 10)

    controller.handlePasteEvent({ text: 'native-only' })
    await Promise.resolve()

    expect(calls).toEqual(['native-only'])
    controller.dispose()
  })

  test('dispose cancels pending fallback timer', async () => {
    const calls: Array<string | undefined> = []
    const controller = createPasteFallbackController((text) => {
      calls.push(text)
      return true
    }, 10)

    controller.handlePasteKey(createKey({ name: 'v', ctrl: true }))
    controller.dispose()

    await sleep(25)
    expect(calls).toEqual([])
  })

  test('callback updates do not reset pending attempt state', async () => {
    const calls: Array<string | undefined> = []
    let callback = (text?: string) => {
      calls.push(text)
      return Boolean(text)
    }

    const controller = createPasteFallbackController((text) => callback(text), 10)

    controller.handlePasteKey(createKey({ name: 'v', ctrl: true }))
    callback = (text?: string) => {
      calls.push(text)
      return Boolean(text)
    }

    await sleep(20)
    controller.handlePasteEvent({ text: 'native-late' })
    await Promise.resolve()

    expect(calls).toEqual([undefined, 'native-late'])
    controller.dispose()
  })
})
