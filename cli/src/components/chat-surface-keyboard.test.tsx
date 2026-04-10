import { test, expect, mock } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { KeyEvent } from '@opentui/core'

let keyboardHandler: ((key: KeyEvent) => void) | null = null

mock.module('@opentui/react', () => ({
  useKeyboard: (handler: (key: KeyEvent) => void) => {
    keyboardHandler = handler
  },
}))

const { ChatSurfaceKeyboard } = await import('./chat-surface-keyboard')

function renderKeyboard(overrides: Partial<Parameters<typeof ChatSurfaceKeyboard>[0]> = {}): (key: KeyEvent) => void {
  keyboardHandler = null
  renderToStaticMarkup(
    <ChatSurfaceKeyboard
      status="idle"
      hasRunningForks={false}
      isBlockingOverlayActive={false}
      nextEscWillKillAll={false}
      setNextEscWillKillAll={() => {}}
      killAllTimeoutRef={{ current: null }}
      onInterrupt={() => {}}
      onInterruptAll={() => {}}
      composerHasContent={false}
      onClearInput={() => {}}
      bashMode={false}
      onExitBashMode={() => {}}
      pendingApproval={false}
      onApprove={() => {}}
      onReject={() => {}}
      {...overrides}
    />,
  )
  if (!keyboardHandler) throw new Error('keyboard handler not registered')
  return keyboardHandler as (key: KeyEvent) => void
}

function makeCtrlCKey() {
  let prevented = false
  const key = {
    name: 'c',
    sequence: '',
    ctrl: true,
    meta: false,
    option: false,
    shift: false,
    defaultPrevented: false,
    preventDefault() {
      prevented = true
    },
  } as unknown as KeyEvent
  return { key, wasPrevented: () => prevented }
}

function makeEscapeKey() {
  let prevented = false
  const key = {
    name: 'escape',
    sequence: '',
    ctrl: false,
    meta: false,
    option: false,
    shift: false,
    defaultPrevented: false,
    preventDefault() {
      prevented = true
    },
  } as unknown as KeyEvent
  return { key, wasPrevented: () => prevented }
}

test('Ctrl-C clears composer when composer has content', () => {
  let cleared = 0
  const handler = renderKeyboard({
    composerHasContent: true,
    onClearInput: () => { cleared += 1 },
  })

  const { key, wasPrevented } = makeCtrlCKey()
  handler(key)

  expect(cleared).toBe(1)
  expect(wasPrevented()).toBe(true)
})

test('Escape no longer clears composer content', () => {
  let cleared = 0
  const handler = renderKeyboard({
    composerHasContent: true,
    onClearInput: () => { cleared += 1 },
  })

  const { key, wasPrevented } = makeEscapeKey()
  handler(key)

  expect(cleared).toBe(0)
  expect(wasPrevented()).toBe(false)
})

test('Escape interrupts when streaming and composer is empty', () => {
  let interrupted = 0
  const handler = renderKeyboard({
    status: 'streaming',
    composerHasContent: false,
    onInterrupt: () => { interrupted += 1 },
  })

  const { key, wasPrevented } = makeEscapeKey()
  handler(key)

  expect(interrupted).toBe(1)
  expect(wasPrevented()).toBe(true)
})

test('Ctrl-C interrupts when streaming and composer is empty', () => {
  let interrupted = 0
  const handler = renderKeyboard({
    status: 'streaming',
    composerHasContent: false,
    onInterrupt: () => { interrupted += 1 },
  })

  const { key, wasPrevented } = makeCtrlCKey()
  handler(key)

  expect(interrupted).toBe(1)
  expect(wasPrevented()).toBe(true)
})

test('keyboard handler no-ops when blocking overlay is active', () => {
  let interrupted = 0
  let interruptedAll = 0
  let setKillAll = 0
  const handler = renderKeyboard({
    status: 'streaming',
    hasRunningForks: true,
    isBlockingOverlayActive: true,
    nextEscWillKillAll: true,
    onInterrupt: () => { interrupted += 1 },
    onInterruptAll: () => { interruptedAll += 1 },
    setNextEscWillKillAll: () => { setKillAll += 1 },
  })

  const { key, wasPrevented } = makeEscapeKey()
  handler(key)

  expect(interrupted).toBe(0)
  expect(interruptedAll).toBe(0)
  expect(setKillAll).toBe(0)
  expect(wasPrevented()).toBe(false)
})