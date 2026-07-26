import { expect, test, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import type { KeyEvent } from '@opentui/core'

const keyboard = vi.hoisted(() => ({
  handler: null as ((key: KeyEvent) => void) | null,
}))

vi.mock('@opentui/react', () => ({
  useKeyboard: (handler: (key: KeyEvent) => void) => {
    keyboard.handler = handler
  },
}))

const { ChatSurfaceKeyboard } = await import('./chat-surface-keyboard')

function renderKeyboard(overrides: Partial<Parameters<typeof ChatSurfaceKeyboard>[0]> = {}): (key: KeyEvent) => void {
  keyboard.handler = null
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
      thinkingOpen={false}
      thinkingOptionCount={0}
      onOpenThinking={() => {}}
      onMoveThinking={() => {}}
      onApplyThinking={() => {}}
      onCancelThinking={() => {}}
      {...overrides}
    />,
  )
  if (!keyboard.handler) throw new Error('keyboard handler not registered')
  return keyboard.handler as (key: KeyEvent) => void
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

function makeKey(name: string, ctrl = false) {
  let prevented = false
  const key = {
    name,
    sequence: '',
    ctrl,
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

test('Ctrl-T opens the thinking selector when the model exposes choices', () => {
  let opened = 0
  const handler = renderKeyboard({
    thinkingOptionCount: 4,
    onOpenThinking: () => { opened += 1 },
  })

  const { key, wasPrevented } = makeKey('t', true)
  handler(key)

  expect(opened).toBe(1)
  expect(wasPrevented()).toBe(true)
})

test('Ctrl-T advances the preview instead of closing an open selector', () => {
  const movements: number[] = []
  let opened = 0
  let applied = 0
  const handler = renderKeyboard({
    thinkingOpen: true,
    thinkingOptionCount: 4,
    onOpenThinking: () => { opened += 1 },
    onMoveThinking: (direction) => { movements.push(direction) },
    onApplyThinking: () => { applied += 1 },
  })

  const { key, wasPrevented } = makeKey('t', true)
  handler(key)

  expect(movements).toEqual([1])
  expect(opened).toBe(0)
  expect(applied).toBe(0)
  expect(wasPrevented()).toBe(true)
})

test('thinking selector owns arrow, Enter, and Escape keys while open', () => {
  const movements: number[] = []
  let applied = 0
  let cancelled = 0
  let interrupted = 0
  const handler = renderKeyboard({
    status: 'streaming',
    thinkingOpen: true,
    thinkingOptionCount: 4,
    onMoveThinking: (direction) => { movements.push(direction) },
    onApplyThinking: () => { applied += 1 },
    onCancelThinking: () => { cancelled += 1 },
    onInterrupt: () => { interrupted += 1 },
  })

  handler(makeKey('up').key)
  handler(makeKey('left').key)
  handler(makeKey('down').key)
  handler(makeKey('right').key)
  handler(makeKey('return').key)
  handler(makeEscapeKey().key)

  expect(movements).toEqual([-1, -1, 1, 1])
  expect(applied).toBe(1)
  expect(cancelled).toBe(1)
  expect(interrupted).toBe(0)
})
