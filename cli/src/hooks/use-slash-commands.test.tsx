import { describe, expect, mock, test } from 'bun:test'
import React, { useEffect } from 'react'
import { act, create, type ReactTestRenderer } from 'react-test-renderer'
import type { KeyEvent } from '@opentui/core'
import {
  slashCommandHandled,
  slashCommandUnhandled,
  useSlashCommands,
  type SlashCommandOutcome,
} from '@magnitudedev/client-common'

type HookSnapshot = ReturnType<typeof useSlashCommands>

function renderHook(inputText: string, onExecute: (commandText: string) => SlashCommandOutcome) {
  let snapshot: HookSnapshot | null = null
  let renderer: ReactTestRenderer | null = null

  function Harness({ text }: { text: string }) {
    const value = useSlashCommands(text, onExecute)
    useEffect(() => {
      snapshot = value
    }, [value])
    return null
  }

  act(() => {
    renderer = create(React.createElement(Harness, { text: inputText }))
  })

  const getSnapshot = () => {
    if (!snapshot) throw new Error('Hook snapshot not initialized')
    return snapshot
  }

  const updateText = (text: string) => {
    act(() => {
      renderer!.update(React.createElement(Harness, { text }))
    })
  }

  const cleanup = () => {
    act(() => {
      renderer?.unmount()
    })
  }

  return { getSnapshot, updateText, cleanup }
}

describe('useSlashCommands', () => {
  test('Enter confirms currently selected slash command', () => {
    const onExecute = mock(() => slashCommandHandled)
    const hook = renderHook('/ne', onExecute)

    const snapshot = hook.getSnapshot()
    expect(snapshot.isSlashMenuOpen).toBe(true)
    expect(snapshot.filteredCommands[0]?.id).toBe('new')

    let intercepted = false
    act(() => {
      intercepted = snapshot.handleKeyIntercept({
        name: 'return',
        sequence: '\r',
        ctrl: false,
        meta: false,
        option: false,
        shift: false,
      } as unknown as KeyEvent)
    })

    expect(onExecute).toHaveBeenCalledWith('/new')
    expect(intercepted).toBe(true)
    hook.cleanup()
  })

  test('Tab confirms currently selected slash command', () => {
    const onExecute = mock(() => slashCommandHandled)
    const hook = renderHook('/in', onExecute)

    act(() => {
      hook.getSnapshot().handleKeyIntercept({
        name: 'tab',
        sequence: '\t',
        ctrl: false,
        meta: false,
        option: false,
        shift: false,
      } as unknown as KeyEvent)
    })

    expect(onExecute).toHaveBeenCalledWith('/init')
    hook.cleanup()
  })

  test('menu closes for slash input containing spaces', () => {
    const onExecute = mock(() => slashCommandHandled)
    const hook = renderHook('/new now', onExecute)
    expect(hook.getSnapshot().isSlashMenuOpen).toBe(false)

    hook.updateText('/new')
    expect(hook.getSnapshot().isSlashMenuOpen).toBe(true)
    hook.cleanup()
  })

  test('does not consume Enter when command execution is unhandled', () => {
    const onExecute = mock(() => slashCommandUnhandled)
    const hook = renderHook('/new', onExecute)

    const intercepted = hook.getSnapshot().handleKeyIntercept({
      name: 'return',
      sequence: '\r',
      ctrl: false,
      meta: false,
      option: false,
      shift: false,
    } as unknown as KeyEvent)

    expect(onExecute).toHaveBeenCalledWith('/new')
    expect(intercepted).toBe(false)
    hook.cleanup()
  })
})
