import { describe, expect, mock, test } from 'bun:test'
import React, { useEffect } from 'react'
import { act, create, type ReactTestRenderer } from 'react-test-renderer'
import { useSlashCommands } from './use-slash-commands'

type HookSnapshot = ReturnType<typeof useSlashCommands>

function renderHook(inputText: string, onExecute: (commandText: string) => void) {
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
    renderer = create(<Harness text={inputText} />)
  })

  const getSnapshot = () => {
    if (!snapshot) throw new Error('Hook snapshot not initialized')
    return snapshot
  }

  const updateText = (text: string) => {
    act(() => {
      renderer!.update(<Harness text={text} />)
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
    const onExecute = mock(() => {})
    const hook = renderHook('/ne', onExecute)

    const snapshot = hook.getSnapshot()
    expect(snapshot.isSlashMenuOpen).toBe(true)
    expect(snapshot.filteredCommands[0]?.id).toBe('new')

    act(() => {
      snapshot.handleKeyIntercept({
        name: 'return',
        sequence: '\r',
        ctrl: false,
        meta: false,
        option: false,
        shift: false,
      })
    })

    expect(onExecute).toHaveBeenCalledWith('/new')
    hook.cleanup()
  })

  test('Tab confirms currently selected slash command', () => {
    const onExecute = mock(() => {})
    const hook = renderHook('/mo', onExecute)

    act(() => {
      hook.getSnapshot().handleKeyIntercept({
        name: 'tab',
        sequence: '\t',
        ctrl: false,
        meta: false,
        option: false,
        shift: false,
      })
    })

    expect(onExecute).toHaveBeenCalledWith('/model')
    hook.cleanup()
  })

  test('menu closes for slash input containing spaces', () => {
    const onExecute = mock(() => {})
    const hook = renderHook('/new now', onExecute)
    expect(hook.getSnapshot().isSlashMenuOpen).toBe(false)

    hook.updateText('/new')
    expect(hook.getSnapshot().isSlashMenuOpen).toBe(true)
    hook.cleanup()
  })
})
