import { afterEach, describe, expect, test, vi } from 'vitest'
import React, { act, useEffect } from 'react'
import { create, type ReactTestRenderer } from 'react-test-renderer'
import type { KeyEvent } from '@opentui/core'
import {
  registerSkillCommands,
  useSlashCommands,
  type SlashCommandConfirmation,
} from '@magnitudedev/client-common'

type HookSnapshot = ReturnType<typeof useSlashCommands>
const mountedRenderers: ReactTestRenderer[] = []

;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true

function renderHook(
  inputText: string,
  onConfirm: (confirmation: SlashCommandConfirmation) => void,
) {
  let snapshot: HookSnapshot | null = null
  let renderer: ReactTestRenderer | null = null

  function Harness({ text }: { text: string }) {
    const value = useSlashCommands(text, onConfirm)
    useEffect(() => {
      snapshot = value
    }, [value])
    return null
  }

  act(() => {
    renderer = create(React.createElement(Harness, { text: inputText }))
  })
  mountedRenderers.push(renderer!)

  const getSnapshot = () => {
    if (!snapshot) throw new Error('Hook snapshot not initialized')
    return snapshot
  }

  const updateText = (text: string) => {
    act(() => {
      renderer!.update(React.createElement(Harness, { text }))
    })
  }

  return { getSnapshot, updateText }
}

afterEach(() => {
  act(() => {
    for (const renderer of mountedRenderers.splice(0)) renderer.unmount()
  })
  registerSkillCommands([])
  vi.clearAllMocks()
})

describe('useSlashCommands', () => {
  test('Enter confirms currently selected slash command', () => {
    const onConfirm = vi.fn((_confirmation: SlashCommandConfirmation) => {})
    const hook = renderHook('/ne', onConfirm)

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
      } as unknown as KeyEvent)
    })

    expect(onConfirm).toHaveBeenCalledWith({
      _tag: 'ExecuteCommand',
      commandText: '/new',
    })
  })

  test('Tab confirms currently selected slash command', () => {
    const onConfirm = vi.fn((_confirmation: SlashCommandConfirmation) => {})
    const hook = renderHook('/in', onConfirm)

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

    expect(onConfirm).toHaveBeenCalledWith({
      _tag: 'ExecuteCommand',
      commandText: '/init',
    })
  })

  test.each(['return', 'tab'] as const)(
    '%s inserts a selected skill into the draft without executing it',
    (keyName) => {
      registerSkillCommands([{
        id: 'githits-code',
        label: 'githits-code',
        description: 'Inspect public source with GitHits',
        source: 'skill',
        skillPath: '/skills/githits-code/SKILL.md',
      }])
      const onConfirm = vi.fn((_confirmation: SlashCommandConfirmation) => {})
      const hook = renderHook('/githits', onConfirm)

      act(() => {
        hook.getSnapshot().handleKeyIntercept({
          name: keyName,
          sequence: keyName === 'tab' ? '\t' : '\r',
          ctrl: false,
          meta: false,
          option: false,
          shift: false,
        } as unknown as KeyEvent)
      })

      expect(onConfirm).toHaveBeenCalledWith({
        _tag: 'PopulateDraft',
        text: '/githits-code ',
      })
    },
  )

  test('menu closes for slash input containing spaces', () => {
    const onConfirm = vi.fn((_confirmation: SlashCommandConfirmation) => {})
    const hook = renderHook('/new now', onConfirm)
    expect(hook.getSnapshot().isSlashMenuOpen).toBe(false)

    hook.updateText('/new')
    expect(hook.getSnapshot().isSlashMenuOpen).toBe(true)
  })
})
