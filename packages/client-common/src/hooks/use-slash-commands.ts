import { useState, useCallback } from 'react'
import { filterSlashCommands } from '../commands/command-router'
import type { SlashCommandDefinition } from '../commands/slash-commands'
import type { KeyEvent } from '../types/key-event'

interface SlashCommandsState {
  /** Whether the slash command menu is currently open */
  isSlashMenuOpen: boolean
  /** Filtered list of matching commands */
  filteredCommands: SlashCommandDefinition[]
  /** Index of the currently highlighted command */
  selectedIndex: number
  /** Set highlighted index (used by mouse hover) */
  setSelectedIndex: (index: number) => void
  /** Key intercept handler to pass to MultilineInput.onKeyIntercept */
  handleKeyIntercept: (key: KeyEvent) => boolean
  /** Get the command string for the currently selected item (e.g., "/exit") */
  getSelectedCommandText: () => string | null
}

export type SlashCommandMenuAction =
  | { readonly _tag: 'Select'; readonly index: number }
  | { readonly _tag: 'Execute'; readonly commandText: string }
  | { readonly _tag: 'Dismiss' }

export function getSlashCommandSuggestions(inputText: string): SlashCommandDefinition[] {
  const trimmed = inputText.trim()
  if (!trimmed.startsWith('/') || trimmed.includes(' ')) return []
  return filterSlashCommands(trimmed.slice(1))
}

export function getSlashCommandMenuAction(
  key: KeyEvent,
  commands: ReadonlyArray<SlashCommandDefinition>,
  selectedIndex: number,
): SlashCommandMenuAction | null {
  if (commands.length === 0) return null

  const withoutModifiers = !key.ctrl && !key.meta && !key.option
  if (key.name === 'up' && withoutModifiers) {
    return { _tag: 'Select', index: Math.max(0, selectedIndex - 1) }
  }
  if (key.name === 'down' && withoutModifiers) {
    return { _tag: 'Select', index: Math.min(commands.length - 1, selectedIndex + 1) }
  }
  if (
    (key.name === 'return' || key.name === 'enter' || key.name === 'tab') &&
    withoutModifiers &&
    !key.shift
  ) {
    const command = commands[selectedIndex]
    return command ? { _tag: 'Execute', commandText: `/${command.id}` } : null
  }
  if (key.name === 'escape') return { _tag: 'Dismiss' }
  return null
}

/**
 * Hook that manages slash command suggestion menu state.
 *
 * The menu opens when input starts with '/' and has no spaces.
 * Arrow keys navigate, Enter selects, Escape closes.
 */
export function useSlashCommands(
  inputText: string,
  onExecute: (commandText: string) => void,
): SlashCommandsState {
  const [selection, setSelection] = useState({ signature: '', index: 0 })

  const filteredCommands = getSlashCommandSuggestions(inputText)
  const isSlashMenuOpen = filteredCommands.length > 0

  const signature = filteredCommands.map(c => c.id).join(',')
  const maximumIndex = Math.max(0, filteredCommands.length - 1)
  const selectedIndex = selection.signature === signature
    ? Math.min(maximumIndex, Math.max(0, selection.index))
    : 0
  const setSelectedIndex = useCallback((index: number) => {
    setSelection({ signature, index: Math.min(maximumIndex, Math.max(0, index)) })
  }, [maximumIndex, signature])

  const getSelectedCommandText = useCallback((): string | null => {
    if (!isSlashMenuOpen || filteredCommands.length === 0) return null
    const cmd = filteredCommands[selectedIndex]
    if (!cmd) return null
    return `/${cmd.id}`
  }, [isSlashMenuOpen, filteredCommands, selectedIndex])

  const handleKeyIntercept = useCallback((key: KeyEvent): boolean => {
    if (!isSlashMenuOpen) return false
    const action = getSlashCommandMenuAction(key, filteredCommands, selectedIndex)
    if (!action) return false

    if (action._tag === 'Select') setSelectedIndex(action.index)
    if (action._tag === 'Execute') onExecute(action.commandText)
    return true
  }, [isSlashMenuOpen, filteredCommands, selectedIndex, onExecute, setSelectedIndex])

  return {
    isSlashMenuOpen,
    filteredCommands,
    selectedIndex,
    setSelectedIndex,
    handleKeyIntercept,
    getSelectedCommandText,
  }
}
