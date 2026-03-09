import { useState, useCallback, useEffect, useRef } from 'react'
import { filterSlashCommands } from '../commands/command-router'
import type { SlashCommandDefinition } from '../commands/slash-commands'
import type { KeyEvent } from '@opentui/core'

interface SlashCommandsState {
  /** Whether the slash command menu is currently open */
  isSlashMenuOpen: boolean
  /** Filtered list of matching commands */
  filteredCommands: SlashCommandDefinition[]
  /** Index of the currently highlighted command */
  selectedIndex: number
  /** Key intercept handler to pass to MultilineInput.onKeyIntercept */
  handleKeyIntercept: (key: KeyEvent) => boolean
  /** Get the command string for the currently selected item (e.g., "/exit") */
  getSelectedCommandText: () => string | null
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
  const [selectedIndex, setSelectedIndex] = useState(0)

  // Determine if the menu should be open and what the query is
  const trimmed = inputText.trim()
  const isSlashInput = trimmed.startsWith('/') && !trimmed.includes(' ')
  const query = isSlashInput ? trimmed.slice(1) : ''
  const filteredCommands = isSlashInput ? filterSlashCommands(query) : []
  const isSlashMenuOpen = isSlashInput && filteredCommands.length > 0

  // Track previous filtered command ids to reset selectedIndex when list changes
  const prevFilteredIdsRef = useRef<string>('')
  useEffect(() => {
    const currentIds = filteredCommands.map(c => c.id).join(',')
    if (currentIds !== prevFilteredIdsRef.current) {
      prevFilteredIdsRef.current = currentIds
      setSelectedIndex(0)
    }
  }, [filteredCommands])

  const getSelectedCommandText = useCallback((): string | null => {
    if (!isSlashMenuOpen || filteredCommands.length === 0) return null
    const cmd = filteredCommands[selectedIndex]
    if (!cmd) return null
    return `/${cmd.id}`
  }, [isSlashMenuOpen, filteredCommands, selectedIndex])

  const handleKeyIntercept = useCallback((key: KeyEvent): boolean => {
    if (!isSlashMenuOpen) return false

    const isUp = key.name === 'up' && !key.ctrl && !key.meta && !key.option
    const isDown = key.name === 'down' && !key.ctrl && !key.meta && !key.option
    const isEnter = (key.name === 'return' || key.name === 'enter') &&
      !key.shift && !key.ctrl && !key.meta && !key.option
    const isEscape = key.name === 'escape'
    const isTab = key.name === 'tab' && !key.shift && !key.ctrl && !key.meta && !key.option

    if (isUp) {
      setSelectedIndex(prev => Math.max(0, prev - 1))
      return true
    }

    if (isDown) {
      setSelectedIndex(prev => Math.min(filteredCommands.length - 1, prev + 1))
      return true
    }

    if (isEnter || isTab) {
      const cmd = filteredCommands[selectedIndex]
      if (cmd) {
        onExecute(`/${cmd.id}`)
      }
      return true
    }

    if (isEscape) {
      // Returning true consumes the key; the input text still has '/' but
      // the parent can handle clearing if needed
      return true
    }

    return false
  }, [isSlashMenuOpen, filteredCommands, selectedIndex, onExecute])

  return {
    isSlashMenuOpen,
    filteredCommands,
    selectedIndex,
    handleKeyIntercept,
    getSelectedCommandText,
  }
}
