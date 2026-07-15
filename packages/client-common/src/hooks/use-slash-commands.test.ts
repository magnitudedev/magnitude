import { describe, expect, test } from 'vitest'
import type { KeyEvent } from '../types/key-event'
import {
  getSlashCommandMenuAction,
  getSlashCommandSuggestions,
} from './use-slash-commands'

const key = (name: string): KeyEvent => ({
  name,
  ctrl: false,
  meta: false,
  option: false,
  shift: false,
})

describe('slash command menu', () => {
  test.each([
    ['return', '/ne', '/new'],
    ['tab', '/in', '/init'],
  ])('%s executes the selected command', (keyName, input, commandText) => {
    const commands = getSlashCommandSuggestions(input)
    expect(getSlashCommandMenuAction(key(keyName), commands, 0)).toEqual({
      _tag: 'Execute',
      commandText,
    })
  })

  test('input containing spaces closes the suggestion menu', () => {
    expect(getSlashCommandSuggestions('/new now')).toEqual([])
    expect(getSlashCommandSuggestions('/new')[0]?.id).toBe('new')
  })

  test('navigation is bounded by the available commands', () => {
    const commands = getSlashCommandSuggestions('/')
    expect(getSlashCommandMenuAction(key('up'), commands, 0)).toEqual({
      _tag: 'Select',
      index: 0,
    })
    expect(getSlashCommandMenuAction(key('down'), commands, commands.length - 1)).toEqual({
      _tag: 'Select',
      index: commands.length - 1,
    })
  })
})
