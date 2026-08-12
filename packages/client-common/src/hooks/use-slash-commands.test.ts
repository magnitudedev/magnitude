import { afterEach, describe, expect, test } from 'vitest'
import type { KeyEvent } from '../types/key-event'
import { registerSkillCommands } from '../commands/slash-commands'
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

afterEach(() => {
  registerSkillCommands([])
})

describe('slash command menu', () => {
  test.each([
    ['return', '/ne', '/new'],
    ['tab', '/in', '/init'],
  ])('%s executes the selected command', (keyName, input, commandText) => {
    const commands = getSlashCommandSuggestions(input)
    expect(getSlashCommandMenuAction(key(keyName), commands, 0)).toEqual({
      _tag: 'Confirm',
      confirmation: { _tag: 'ExecuteCommand', commandText },
    })
  })

  test.each(['return', 'tab'])('%s populates the draft for a selected skill', (keyName) => {
    registerSkillCommands([{
      id: 'githits-code',
      label: 'githits-code',
      description: 'Inspect public source with GitHits',
      source: 'skill',
      skillPath: '/skills/githits-code/SKILL.md',
    }])

    const commands = getSlashCommandSuggestions('/githits')
    expect(getSlashCommandMenuAction(key(keyName), commands, 0)).toEqual({
      _tag: 'Confirm',
      confirmation: { _tag: 'PopulateDraft', text: '/githits-code ' },
    })
  })

  test('input containing spaces closes the suggestion menu', () => {
    expect(getSlashCommandSuggestions('/new now')).toEqual([])
    expect(getSlashCommandSuggestions('/githits-code ')).toEqual([])
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
