import { expect, test } from 'vitest'
import type { SlashCommandDefinition } from '@magnitudedev/client-common'
import { getSlashCommandRowLayout } from './slash-menu'

const command: SlashCommandDefinition = {
  id: 'formulate-abstractions',
  label: 'formulate-abstractions',
  description: 'Formulate clear semantic abstractions from desired behavior',
  source: 'skill',
}

test('reserves the full skill name and truncates the description to the remaining row width', () => {
  const row = getSlashCommandRowLayout(command, 40)

  expect(row.label).toBe('/formulate-abstractions')
  expect(row.description).toBe('Formulate cle…')
  expect(`${row.label} ${row.description}`).toBe('/formulate-abstractions Formulate cle…')
})

test('omits the description rather than truncating the skill name when space is exhausted', () => {
  const row = getSlashCommandRowLayout(command, 20)

  expect(row.label).toBe('/formulate-abstractions')
  expect(row.description).toBe('')
})
