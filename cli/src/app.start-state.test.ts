import { expect, test } from 'vitest'
import { hasConversationActivity } from './utils/start-state'

test('hasConversationActivity is true when there are display messages', () => {
  expect(hasConversationActivity(1)).toBe(true)
})

test('hasConversationActivity is false when there are no display messages', () => {
  expect(hasConversationActivity(0)).toBe(false)
})
