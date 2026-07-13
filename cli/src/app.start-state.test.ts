import { expect, test } from 'bun:test'
import { hasConversationActivity } from './utils/start-state'

test('hasConversationActivity is true when there are display messages', () => {
  expect(hasConversationActivity({ displayMessageCount: 1, bashOutputCount: 0 })).toBe(true)
})

test('hasConversationActivity is true when first interaction is bash output', () => {
  expect(hasConversationActivity({ displayMessageCount: 0, bashOutputCount: 1 })).toBe(true)
})

test('hasConversationActivity is false when no messages and no bash output', () => {
  expect(hasConversationActivity({ displayMessageCount: 0, bashOutputCount: 0 })).toBe(false)
})
