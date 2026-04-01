import { test, expect } from 'bun:test'

type OverlayKind =
  | 'none'
  | 'recent-chats'
  | 'fork-detail'
  | 'settings'
  | 'setup-browser'
  | 'setup-wizard'
  | 'browser-setup'
  | 'auth-method'
  | 'local-provider'
  | 'api-key'
  | 'oauth'

function canToggleRecentChatsWithCtrlR(activeOverlayKind: OverlayKind): boolean {
  return activeOverlayKind === 'none' || activeOverlayKind === 'recent-chats'
}

test('Ctrl+R toggles when no overlay is open', () => {
  expect(canToggleRecentChatsWithCtrlR('none')).toBe(true)
})

test('Ctrl+R closes recent chats when recent chats overlay is open', () => {
  expect(canToggleRecentChatsWithCtrlR('recent-chats')).toBe(true)
})

test('Ctrl+R is ignored for non-recent-chats overlays', () => {
  const blockedKinds: OverlayKind[] = [
    'fork-detail',
    'settings',
    'setup-browser',
    'setup-wizard',
    'browser-setup',
    'auth-method',
    'local-provider',
    'api-key',
    'oauth',
  ]
  for (const kind of blockedKinds) {
    expect(canToggleRecentChatsWithCtrlR(kind)).toBe(false)
  }
})
