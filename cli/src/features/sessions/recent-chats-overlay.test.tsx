import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import { expect, test, vi } from 'vitest'

vi.mock('@opentui/react', () => ({
  useKeyboard: () => {},
  useRenderer: () => ({
    requestRender: () => {},
  }),
}))

vi.mock('@magnitudedev/client-common', async () => {
  const actual = await vi.importActual<typeof import('@magnitudedev/client-common')>(
    '@magnitudedev/client-common',
  )
  return {
    ...actual,
    subscribeAnimationTick: () => () => {},
    getAnimationTickSnapshot: () => 0,
  }
})

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => ({
    primary: '#00aaff',
    foreground: '#ffffff',
    muted: '#888888',
    border: '#444444',
    error: '#ff0000',
  }),
}))

vi.mock('../../components/button', () => ({
  Button: ({ children }: { children?: ReactNode }) => <button>{children}</button>,
}))

const { RecentChatsOverlay } = await import('./recent-chats-overlay')

const render = (error: string | null): string =>
  renderToStaticMarkup(
    <RecentChatsOverlay
      onClose={() => {}}
      onSelect={() => {}}
      chats={[]}
      hasMore={false}
      isLoading={false}
      error={error}
      loadMore={() => {}}
    />,
  )

test('shows the empty state only after a successful empty list', () => {
  const html = render(null)
  expect(html).toContain('No recent conversations found.')
  expect(html).not.toContain('Failed to load conversations.')
})

test('shows a list-level failure instead of the empty state', () => {
  const html = render('Failed to load conversations.')
  expect(html).toContain('Failed to load conversations.')
  expect(html).not.toContain('No recent conversations found.')
})
