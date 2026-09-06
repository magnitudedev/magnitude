import { act, useState } from 'react'
import { testRender } from '@opentui/react/test-utils'
import { DirectoryPathSchema } from '@magnitudedev/sdk'
import type { RecentChat } from '@magnitudedev/client-common'
import { expect, test, vi } from 'vitest'
import { defaultCliThemes } from '../../utils/theme'

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => defaultCliThemes.dark,
}))

const { RecentChatsOverlay } = await import('./recent-chats-overlay')

const makeChat = (index: number): RecentChat => ({
  id: `chat-${index}`,
  title: `Conversation ${String(index).padStart(2, '0')}`,
  lastMessage: '',
  timestamp: 0,
  messageCount: index,
  workingDirectory: DirectoryPathSchema.make('C:/project'),
  archived: false,
  pinnedAt: null,
})

const renderOverlay = (options: {
  readonly chats?: RecentChat[]
  readonly hasMore?: boolean
  readonly isLoading?: boolean
  readonly error?: string | null
  readonly loadMore?: () => void
  readonly onSelect?: (chat: RecentChat) => void
} = {}) => testRender(
  <RecentChatsOverlay
    onClose={() => {}}
    onSelect={options.onSelect ?? (() => {})}
    chats={options.chats ?? []}
    hasMore={options.hasMore ?? false}
    isLoading={options.isLoading ?? false}
    error={options.error ?? null}
    loadMore={options.loadMore ?? (() => {})}
  />,
  { width: 96, height: 18 },
)

test.each([
  {
    error: null,
    visible: 'No recent conversations found.',
    hidden: 'Failed to load conversations.',
  },
  {
    error: 'Failed to load conversations.',
    visible: 'Failed to load conversations.',
    hidden: 'No recent conversations found.',
  },
])('renders the expected empty-list state', async ({ error, visible, hidden }) => {
  const view = await renderOverlay({ error })

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain(visible)
    expect(frame).not.toContain(hidden)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test('keeps keyboard selection visible and valid as the list changes', async () => {
  const chats = Array.from({ length: 24 }, (_, index) => makeChat(index + 1))
  const onSelect = vi.fn()
  let replaceChats: (next: RecentChat[]) => void = () => {}

  const Harness = () => {
    const [currentChats, setCurrentChats] = useState(chats)
    replaceChats = setCurrentChats
    return (
      <RecentChatsOverlay
        onClose={() => {}}
        onSelect={onSelect}
        chats={currentChats}
        hasMore={false}
        isLoading={false}
        error={null}
        loadMore={() => {}}
      />
    )
  }

  const view = await testRender(<Harness />, { width: 96, height: 18 })

  try {
    await act(view.renderOnce)
    for (let index = 0; index < 14; index += 1) {
      await act(async () => view.mockInput.pressArrow('down'))
    }
    await act(view.renderOnce)

    expect(view.captureCharFrame()).toContain('> Conversation 15')
    await act(async () => view.mockInput.pressEnter())
    expect(onSelect).toHaveBeenLastCalledWith(chats[14])

    await act(async () => replaceChats(chats.slice(0, 3)))
    await act(view.renderOnce)
    await act(async () => view.mockInput.pressEnter())
    expect(onSelect).toHaveBeenLastCalledWith(chats[2])
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test('loads another page when the first page does not fill the viewport', async () => {
  const loadMore = vi.fn()
  const view = await renderOverlay({ chats: [makeChat(1)], hasMore: true, loadMore })

  try {
    await act(view.renderOnce)
    expect(loadMore).toHaveBeenCalled()
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
