/**
 * Sessions feature (spec §5.6) — session selection for the terminal app:
 * the startup recent-chats widget and the full-screen recent-chats overlay.
 *
 * Session switching goes through the shared session action contract; the
 * display view controller owns the stream transition. No remount.
 */
import { useCallback, useRef, type ReactNode } from 'react'
import { Effect, Option, Runtime } from 'effect'
import { useAtomValue, useAtomSet, Result } from '@effect-atom/atom-react'
import {
  useAgentClient,
  useDisplayState,
  getFork,
  useSessionsList,
  useRecentChatsNavigation,
  useSessionActions,
  pendingUserSubmitAtom,
  bashOutputsAtom,
  composerHasContentAtom,
  settingsOpenAtom,
  usageOpenAtom,
  sessionsToRecentChats,
  type RecentChat,
  type RecentChatsPage,
} from '@magnitudedev/client-common'
import type { SessionMetadata } from '@magnitudedev/sdk'
import { showRecentChatsOverlayAtom } from '../../state/cli-atoms'
import { hasConversationActivity } from '../../utils/start-state'
import { RecentChatsWidget } from './recent-chats-widget'
import { RecentChatsOverlay } from './recent-chats-overlay'

/** Switch the app to a session. Stream, store, and title all react to the atom. */
export function useResumeSession(): (sessionId: string) => void {
  const setShowOverlay = useAtomSet(showRecentChatsOverlayAtom)
  const { resumeSession } = useSessionActions()

  return useCallback((sessionId: string) => {
    setShowOverlay(false)
    resumeSession(sessionId)
  }, [setShowOverlay, resumeSession])
}

export interface RecentChatsWidgetState {
  chats: RecentChat[] | null
  widgetNavActive: boolean
  navigation: ReturnType<typeof useRecentChatsNavigation>
  resumeChat: (chat: RecentChat) => void
  hasActivity: boolean
}

/**
 * Widget state hook — called once by the orchestrator because the widget's
 * keyboard navigation is forwarded through the composer's input handler.
 */
export function useRecentChatsWidgetState(): RecentChatsWidgetState {
  const { loading, sessions } = useSessionsList({ limit: 5 })
  const chats = loading ? null : sessionsToRecentChats(sessions)

  const showOverlay = useAtomValue(showRecentChatsOverlayAtom)
  const settingsOpen = useAtomValue(settingsOpenAtom)
  const usageOpen = useAtomValue(usageOpenAtom)
  const composerHasContent = useAtomValue(composerHasContentAtom)
  const pendingUserSubmit = useAtomValue(pendingUserSubmitAtom)
  const bashOutputs = useAtomValue(bashOutputsAtom)
  const messageCount = useDisplayState((state) => getFork(state, null)?.messages.order.length ?? 0)

  const hasActivity = pendingUserSubmit || hasConversationActivity({
    displayMessageCount: messageCount,
    bashOutputCount: bashOutputs.length,
  })

  const resumeSession = useResumeSession()
  const resumeChat = useCallback((chat: RecentChat) => resumeSession(chat.id), [resumeSession])

  const widgetNavActive = !showOverlay && !settingsOpen && !usageOpen && !hasActivity && !composerHasContent
  const navigation = useRecentChatsNavigation(chats ? chats.slice(0, 5) : [], resumeChat, widgetNavActive)

  return { chats, widgetNavActive, navigation, resumeChat, hasActivity }
}

export function RecentChatsWidgetView({ state }: { state: RecentChatsWidgetState }): ReactNode {
  const setShowOverlay = useAtomSet(showRecentChatsOverlayAtom)
  return (
    <RecentChatsWidget
      chats={state.chats ?? []}
      loading={state.chats === null}
      selectedIndex={state.navigation.selectedIndex}
      onSelect={state.resumeChat}
      onHoverIndex={state.navigation.setSelectedIndex}
      onOpenAll={() => setShowOverlay(true)}
      isNavigationActive={state.widgetNavActive}
    />
  )
}

export function RecentChatsOverlayContainer(): ReactNode {
  const client = useAgentClient()
  const runtimeResult = useAtomValue(client.runtime)
  const setShowOverlay = useAtomSet(showRecentChatsOverlayAtom)
  const resumeSession = useResumeSession()

  // The protocol pages by cursor; the overlay pages by offset. The cursor for
  // the next page is threaded here — offset 0 restarts, any other offset
  // continues from the last result's cursor.
  const nextCursorRef = useRef<string | null>(null)

  const loadPage = useCallback(async (offset: number, limit: number): Promise<RecentChatsPage> => {
    if (!Result.isSuccess(runtimeResult)) return { items: [], hasMore: false }
    const cursor = offset === 0 ? Option.none<string>() : Option.fromNullable(nextCursorRef.current)
    const result = await Runtime.runPromise(runtimeResult.value)(
      Effect.flatMap(client, (c) =>
        c('ListSessions', { cwd: Option.none(), query: Option.none(), cursor, limit })
      )
    )
    nextCursorRef.current = Option.getOrNull(result.nextCursor)
    return { items: sessionsToRecentChats([...result.items]), hasMore: result.hasMore }
  }, [client, runtimeResult])

  return (
    <RecentChatsOverlay
      onClose={() => setShowOverlay(false)}
      onSelect={(chat) => resumeSession(chat.id)}
      loadPage={loadPage}
    />
  )
}
