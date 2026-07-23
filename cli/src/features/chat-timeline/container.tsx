/**
 * ChatTimeline feature container (spec §5.6) — the scrollback: message
 * history, streaming, and pagination via display-view reshaping.
 *
 * Scroll behavior lives in TimelineScrollController (plain TS, no React):
 * OpenTUI owns interactive scrolling; the controller declares the data
 * window, follows the bottom, and holds the viewport's bottom-distance across
 * window changes. This component builds the metrics adapter and ties the
 * controller's lifetime to the scrollbox via a callback ref.
 *
 * The `header` slot (logo, tip, recent-chats) renders only when the loaded
 * window truly starts at the beginning of the session. While older history
 * exists, the top of the scrollback is a spacer that shows a loading line
 * whenever a page is in flight.
 */
import { useCallback, useRef, useSyncExternalStore, useMemo, type ReactNode } from 'react'
import { Atom, useAtomMount } from '@effect-atom/atom-react'
import { Effect } from 'effect'
import { TextAttributes } from '@opentui/core'
import type { ScrollBoxRenderable } from '@opentui/core'
import {
  useDisplayState,
  getFork,
  useDisplayViewController,
  useDisplayViewControllerCore,
  useDisplayReader,
  useRootHistoryLoading,
  subscribeSystemMessages,
  getSystemMessagesSnapshot,
  TimelineScrollController,
  type TimelineScrollAdapter,
} from '@magnitudedev/client-common'
import { safeRenderableAccess } from '../../utils/safe-renderable-access'
import { subscribeScrollboxActivity } from '../../utils/scroll-helpers'
import { useTheme } from '../../hooks/use-theme'
import type { ActionId } from '../../types/ui-actions'
import { useOpenFile } from '../file-viewer/container'
import { ChatScrollbox } from './scrollbox'
import { ChatTimeline } from './timeline'

export function ChatTimelineContainer({
  header,
  chatColumnWidth,
  dispatchErrorAction,
  isOverlayActive,
}: {
  header: ReactNode
  chatColumnWidth: number
  dispatchErrorAction: (actionId: ActionId) => void
  isOverlayActive: boolean
}): ReactNode {
  const theme = useTheme()
  const { pushFork } = useDisplayViewController()
  const core = useDisplayViewControllerCore()
  const reader = useDisplayReader()
  const isLoadingMore = useRootHistoryLoading()
  const openFile = useOpenFile()

  const rootTimeline = useDisplayState((state) => getFork(state, null) ?? null)
  const hasMoreBefore = rootTimeline?.window.hasMoreBefore ?? false

  const systemMessages = useSyncExternalStore(subscribeSystemMessages, getSystemMessagesSnapshot)

  // ── Scroll: TimelineScrollController over a metrics adapter ─────────────
  const scrollboxRef = useRef<ScrollBoxRenderable | null>(null)
  const scrollControllerRef = useRef<TimelineScrollController | null>(null)

  const adapter = useMemo<TimelineScrollAdapter>(
    () => ({
      getScrollMetrics: () => {
        const sb = scrollboxRef.current
        if (!sb) return null
        return safeRenderableAccess<ScrollBoxRenderable, { scrollTop: number; viewportHeight: number; scrollHeight: number } | null>(
          sb,
          (s) => ({
            scrollTop: s.scrollTop ?? 0,
            viewportHeight: s.viewport?.height ?? 0,
            scrollHeight: s.scrollHeight ?? 0,
          }),
          { fallback: null },
        )
      },
      setScrollTop: (value) => {
        const sb = scrollboxRef.current
        if (!sb) return
        safeRenderableAccess(
          sb,
          (s) => {
            s.scrollTo(Math.max(0, value))
          },
          { fallback: undefined },
        )
      },
      subscribeActivity: (handler) => subscribeScrollboxActivity(scrollboxRef.current, handler),
      stickyThreshold: 2,
      loadThreshold: 3,
    }),
    [],
  )

  // Callback ref: the scrollbox's mount/unmount IS the controller's lifetime.
  const attachScrollbox = useCallback(
    (sb: ScrollBoxRenderable | null) => {
      scrollboxRef.current = sb
      if (sb) {
        if (scrollControllerRef.current === null) {
          const controller = new TimelineScrollController({ adapter, core, reader, forkId: null })
          controller.init()
          scrollControllerRef.current = controller
        }
      } else if (scrollControllerRef.current !== null) {
        scrollControllerRef.current.dispose()
        scrollControllerRef.current = null
      }
    },
    [adapter, core, reader],
  )

  // Suspend/resume the scroll controller when an overlay hides the timeline.
  // This is view lifecycle management (not scroll behavior): while suspended,
  // the controller preserves all state — window position, scroll distance,
  // followingBottom — so the user returns to exactly what they left.
  const suspendResumeAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          const controller = scrollControllerRef.current
          if (!controller) return
          if (isOverlayActive) {
            controller.suspend()
          } else {
            controller.resume()
          }
          yield* Effect.addFinalizer(() =>
            Effect.sync(() => {
              controller.resume()
            }),
          )
        }),
      ),
    [isOverlayActive],
  )
  useAtomMount(suspendResumeAtom)

  return (
    <ChatScrollbox scrollRef={attachScrollbox} hasMoreBefore={hasMoreBefore}>
      {hasMoreBefore ? (
        // Fixed-height slot: one centered loading line + one blank row below.
        // Constant height whether or not a load is in flight, so the text
        // toggling never shifts the content.
        <box style={{ height: 2, alignItems: 'center' }}>
          {isLoadingMore && (
            <text style={{ fg: theme.muted }}>
              <span attributes={TextAttributes.DIM}>Loading earlier messages…</span>
            </text>
          )}
        </box>
      ) : (
        header
      )}
      <ChatTimeline
        timeline={rootTimeline}
        chatColumnWidth={chatColumnWidth}
        themeErrorColor={theme.error}
        systemMessages={systemMessages}
        onFileClick={openFile}
        onForkExpand={pushFork}
        onErrorAction={dispatchErrorAction}
      />
    </ChatScrollbox>
  )
}
