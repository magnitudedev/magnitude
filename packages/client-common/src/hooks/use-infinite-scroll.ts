/**
 * useInfiniteScroll — composable infinite scroll for cursor-paginated lists.
 *
 * Combines viewport-fill detection and scroll-threshold triggering. Uses
 * the same `TimelineScrollAdapter` interface as the chat timeline's
 * `TimelineScrollController`, so each platform (CLI OpenTUI, web DOM) builds
 * its adapter and passes it in.
 *
 * For display-view-reshaped surfaces (chat timeline), use
 * `TimelineScrollController` directly — it has anchoring and bottom-following
 * concerns that this hook doesn't cover.
 *
 * Uses `useAtomMount` for the subscription lifecycle per AGENTS.md.
 */
import { useMemo, useRef } from "react"
import { Effect } from "effect"
import { Atom, useAtomMount } from "@effect-atom/atom-react"
import type { TimelineScrollAdapter } from "../display-view-controller/timeline-scroll-controller"

export interface InfiniteScrollSource {
  readonly hasMore: boolean
  readonly loadingMore: boolean
  readonly loadMore: () => void
}

export interface UseInfiniteScrollOptions {
  /** Platform-specific scroll metrics adapter (OpenTUI scrollbox or DOM element). */
  readonly adapter: TimelineScrollAdapter
  /** Paginated data source state. */
  readonly source: InfiniteScrollSource
  /** Which edge triggers loading: "top" for chat-timeline-style, "bottom" for list-style. */
  readonly direction: "top" | "bottom"
  /** Enable viewport-fill (proactive loading until content fills viewport). Default: true. */
  readonly fillViewport?: boolean
}

/**
 * Check if more content should be loaded based on scroll metrics.
 * Pure function — no side effects.
 */
function shouldLoadMore(
  metrics: ReturnType<TimelineScrollAdapter["getScrollMetrics"]>,
  source: InfiniteScrollSource,
  direction: "top" | "bottom",
  loadThreshold: number,
  fillViewport: boolean,
): boolean {
  if (!metrics) return false
  if (!source.hasMore) return false
  if (source.loadingMore) return false

  const maxScrollTop = metrics.scrollHeight - metrics.viewportHeight

  // Viewport-fill: content doesn't fill the viewport → load more
  if (fillViewport && maxScrollTop <= 0) return true

  // Threshold: user scrolled near the target edge
  if (direction === "top") {
    return metrics.scrollTop <= loadThreshold
  }
  return maxScrollTop - metrics.scrollTop <= loadThreshold
}

export function useInfiniteScroll({
  adapter,
  source,
  direction,
  fillViewport = true,
}: UseInfiniteScrollOptions): void {
  // Keep latest source in a ref so the subscription closure always reads fresh values
  // without re-subscribing on every render.
  const sourceRef = useRef(source)
  sourceRef.current = source

  const scrollAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          const unsub = adapter.subscribeActivity(() => {
            const current = sourceRef.current
            if (shouldLoadMore(
              adapter.getScrollMetrics(),
              current,
              direction,
              adapter.loadThreshold,
              fillViewport,
            )) {
              current.loadMore()
            }
          })

          // Check immediately on mount — content might not fill the viewport
          const current = sourceRef.current
          if (shouldLoadMore(
            adapter.getScrollMetrics(),
            current,
            direction,
            adapter.loadThreshold,
            fillViewport,
          )) {
            current.loadMore()
          }

          yield* Effect.addFinalizer(() => Effect.sync(unsub))
        }),
      ),
    [adapter, direction, fillViewport],
  )

  useAtomMount(scrollAtom)
}
