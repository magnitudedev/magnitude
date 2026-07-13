/**
 * TimelineScrollController — plain-TS owner of history loading, window
 * anchoring, and bottom-following for the chat timeline. No React. The
 * component constructs it, calls init() via a callback ref, and calls
 * dispose() on unmount; everything else happens through direct subscriptions.
 *
 * Lifecycle:
 *   init() / dispose() — the controller exists or it doesn't. Subscribes to
 *     reader and core, initializes state. The container creates the
 *     controller when the scrollbox mounts and disposes when it unmounts.
 *   suspend() / resume() — the controller is active or dormant. When the
 *     timeline is hidden behind an overlay, the container calls suspend():
 *     the controller clears timers, ignores scroll activity, but preserves
 *     all state (window position, followingBottom, scroll distance). On
 *     resume(), it restores the viewport from preserved state — the user
 *     sees exactly what they left.
 *
 * The controller is the SOLE scroll writer besides the user. It owns four
 * concerns:
 *
 *   1. LOAD — on scroll activity, one comparison: near the top, more history
 *      exists, nothing in flight → declare one more page.
 *   2. ANCHOR — across a window change while not at the bottom, restore the
 *      viewport by DISTANCE FROM THE BOTTOM of the content, exactly once.
 *   3. FOLLOW — when content grows at the tail and the user is at the
 *      bottom, keep the viewport pinned to the bottom.
 *   4. SUFFICIENCY — when the visible root window has older history but
 *      renders too short to create a usable scroll range, request one more
 *      page. A view must load enough data to be navigable without depending
 *      on a scroll event the user cannot physically produce.
 *
 * Why bottom-distance anchoring: prepending content above the viewport does
 * not change how far the visible content is from the bottom. So the restore
 * is `scrollTop = scrollHeight - viewport - d`, written ONE TIME in the next
 * resize event — which fires AFTER the surface's own layout settles,
 * pre-paint. The store listener that detects the change fires synchronously
 * on accept, BEFORE React re-renders, so pre-change geometry is still
 * readable there; the resize event that consumes the anchor fires after
 * layout settles. Both land in the same frame, so there is no race and no
 * timer needed. Scroll-to-tail is the same mechanism with d = 0.
 *
 * Bottom-following uses a `followingBottom` flag: set true when the user is
 * at the bottom (on scroll events and on initial init / session switch),
 * cleared when the user scrolls away from the bottom. On each `"resize"`
 * event while `followingBottom` is true, `armAnchor(0)` scrolls to the new
 * bottom. The flag is essential because `isAtBottom()` checked on a resize
 * event sees POST-growth geometry — the content just grew, so the viewport
 * is no longer at the bottom by definition.
 *
 * The adapter passes an `ActivityKind` so the controller can distinguish
 * user scroll (`"scroll"`) from content size changes (`"resize"`). Anchors
 * are consumed only on `"resize"` events — a `"scroll"` event while an
 * anchor is pending means the user scrolled before layout settled; the
 * anchor is cleared without writing, letting the user's position stand.
 * Self-triggered scroll events (from the controller's own `setScrollTop`)
 * are suppressed via a guard flag so they don't clear anchors or toggle
 * `followingBottom`.
 *
 * There is intentionally no automatic bottom-idle eviction here. Shrinking an
 * active root window behind the user's back can make the visible view invalid
 * again. Shape shrink/reset belongs to session/view lifecycle or a future
 * invariant-preserving compaction path.
 */

import type { DisplayTimeline } from '@magnitudedev/sdk'
import { getFork } from '../sync/get-fork'
import type { DisplayReader } from '../sync/display-view-store'
import {
  INCREMENTAL_ROOT_PAGE_SIZE,
} from '../sync/display-view-shape'
import type { DisplayMode, DisplayViewControllerCore } from './controller'

// ─── Adapter contract ──────────────────────────────────────────────────────

export interface ScrollMetrics {
  readonly scrollTop: number
  readonly viewportHeight: number
  readonly scrollHeight: number
}

export interface NormalizedScrollMetrics {
  readonly rawScrollTop: number
  readonly scrollTop: number
  readonly viewportHeight: number
  readonly scrollHeight: number
  readonly maxScrollTop: number
  readonly distFromBottom: number
}

/** What kind of activity fired — distinguishes user scroll from content changes. */
export type ActivityKind = "scroll" | "resize"

export interface TimelineScrollAdapter {
  readonly getScrollMetrics: () => ScrollMetrics | null

  /** Set the scroll position, in surface units. The surface clamps. */
  readonly setScrollTop: (value: number) => void

  /**
   * Notify the controller of scroll AND content-size activity. The handler
   * does one comparison or one subtraction — never attach anything that
   * does work per event.
   * CLI: scrollbar "change" + content "resize" renderable events.
   * Web: scroll events + a content ResizeObserver.
   */
  readonly subscribeActivity: (handler: (kind: ActivityKind) => void) => () => void

  /** Distance from bottom below which the user counts as at-the-bottom. Surface units. */
  readonly stickyThreshold: number

  /** Distance from the top below which scroll-up triggers a history load. Surface units. */
  readonly loadThreshold: number
}

export interface TimelineScrollControllerOptions {
  readonly adapter: TimelineScrollAdapter
  readonly core: DisplayViewControllerCore
  readonly reader: DisplayReader
  /** null for root; forks have fixed limits — anchoring only, no loading/eviction. */
  readonly forkId: string | null
}

// ─── Controller ─────────────────────────────────────────────────────────────

export class TimelineScrollController {
  private readonly adapter: TimelineScrollAdapter
  private readonly core: DisplayViewControllerCore
  private readonly reader: DisplayReader
  private readonly forkId: string | null

  private unsubscribes: Array<() => void> = []
  private prevWindowStart: number | null = null
  private prevSessionId: string | null = null
  private prevDisplayMode: DisplayMode = 'default'
  /** A pending one-shot anchor restore: bottom-distance to apply once. */
  private anchorPending = false
  private pendingDistFromBottom: number | null = null
  /**
   * Whether the viewport should follow the bottom as content grows. Set when
   * the user is at the bottom; cleared when they scroll away. Checked on
   * resize events (where post-growth geometry makes isAtBottom() unreliable).
   */
  private followingBottom = true
  /** Guard: suppress the next self-triggered scroll event from setScrollTop. */
  private suppressScroll = false
  /** Whether the controller is dormant (timeline hidden behind an overlay). */
  private suspended = false
  /** Bottom-distance captured on suspend, used to restore position on resume. */
  private preservedDistFromBottom: number | null = null

  constructor(options: TimelineScrollControllerOptions) {
    this.adapter = options.adapter
    this.core = options.core
    this.reader = options.reader
    this.forkId = options.forkId
  }

  init = (): void => {
    const snapshot = this.core.getSnapshot()
    this.prevSessionId = snapshot.selectedSessionId
    this.prevDisplayMode = snapshot.displayMode
    this.prevWindowStart = null
    this.unsubscribes = [
      this.adapter.subscribeActivity(this.onActivity),
      this.reader.subscribe(this.onDisplayChange),
      this.core.subscribe(this.onCoreChange),
    ]
    this.followingBottom = true
    this.armAnchor(0) // start at the tail
    this.reconcileRootShape()
  }

  dispose = (): void => {
    for (const unsubscribe of this.unsubscribes) unsubscribe()
    this.unsubscribes = []
    this.clearAnchor()
  }

  // ── Suspend / resume: dormant while the timeline is hidden ──────────────
  // The container calls suspend() when an overlay opens (before the scrollbox
  // goes blind) and resume() when it closes. While suspended, the controller
  // preserves all state and ignores scroll activity — no eviction, no loading,
  // no anchoring. On resume, it restores the viewport from the captured
  // bottom-distance so the user sees exactly what they left.

  suspend = (): void => {
    if (this.suspended) return
    this.suspended = true
    this.preservedDistFromBottom = this.currentDistFromBottom()
    this.clearAnchor()
  }

  resume = (): void => {
    if (!this.suspended) return
    this.suspended = false
    if (this.followingBottom) {
      this.armAnchor(0)
    } else if (this.preservedDistFromBottom !== null) {
      this.armAnchor(this.preservedDistFromBottom)
    }
    this.preservedDistFromBottom = null
    this.reconcileRootShape()
  }

  // ── Snapshot reads ────────────────────────────────────────────────────────

  private timeline(): DisplayTimeline | null {
    return getFork(this.reader.getSnapshot().state, this.forkId) ?? null
  }

  private acceptedRootLimit(): number | null {
    const root = this.reader.getSnapshot().shape.timelines.root
    return root !== undefined && root.kind === 'tail' ? root.limit : null
  }

  private shapeInFlight(): boolean {
    const accepted = this.acceptedRootLimit()
    return accepted === null || accepted !== this.core.getSnapshot().rootTailLimit
  }

  private isAtBottom(): boolean {
    const metrics = this.normalizedMetrics()
    if (!metrics) return false
    return metrics.distFromBottom <= this.adapter.stickyThreshold
  }

  private normalizedMetrics(): NormalizedScrollMetrics | null {
    const raw = this.adapter.getScrollMetrics()
    if (!raw) return null
    const viewportHeight = Math.max(0, raw.viewportHeight)
    if (viewportHeight <= 0) return null
    const scrollHeight = Math.max(0, raw.scrollHeight)
    const maxScrollTop = Math.max(0, scrollHeight - viewportHeight)
    const scrollTop = Math.min(Math.max(0, raw.scrollTop), maxScrollTop)
    return {
      rawScrollTop: raw.scrollTop,
      scrollTop,
      viewportHeight,
      scrollHeight,
      maxScrollTop,
      distFromBottom: maxScrollTop - scrollTop,
    }
  }

  // ── The anchor: one-shot bottom-distance restore ──────────────────────────
  // The store listener arms the anchor synchronously on accept (pre-render,
  // pre-change geometry readable). The next activity/resize event consumes
  // it with ONE write, after layout settles, pre-paint. No timer, no repeat,
  // no blocking user scroll.

  private armAnchor(distFromBottom: number): void {
    this.pendingDistFromBottom = distFromBottom
    this.anchorPending = true
    // Optimistically apply now — if the new layout has already settled (e.g.
    // the change was accepted in the same frame the surface measured), this
    // is the restore. Otherwise it is a no-op (same geometry) and the next
    // resize event does the real work.
    this.applyAnchor()
  }

  private clearAnchor(): void {
    this.anchorPending = false
    this.pendingDistFromBottom = null
  }

  private applyAnchor(): void {
    const d = this.pendingDistFromBottom
    if (d === null) return
    const metrics = this.normalizedMetrics()
    if (!metrics) return
    const target = Math.min(Math.max(0, metrics.maxScrollTop - d), metrics.maxScrollTop)
    if (target !== metrics.rawScrollTop) {
      // Suppress the self-triggered scroll event so it doesn't clear the
      // anchor or toggle followingBottom. On the web, `scrollTop = value`
      // fires a synchronous scroll event that consumes the guard. On the
      // TUI, `scrollTo` does NOT emit "change" (only user interaction does),
      // so the guard would leak — the setTimeout safety net clears it on the
      // next microtask, ensuring the next real user scroll isn't swallowed.
      this.suppressScroll = true
      this.adapter.setScrollTop(target)
      setTimeout(() => { this.suppressScroll = false }, 0)
    }
  }

  /** Current bottom-distance, from pre-change geometry. */
  private currentDistFromBottom(): number | null {
    const metrics = this.normalizedMetrics()
    if (!metrics) return null
    return metrics.distFromBottom
  }

  private reconcileRootShape(): void {
    if (this.suspended || this.forkId !== null) return
    const timeline = this.timeline()
    if (!timeline?.window.hasMoreBefore) return
    if (this.shapeInFlight()) return
    const metrics = this.normalizedMetrics()
    if (!metrics) return
    if (metrics.maxScrollTop > 0) return
    this.core.declareRootTailLimit(
      this.core.getSnapshot().rootTailLimit + INCREMENTAL_ROOT_PAGE_SIZE,
    )
  }

  // ── Activity: anchor restore, bottom-follow, then one comparison ─────────

  private onActivity = (kind: ActivityKind): void => {
    if (this.suspended) return

    // Consume the self-triggered scroll guard. When the controller calls
    // setScrollTop, the surface fires a scroll event synchronously. This
    // is the controller's own write, not user input — ignore it.
    if (kind === "scroll" && this.suppressScroll) {
      this.suppressScroll = false
      return
    }

    if (kind === "scroll") {
      // User scroll: update followingBottom based on where they are now.
      const atBottom = this.isAtBottom()
      this.followingBottom = atBottom
    }

    if (this.anchorPending) {
      if (kind === "scroll") {
        // User scrolled before the resize that would consume the anchor.
        // The user's position stands — clear without writing.
        this.clearAnchor()
        return
      }
      // kind === "resize": consume the anchor with a single write after
      // layout settles, then release.
      this.applyAnchor()
      this.clearAnchor()
      this.reconcileRootShape()
      return
    }

    if (kind === "resize") {
      // Bottom-following: content grew and no anchor is pending. If the
      // user was following the bottom, scroll to the new bottom. We use the
      // flag (not isAtBottom()) because the resize event sees POST-growth
      // geometry — the viewport is no longer at the bottom by definition.
      if (this.followingBottom) {
        this.armAnchor(0)
        this.reconcileRootShape()
        return
      }
      this.reconcileRootShape()
      return
    }

    if (this.forkId !== null) return
    const metrics = this.normalizedMetrics()
    if (!metrics) return

    // The load trigger: the whole pagination logic. The anchor is NOT armed
    // here — it is armed in onDisplayChange when the window change is
    // accepted, which still reads pre-change geometry (store listeners fire
    // before React commits) and reflects wherever the user scrolled to in
    // the meantime.
    if (metrics.scrollTop < this.adapter.loadThreshold) {
      const timeline = this.timeline()
      if (!timeline?.window.hasMoreBefore) return
      if (this.shapeInFlight()) return
      this.core.declareRootTailLimit(
        this.core.getSnapshot().rootTailLimit + INCREMENTAL_ROOT_PAGE_SIZE,
      )
    }
  }

  // ── Display changes ───────────────────────────────────────────────────────
  // Store listeners fire synchronously on accept, BEFORE React re-renders —
  // the new state is readable while the surface still shows old geometry.

  private onDisplayChange = (): void => {
    const timeline = this.timeline()
    if (!timeline) {
      this.prevWindowStart = null
      return
    }

    const start = timeline.window.start
    if (this.prevWindowStart === null) {
      // First content for this session/fork — start at the tail.
      this.prevWindowStart = start
      if (!this.suspended) {
        this.armAnchor(0)
        this.reconcileRootShape()
      }
      return
    }
    if (start === this.prevWindowStart) {
      this.reconcileRootShape()
      return
    }
    this.prevWindowStart = start

    // Window top moved (load accepted, tail slide, evict, resync).
    // When at the bottom, distFromBottom ≈ 0 → armAnchor(0) follows the
    // bottom. When scrolled up, armAnchor(d) restores the position.
    if (this.anchorPending) {
      this.reconcileRootShape()
      return
    }
    if (this.suspended) return // track silently, restore on resume
    const distFromBottom = this.currentDistFromBottom()
    if (distFromBottom !== null) this.armAnchor(distFromBottom)
    this.reconcileRootShape()
  }

  // ── Core changes: session switch, display-mode toggle ────────────────────

  private onCoreChange = (): void => {
    const snapshot = this.core.getSnapshot()

    if (snapshot.selectedSessionId !== this.prevSessionId) {
      this.prevSessionId = snapshot.selectedSessionId
      this.prevDisplayMode = snapshot.displayMode
      this.prevWindowStart = null
      this.followingBottom = true
      if (!this.suspended) {
        this.armAnchor(0)
        this.reconcileRootShape()
      }
      return
    }

    if (snapshot.displayMode !== this.prevDisplayMode) {
      this.prevDisplayMode = snapshot.displayMode
      // Presentation rebuild: keep the reading position, proportionally from
      // the tail. Core listeners fire before React re-renders, so this reads
      // pre-toggle geometry.
      if (this.suspended) return
      if (!this.anchorPending && !this.isAtBottom()) {
        const distFromBottom = this.currentDistFromBottom()
        if (distFromBottom !== null) this.armAnchor(distFromBottom)
      }
      this.reconcileRootShape()
    }
  }
}
