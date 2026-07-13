import { describe, expect, it, vi } from "vitest"
import type { DisplayTimeline, DisplayViewSnapshot } from "@magnitudedev/sdk"
import { displayShapeFor, EMPTY_DISPLAY_VIEW_SHAPE, INITIAL_ROOT_PAGE_SIZE, INCREMENTAL_ROOT_PAGE_SIZE } from "../sync/index"
import { EMPTY_DISPLAY_STATE } from "../state/empty-display-state"
import type { DisplayReader } from "../sync/display-view-store"
import type { DisplayViewControllerCore, DisplayViewControllerSnapshot } from "./controller"
import {
  TimelineScrollController,
  type ActivityKind,
  type ScrollMetrics,
  type TimelineScrollAdapter,
} from "./timeline-scroll-controller"

const timeline = (hasMoreBefore: boolean): DisplayTimeline => ({
  mode: "idle",
  messages: { byId: {}, order: [] },
  streamingMessageId: null,
  window: {
    start: hasMoreBefore ? 50 : 0,
    end: hasMoreBefore ? 100 : 50,
    totalCount: hasMoreBefore ? 100 : 50,
    hasMoreBefore,
    hasMoreAfter: false,
  },
  presentation: {
    mode: "default",
    entries: [
      {
        kind: "message",
        id: "entry:m1",
        messageId: "m1",
        timestamp: 1,
        role: "assistant",
        streaming: false,
        interrupted: false,
        nextMessageInterrupted: false,
      },
    ],
    statusSlot: { kind: "none" },
  },
})

class FakeCore {
  declarations: number[] = []
  private listeners = new Set<() => void>()

  constructor(public rootTailLimit = INITIAL_ROOT_PAGE_SIZE) {}

  getSnapshot = (): DisplayViewControllerSnapshot => ({
    selectedSessionId: "session-a",
    viewId: "main:session-a",
    expandedForkStack: [],
    rootTailLimit: this.rootTailLimit,
    displayMode: "default",
    phase: "open",
    hasReceivedDisplay: true,
    connectionError: null,
  })

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  declareRootTailLimit = (limit: number): void => {
    this.declarations.push(limit)
    this.rootTailLimit = Math.max(
      INITIAL_ROOT_PAGE_SIZE,
      Math.ceil(limit / INCREMENTAL_ROOT_PAGE_SIZE) * INCREMENTAL_ROOT_PAGE_SIZE,
    )
    for (const listener of this.listeners) listener()
  }
}

class FakeReader implements DisplayReader {
  private listeners = new Set<() => void>()
  private acceptedLimit = INITIAL_ROOT_PAGE_SIZE
  private rootTimeline = timeline(false)

  getSnapshot = (): DisplayViewSnapshot => ({
    shape: this.acceptedLimit === 0 ? EMPTY_DISPLAY_VIEW_SHAPE : displayShapeFor(this.acceptedLimit, []),
    state: {
      ...EMPTY_DISPLAY_STATE,
      timelines: this.acceptedLimit === 0 ? {} : { root: this.rootTimeline },
    },
  })

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  acceptRoot(limit: number, rootTimeline: DisplayTimeline): void {
    this.acceptedLimit = limit
    this.rootTimeline = rootTimeline
    for (const listener of this.listeners) listener()
  }
}

class FakeAdapter implements TimelineScrollAdapter {
  handlers: Array<(kind: ActivityKind) => void> = []
  scrollWrites: number[] = []
  metrics: ScrollMetrics | null = { scrollTop: 0, viewportHeight: 50, scrollHeight: 20 }

  getScrollMetrics = (): ScrollMetrics | null => this.metrics

  setScrollTop = (value: number): void => {
    this.scrollWrites.push(value)
    if (this.metrics) this.metrics = { ...this.metrics, scrollTop: value }
  }

  subscribeActivity = (handler: (kind: ActivityKind) => void): (() => void) => {
    this.handlers.push(handler)
    return () => {
      this.handlers = this.handlers.filter((candidate) => candidate !== handler)
    }
  }

  stickyThreshold = 2
  loadThreshold = 3

  emit(kind: ActivityKind): void {
    for (const handler of this.handlers) handler(kind)
  }
}

const setup = ({
  rootTailLimit = INITIAL_ROOT_PAGE_SIZE,
  acceptedLimit = INITIAL_ROOT_PAGE_SIZE,
  rootTimeline = timeline(false),
  metrics = { scrollTop: 0, viewportHeight: 50, scrollHeight: 20 },
}: {
  readonly rootTailLimit?: number
  readonly acceptedLimit?: number
  readonly rootTimeline?: DisplayTimeline
  readonly metrics?: ScrollMetrics | null
} = {}) => {
  const core = new FakeCore(rootTailLimit)
  const reader = new FakeReader()
  const adapter = new FakeAdapter()
  adapter.metrics = metrics
  reader.acceptRoot(acceptedLimit, rootTimeline)
  const controller = new TimelineScrollController({
    adapter,
    core: core as unknown as DisplayViewControllerCore,
    reader,
    forkId: null,
  })
  return { adapter, controller, core, reader }
}

describe("TimelineScrollController root shape sufficiency", () => {
  it("grows an underfilled accepted root without a scroll event", () => {
    const { controller, core } = setup({ rootTimeline: timeline(true) })

    controller.init()

    expect(core.declarations).toEqual([INITIAL_ROOT_PAGE_SIZE + INCREMENTAL_ROOT_PAGE_SIZE])
  })

  it("converges page by page until the rendered root has a usable scroll range", () => {
    const { adapter, controller, core, reader } = setup({ rootTimeline: timeline(true) })
    controller.init()

    reader.acceptRoot(core.rootTailLimit, timeline(true))
    expect(core.declarations).toEqual([100, 150])

    adapter.metrics = { scrollTop: 0, viewportHeight: 50, scrollHeight: 80 }
    reader.acceptRoot(core.rootTailLimit, timeline(true))

    expect(core.declarations).toEqual([100, 150])
  })

  it("does not grow when the accepted root starts at history beginning", () => {
    const { controller, core } = setup({ rootTimeline: timeline(false) })

    controller.init()

    expect(core.declarations).toEqual([])
  })

  it("does not duplicate requests while a root shape change is in flight", () => {
    const { controller, core } = setup({
      rootTailLimit: INITIAL_ROOT_PAGE_SIZE + INCREMENTAL_ROOT_PAGE_SIZE,
      acceptedLimit: INITIAL_ROOT_PAGE_SIZE,
      rootTimeline: timeline(true),
    })

    controller.init()

    expect(core.declarations).toEqual([])
  })

  it("does not reconcile while suspended but reconciles on resume", () => {
    const { controller, core, reader } = setup({ rootTimeline: timeline(false) })
    controller.init()
    controller.suspend()

    reader.acceptRoot(core.rootTailLimit, timeline(true))
    expect(core.declarations).toEqual([])

    controller.resume()
    expect(core.declarations).toEqual([INITIAL_ROOT_PAGE_SIZE + INCREMENTAL_ROOT_PAGE_SIZE])
  })

  it("uses normalized metrics for anchor and sufficiency decisions", () => {
    const { adapter, controller, core } = setup({
      rootTimeline: timeline(true),
      metrics: { scrollTop: 999, viewportHeight: 50, scrollHeight: 100 },
    })

    controller.init()

    expect(adapter.scrollWrites).toEqual([50])
    expect(core.declarations).toEqual([])
  })

  it("does not automatically shrink the root shape while parked at bottom", () => {
    vi.useFakeTimers()
    try {
      const { adapter, controller, core } = setup({
        rootTailLimit: INITIAL_ROOT_PAGE_SIZE + INCREMENTAL_ROOT_PAGE_SIZE,
        acceptedLimit: INITIAL_ROOT_PAGE_SIZE + INCREMENTAL_ROOT_PAGE_SIZE,
        rootTimeline: timeline(false),
        metrics: { scrollTop: 50, viewportHeight: 50, scrollHeight: 100 },
      })
      controller.init()
      core.declarations = []

      adapter.emit("scroll")
      vi.advanceTimersByTime(5000)

      expect(core.declarations).toEqual([])
      expect(core.rootTailLimit).toBe(INITIAL_ROOT_PAGE_SIZE + INCREMENTAL_ROOT_PAGE_SIZE)
    } finally {
      vi.useRealTimers()
    }
  })
})
