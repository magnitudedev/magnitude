import { Atom, type Registry } from "@effect-atom/atom-react"
import { Effect, Option, Stream } from "effect"
import { Subscription, type Client } from "@magnitudedev/effect-query"
import { forkIdToKey, type DisplayTimeline, type DisplayViewShape, type StreamEvent } from "@magnitudedev/sdk"
import { type AcnQueries } from "../operations"
import type { StreamDisplayViewFailure } from "../state/stream-errors"
import {
  applyStreamEvent,
  ceilToPageMultiple,
  displayShapeFor,
  EMPTY_DISPLAY_VIEW_SHAPE,
  INITIAL_ROOT_PAGE_SIZE,
  type DisplaySyncSink,
} from "../sync/index"
import { EMPTY_DISPLAY_STATE } from "../state/empty-display-state"
import { classifyStreamError } from "../state/stream-errors"

export type DisplayMode = "default" | "transcript"

export type DisplayViewConnectionPhase =
  | "no_session"
  | "opening"
  | "open"
  | "reconnecting"
  | "failed"
  | "stopped"

export type TimelineStatus =
  | { readonly _tag: "none" }
  | { readonly _tag: "pending"; readonly forkId: string | null }
  | {
      readonly _tag: "ready"
      readonly forkId: string | null
      readonly timeline: DisplayTimeline
    }
  | {
      readonly _tag: "empty"
      readonly forkId: string | null
      readonly timeline: DisplayTimeline
    }
  | {
      readonly _tag: "unavailable"
      readonly forkId: string | null
      readonly reason: string
    }
  | {
      readonly _tag: "error"
      readonly forkId: string | null
      readonly message: string
    }

export interface DisplayConnectionError {
  readonly message: string
  readonly reconnecting: boolean
  readonly invariantViolation: boolean
}

export interface DisplayViewControllerSnapshot {
  readonly selectedSessionId: string | null
  readonly expandedForkStack: readonly string[]
  readonly rootTailLimit: number
  readonly displayMode: DisplayMode
  readonly phase: DisplayViewConnectionPhase
  readonly hasReceivedDisplay: boolean
  readonly connectionError: DisplayConnectionError | null
}

/** The part of the connection client the controller materializes the display subscription with. */
export type DisplayViewClient = Pick<Client.GroupClient<typeof AcnQueries, any, any>, "Display" | "runtime">

export interface DisplayViewControllerOptions {
  readonly client: DisplayViewClient
  readonly registry: Registry.Registry
  readonly displaySync: DisplaySyncSink
  readonly onRestoreQueuedInputText?: (text: string | null) => void
}

type Listener = () => void

type DisplaySubscription = ReturnType<DisplayViewClient["Display"]["StreamDisplayView"]> extends infer S
  ? S extends Subscription.SubscriptionAtom<infer I, infer E, infer Err, infer R>
    ? Subscription.SubscriptionAtom<I, E, Err, R>
    : never
  : never

interface ActiveView {
  readonly sessionId: string
  readonly shape: DisplayViewShape
  readonly subscription: DisplaySubscription
  readonly release: () => void
}

const sameTimelineShape = (
  left: DisplayViewShape["timelines"][string],
  right: DisplayViewShape["timelines"][string],
): boolean => {
  if (left.kind !== right.kind || left.live !== right.live || left.limit !== right.limit)
    return false
  if ((left.presentation ?? "default") !== (right.presentation ?? "default")) return false
  if (left.kind === "tail") return true
  return right.kind === "range" && left.start === right.start
}

export const sameDisplayShape = (left: DisplayViewShape, right: DisplayViewShape): boolean => {
  const leftKeys = Object.keys(left.timelines)
  const rightKeys = Object.keys(right.timelines)
  if (leftKeys.length !== rightKeys.length) return false
  return leftKeys.every((key) => {
    const leftShape = left.timelines[key]
    const rightShape = right.timelines[key]
    return (
      leftShape !== undefined &&
      rightShape !== undefined &&
      sameTimelineShape(leftShape, rightShape)
    )
  })
}

const sameStringArray = (left: readonly string[], right: readonly string[]): boolean =>
  left.length === right.length && left.every((value, index) => value === right[index])

export const desiredShapeForSnapshot = (
  snapshot: DisplayViewControllerSnapshot,
): DisplayViewShape =>
  displayShapeFor(snapshot.rootTailLimit, snapshot.expandedForkStack, snapshot.displayMode)

const initialSnapshot = (): DisplayViewControllerSnapshot => ({
  selectedSessionId: null,
  expandedForkStack: [],
  rootTailLimit: INITIAL_ROOT_PAGE_SIZE,
  displayMode: "default",
  phase: "no_session",
  hasReceivedDisplay: false,
  connectionError: null,
})

/**
 * Owns the selected session, the requested display shape, and the one
 * display subscription those imply. The subscription is a contract
 * subscription (`StreamDisplayView(sessionId, shape)`): a shape change is a
 * different subscription, resync and retry reopen it, and its status is the
 * connection phase. Accepted display state lives in the display store.
 */
export class DisplayViewControllerCore {
  private readonly client: DisplayViewClient
  private readonly registry: Registry.Registry
  private readonly displaySync: DisplaySyncSink
  private readonly onRestoreQueuedInputText: ((text: string | null) => void) | undefined
  private readonly listeners = new Set<Listener>()
  private snapshot: DisplayViewControllerSnapshot = initialSnapshot()
  private active: ActiveView | null = null
  private disposed = false

  constructor(options: DisplayViewControllerOptions) {
    this.client = options.client
    this.registry = options.registry
    this.displaySync = options.displaySync
    this.onRestoreQueuedInputText = options.onRestoreQueuedInputText
    this.resetAcceptedStore()
  }

  getSnapshot = (): DisplayViewControllerSnapshot => this.snapshot

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  selectSession = (sessionId: string): void => {
    if (this.snapshot.selectedSessionId === sessionId && this.snapshot.phase !== "stopped") return
    this.closeView()
    this.setSnapshot({
      ...this.snapshot,
      selectedSessionId: sessionId,
      expandedForkStack: [],
      rootTailLimit: INITIAL_ROOT_PAGE_SIZE,
      phase: "opening",
      hasReceivedDisplay: false,
      connectionError: null,
    })
    this.onRestoreQueuedInputText?.(null)
    this.resetAcceptedStore()
    this.openView()
  }

  clearSession = (): void => {
    if (this.snapshot.selectedSessionId === null && this.snapshot.phase === "no_session") return
    this.closeView()
    this.setSnapshot({
      ...this.snapshot,
      selectedSessionId: null,
      expandedForkStack: [],
      rootTailLimit: INITIAL_ROOT_PAGE_SIZE,
      phase: "no_session",
      hasReceivedDisplay: false,
      connectionError: null,
    })
    this.onRestoreQueuedInputText?.(null)
    this.resetAcceptedStore()
  }

  pushFork = (forkId: string): void => {
    const current = this.snapshot.expandedForkStack
    if (current[current.length - 1] === forkId) return
    this.updateIntent({ expandedForkStack: [...current, forkId] })
  }

  popFork = (): void => {
    const current = this.snapshot.expandedForkStack
    if (current.length === 0) return
    this.updateIntent({ expandedForkStack: current.slice(0, -1) })
  }

  setForkStack = (forkIds: readonly string[]): void => {
    if (sameStringArray(this.snapshot.expandedForkStack, forkIds)) return
    this.updateIntent({ expandedForkStack: [...forkIds] })
  }

  /**
   * Declare the root tail limit the client needs. Grow and evict are the
   * same operation — the caller computes the need from the viewport anchor;
   * clamping and page quantization live here so every caller is safe.
   */
  declareRootTailLimit = (limit: number): void => {
    const next = Math.max(INITIAL_ROOT_PAGE_SIZE, ceilToPageMultiple(limit))
    if (next === this.snapshot.rootTailLimit) return
    this.updateIntent({ rootTailLimit: next })
  }

  setPresentationMode = (displayMode: DisplayMode): void => {
    if (this.snapshot.displayMode === displayMode) return
    this.updateIntent({ displayMode })
  }

  togglePresentationMode = (): void => {
    this.setPresentationMode(this.snapshot.displayMode === "default" ? "transcript" : "default")
  }

  /** Reopens the display subscription; a reopened subscription rereads a complete snapshot. */
  retry = (): boolean => {
    const sessionId = this.snapshot.selectedSessionId
    if (!sessionId) return false
    if (this.snapshot.phase === "stopped" || this.active === null) {
      this.setSnapshot({ ...this.snapshot, phase: "opening", connectionError: null })
      this.openView()
      return true
    }
    this.setSnapshot({
      ...this.snapshot,
      phase: "reconnecting",
      connectionError: {
        message: "Reconnecting to Magnitude service...",
        reconnecting: true,
        invariantViolation: false,
      },
    })
    this.registry.set(this.active.subscription, Atom.Reset)
    return true
  }

  /** Rereads a complete snapshot by reopening the display subscription. */
  resync = (): void => {
    if (this.active === null) return
    this.registry.set(this.active.subscription, Atom.Reset)
  }

  stop = (): void => {
    if (this.snapshot.phase === "stopped") return
    this.closeView()
    this.setSnapshot({
      ...this.snapshot,
      phase: "stopped",
      connectionError: null,
    })
  }

  dispose = (): void => {
    if (this.disposed) return
    this.stop()
    this.disposed = true
    this.listeners.clear()
  }

  private updateIntent(update: {
    readonly expandedForkStack?: readonly string[]
    readonly rootTailLimit?: number
    readonly displayMode?: DisplayMode
  }): void {
    const expandedForkStack = update.expandedForkStack ?? this.snapshot.expandedForkStack
    const rootTailLimit = update.rootTailLimit ?? this.snapshot.rootTailLimit
    const displayMode = update.displayMode ?? this.snapshot.displayMode

    if (
      sameStringArray(this.snapshot.expandedForkStack, expandedForkStack) &&
      this.snapshot.rootTailLimit === rootTailLimit &&
      this.snapshot.displayMode === displayMode
    ) {
      return
    }

    this.setSnapshot({
      ...this.snapshot,
      expandedForkStack,
      rootTailLimit,
      displayMode,
    })
    this.openView()
  }

  /** Materializes the subscription for the current session and desired shape, if it changed. */
  private openView(): void {
    if (this.disposed || this.snapshot.phase === "stopped") return
    const sessionId = this.snapshot.selectedSessionId
    if (!sessionId) return
    const shape = desiredShapeForSnapshot(this.snapshot)
    if (
      this.active !== null &&
      this.active.sessionId === sessionId &&
      sameDisplayShape(this.active.shape, shape)
    ) {
      return
    }
    this.closeView()

    const subscription = this.client.Display.StreamDisplayView({ sessionId, shape })
    const events = this.client.runtime.atom(
      Subscription.events(subscription).pipe(
        Stream.runForEach((event) => this.acceptEvent(sessionId, shape, event)),
      ),
    )
    const unmount = this.registry.mount(events)
    const unsubscribe = this.registry.subscribe(
      subscription,
      (state) => this.reflectStatus(sessionId, shape, state),
      { immediate: true },
    )
    this.active = {
      sessionId,
      shape,
      subscription,
      release: () => {
        unsubscribe()
        unmount()
      },
    }
  }

  private closeView(): void {
    if (this.active === null) return
    const active = this.active
    this.active = null
    active.release()
  }

  private isCurrent(sessionId: string, shape: DisplayViewShape): boolean {
    return (
      this.active !== null &&
      this.active.sessionId === sessionId &&
      sameDisplayShape(this.active.shape, shape) &&
      this.snapshot.selectedSessionId === sessionId
    )
  }

  private acceptEvent(
    sessionId: string,
    shape: DisplayViewShape,
    event: StreamEvent,
  ): Effect.Effect<void> {
    return Effect.suspend(() => {
      if (!this.isCurrent(sessionId, shape)) return Effect.void
      return applyStreamEvent(
        this.displaySync,
        event,
        () => {
          if (this.isCurrent(sessionId, shape)) this.resync()
        },
        sessionId,
        (payload) => {
          if (!this.isCurrent(sessionId, shape)) return
          if (payload.forkId !== null || payload.messages.length === 0) return
          this.onRestoreQueuedInputText?.(
            payload.messages.map((message) => message.content).join("\n"),
          )
        },
      ).pipe(
        Effect.tap(() =>
          Effect.sync(() => {
            if (!this.isCurrent(sessionId, shape)) return
            if (event._tag === "restore_queued_messages") return
            if (
              this.snapshot.phase !== "open" ||
              !this.snapshot.hasReceivedDisplay ||
              this.snapshot.connectionError !== null
            ) {
              this.setSnapshot({
                ...this.snapshot,
                phase: "open",
                hasReceivedDisplay: true,
                connectionError: null,
              })
            }
          }),
        ),
      )
    })
  }

  /** The subscription's status is the connection phase. */
  private reflectStatus(
    sessionId: string,
    shape: DisplayViewShape,
    state: Subscription.State<StreamEvent, StreamDisplayViewFailure>,
  ): void {
    if (!this.isCurrent(sessionId, shape) || this.snapshot.phase === "stopped") return
    switch (state.status) {
      case "failed": {
        const info = Option.match(state.failure, {
          onNone: () => ({
            message: "Display stream failed",
            invariantViolation: true,
            isAcnAvailabilityError: false,
          }),
          onSome: classifyStreamError,
        })
        this.setSnapshot({
          ...this.snapshot,
          phase: "failed",
          connectionError: {
            message: info.message,
            reconnecting: false,
            invariantViolation: info.invariantViolation,
          },
        })
        return
      }
      case "reconnecting":
        if (this.snapshot.phase === "reconnecting") return
        this.setSnapshot({
          ...this.snapshot,
          phase: "reconnecting",
          connectionError: {
            message: "Reconnecting to Magnitude service...",
            reconnecting: true,
            invariantViolation: false,
          },
        })
        return
      case "active":
        // `open` is entered when the first event is accepted (see acceptEvent).
        return
      case "idle":
      case "connecting":
        if (this.snapshot.hasReceivedDisplay || this.snapshot.phase === "reconnecting") return
        if (this.snapshot.phase !== "opening") {
          this.setSnapshot({ ...this.snapshot, phase: "opening", connectionError: null })
        }
        return
      case "completed":
        return
    }
  }

  private resetAcceptedStore(): void {
    this.displaySync.resetAccepted({
      shape: EMPTY_DISPLAY_VIEW_SHAPE,
      state: EMPTY_DISPLAY_STATE,
    })
  }

  private setSnapshot(next: DisplayViewControllerSnapshot): void {
    if (this.snapshot === next) return
    this.snapshot = next
    for (const listener of this.listeners) listener()
  }
}

export const timelineStatusEqual = (left: TimelineStatus, right: TimelineStatus): boolean => {
  if (left._tag !== right._tag) return false
  switch (left._tag) {
    case "none":
      return true
    case "pending":
      return right._tag === "pending" && left.forkId === right.forkId
    case "ready":
      return (
        right._tag === "ready" && left.forkId === right.forkId && left.timeline === right.timeline
      )
    case "empty":
      return (
        right._tag === "empty" && left.forkId === right.forkId && left.timeline === right.timeline
      )
    case "unavailable":
      return (
        right._tag === "unavailable" && left.forkId === right.forkId && left.reason === right.reason
      )
    case "error":
      return (
        right._tag === "error" && left.forkId === right.forkId && left.message === right.message
      )
  }
}

export const timelineStatusFor = (
  selectedSessionId: string | null,
  desiredShape: DisplayViewShape,
  acceptedShape: DisplayViewShape,
  timeline: DisplayTimeline | undefined,
  forkId: string | null,
): TimelineStatus => {
  if (!selectedSessionId) return { _tag: "none" }
  const forkKey = forkIdToKey(forkId)
  if (desiredShape.timelines[forkKey] === undefined) return { _tag: "none" }
  if (acceptedShape.timelines[forkKey] === undefined) return { _tag: "pending", forkId }
  if (!timeline) return { _tag: "pending", forkId }
  return timeline.presentation.entries.length === 0
    ? { _tag: "empty", forkId, timeline }
    : { _tag: "ready", forkId, timeline }
}
