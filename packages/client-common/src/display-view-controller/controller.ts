import { Cause, Effect, Fiber, Queue, Ref, Scope, Schedule, Stream } from "effect"
import { RpcClient } from "@effect/rpc"
import {
  forkIdToKey,
  MagnitudeRpcs,
  type DisplayTimeline,
  type DisplayViewShape,
  type StreamDisplayViewFailure,
  type StreamEvent,
  type StreamWireEvent,
} from "@magnitudedev/sdk"
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
  | { readonly _tag: "ready"; readonly forkId: string | null; readonly timeline: DisplayTimeline }
  | { readonly _tag: "empty"; readonly forkId: string | null; readonly timeline: DisplayTimeline }
  | { readonly _tag: "unavailable"; readonly forkId: string | null; readonly reason: string }
  | { readonly _tag: "error"; readonly forkId: string | null; readonly message: string }

export interface DisplayConnectionError {
  readonly message: string
  readonly reconnecting: boolean
  readonly invariantViolation: boolean
}

export interface DisplayViewControllerSnapshot {
  readonly selectedSessionId: string | null
  readonly viewId: string | null
  readonly expandedForkStack: readonly string[]
  readonly rootTailLimit: number
  readonly displayMode: DisplayMode
  readonly phase: DisplayViewConnectionPhase
  readonly hasReceivedDisplay: boolean
  readonly connectionError: DisplayConnectionError | null
}

export interface DisplayViewControllerOptions {
  readonly displaySync: DisplaySyncSink
  readonly onRestoreQueuedInputText?: (text: string | null) => void
}

type Listener = () => void

type Command =
  | { readonly _tag: "set-shape"; readonly sessionId: string; readonly viewId: string; readonly shape: DisplayViewShape; readonly generation: number; readonly requestId: number }
  | { readonly _tag: "resync"; readonly sessionId: string; readonly viewId: string; readonly generation: number }
  | { readonly _tag: "close-view"; readonly sessionId: string; readonly viewId: string }

const viewIdForSession = (sessionId: string): string => `main:${sessionId}`

const makeClient = () => RpcClient.make(MagnitudeRpcs)
type DisplayRpcClient = Effect.Effect.Success<ReturnType<typeof makeClient>>

const sameTimelineShape = (
  left: DisplayViewShape["timelines"][string],
  right: DisplayViewShape["timelines"][string],
): boolean => {
  if (left.kind !== right.kind || left.live !== right.live || left.limit !== right.limit) return false
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
    return leftShape !== undefined && rightShape !== undefined && sameTimelineShape(leftShape, rightShape)
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
  viewId: null,
  expandedForkStack: [],
  rootTailLimit: INITIAL_ROOT_PAGE_SIZE,
  displayMode: "default",
  phase: "no_session",
  hasReceivedDisplay: false,
  connectionError: null,
})

export class DisplayViewControllerCore {
  private readonly client: DisplayRpcClient
  private readonly displaySync: DisplaySyncSink
  private readonly onRestoreQueuedInputText: ((text: string | null) => void) | undefined
  private readonly listeners = new Set<Listener>()
  private snapshot: DisplayViewControllerSnapshot = initialSnapshot()
  private streamFiber: Fiber.RuntimeFiber<void, unknown> | null = null
  private readonly commandQueue: Queue.Queue<Command>
  private commandFiber: Fiber.RuntimeFiber<void, never> | null = null
  private disposed = false
  private streamGeneration = 0
  private shapeRequestId = 0
  private lastRequestedShape: DisplayViewShape = EMPTY_DISPLAY_VIEW_SHAPE

  private constructor(
    client: DisplayRpcClient,
    commandQueue: Queue.Queue<Command>,
    commandFiber: Fiber.RuntimeFiber<void, never>,
    options: DisplayViewControllerOptions,
  ) {
    this.client = client
    this.commandQueue = commandQueue
    this.commandFiber = commandFiber
    this.displaySync = options.displaySync
    this.onRestoreQueuedInputText = options.onRestoreQueuedInputText
    this.resetAcceptedStore()
  }

  /**
   * Factory — bakes in the protocol dependency. Creates one RpcClient from
   * the shared protocol layer, one command queue + fiber, and returns the
   * controller. The scope finalizer interrupts all fibers and shuts down
   * the queue.
   */
  static make = (options: DisplayViewControllerOptions): Effect.Effect<
    DisplayViewControllerCore,
    never,
    RpcClient.Protocol | Scope.Scope
  > =>
    Effect.gen(function* () {
      const client = yield* RpcClient.make(MagnitudeRpcs)
      const queue = yield* Queue.unbounded<Command>()

      const controller = new DisplayViewControllerCore(client, queue, null!, options)

      const commandFiber = yield* Effect.fork(
        controller.runCommandLoop(),
      )
      controller.commandFiber = commandFiber

      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          controller.dispose()
        }),
      )

      return controller
    })

  /**
   * Single fiber drains the command queue serially — same serialization
   * semantics as the old Promise commandChain, but proper Effect. The client
   * is baked in from construction.
   */
  private runCommandLoop = (): Effect.Effect<void, never, never> =>
    Stream.fromQueue(this.commandQueue).pipe(
      Stream.runForEach((cmd) =>
        this.executeCommand(cmd).pipe(
          Effect.catchAll((error) =>
            Effect.logWarning(`Display view controller command failed (${cmd._tag})`).pipe(
              Effect.annotateLogs({
                error: error instanceof Error ? error.message : String(error),
              }),
            ),
          ),
        ),
      ),
    )

  private executeCommand = (cmd: Command): Effect.Effect<void, unknown, never> => {
    switch (cmd._tag) {
      case "set-shape": {
        if (!this.isCurrent(cmd.generation, cmd.sessionId)) return Effect.void
        if (this.snapshot.phase !== "open" || cmd.requestId !== this.shapeRequestId) return Effect.void
        return this.client.SetDisplayViewShape({ sessionId: cmd.sessionId, viewId: cmd.viewId, shape: cmd.shape }).pipe(
          Effect.catchAll((error) =>
            Effect.gen(this, function* () {
              if (!this.isCurrent(cmd.generation, cmd.sessionId) || cmd.requestId !== this.shapeRequestId) return
              yield* Effect.logWarning(
                `Failed to set display view shape; reopening display stream: ${
                  error instanceof Error ? error.message : String(error)
                }`,
              )
              this.reopenStream(cmd.sessionId, "reconnecting")
            }),
          ),
        )
      }
      case "resync": {
        if (!this.isCurrent(cmd.generation, cmd.sessionId)) return Effect.void
        return this.client.ResyncDisplayView({ sessionId: cmd.sessionId, viewId: cmd.viewId })
      }
      case "close-view": {
        return this.client.CloseDisplayView({ sessionId: cmd.sessionId, viewId: cmd.viewId }).pipe(
          Effect.catchAll(() => Effect.void),
        )
      }
    }
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
    this.transitionToSession(sessionId)
  }

  clearSession = (): void => {
    if (this.snapshot.selectedSessionId === null && this.snapshot.phase === "no_session") return
    this.closeActiveView()
    this.setSnapshot({
      ...this.snapshot,
      selectedSessionId: null,
      viewId: null,
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

  retry = (): boolean => {
    const sessionId = this.snapshot.selectedSessionId
    if (!sessionId) return false
    this.reopenStream(sessionId, "reconnecting")
    return true
  }

  resync = (): void => {
    const { selectedSessionId, viewId } = this.snapshot
    const generation = this.streamGeneration
    if (!selectedSessionId || !viewId) return
    Effect.runFork(
      Queue.offer(this.commandQueue, { _tag: "resync", sessionId: selectedSessionId, viewId, generation }),
    )
  }

  stop = (): void => {
    if (this.disposed && this.snapshot.phase === "stopped") return
    this.disposed = true
    this.closeActiveView()
    this.setSnapshot({
      ...this.snapshot,
      phase: "stopped",
      connectionError: null,
    })
  }

  dispose = (): void => {
    this.stop()
    if (this.commandFiber) {
      Effect.runFork(Fiber.interrupt(this.commandFiber))
      this.commandFiber = null
    }
    Effect.runSync(Queue.shutdown(this.commandQueue))
    this.listeners.clear()
  }

  private transitionToSession(sessionId: string): void {
    this.closeActiveView()
    this.disposed = false
    this.setSnapshot({
      ...this.snapshot,
      selectedSessionId: sessionId,
      viewId: viewIdForSession(sessionId),
      expandedForkStack: [],
      rootTailLimit: INITIAL_ROOT_PAGE_SIZE,
      phase: "opening",
      hasReceivedDisplay: false,
      connectionError: null,
    })
    this.onRestoreQueuedInputText?.(null)
    this.resetAcceptedStore()
    this.startStream(sessionId, "opening")
  }

  private reopenStream(sessionId: string, phase: "opening" | "reconnecting"): void {
    this.closeActiveView({ closeRemote: false })
    this.setSnapshot({
      ...this.snapshot,
      selectedSessionId: sessionId,
      viewId: viewIdForSession(sessionId),
      phase,
      connectionError: phase === "reconnecting"
        ? { message: "Reconnecting to daemon...", reconnecting: true, invariantViolation: false }
        : null,
    })
    this.startStream(sessionId, phase)
  }

  private updateIntent(update: {
    readonly expandedForkStack?: readonly string[]
    readonly rootTailLimit?: number
    readonly displayMode?: DisplayMode
  }): void {
    const previousShape = desiredShapeForSnapshot(this.snapshot)
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

    const nextShape = desiredShapeForSnapshot(this.snapshot)
    if (!sameDisplayShape(previousShape, nextShape)) {
      this.syncDesiredShape()
    }
  }

  private syncDesiredShape(): void {
    const { selectedSessionId, viewId, phase } = this.snapshot
    if (!selectedSessionId || !viewId || phase !== "open") return

    const shape = desiredShapeForSnapshot(this.snapshot)
    if (sameDisplayShape(this.lastRequestedShape, shape)) return

    const generation = this.streamGeneration
    const requestId = ++this.shapeRequestId
    this.lastRequestedShape = shape

    Effect.runFork(
      Queue.offer(this.commandQueue, {
        _tag: "set-shape",
        sessionId: selectedSessionId,
        viewId,
        shape,
        generation,
        requestId,
      }),
    )
  }

  private startStream(sessionId: string, phase: "opening" | "reconnecting"): void {
    if (this.disposed) return
    this.interruptStream()

    const generation = ++this.streamGeneration
    const viewId = viewIdForSession(sessionId)
    const isDisplayEvent = (event: StreamWireEvent): event is StreamEvent =>
      event._tag !== "heartbeat"

    const streamEffect = Effect.gen(this, function* () {
      const wasReconnecting = yield* Ref.make(phase === "reconnecting")
      let hasReceivedEvent = false

      const resync = (sid: string, targetViewId: string): void => {
        if (!this.isCurrent(generation, sessionId)) return
        Effect.runFork(
          Queue.offer(this.commandQueue, {
            _tag: "resync",
            sessionId: sid,
            viewId: targetViewId,
            generation,
          }),
        )
      }

      const buildStream = () =>
        Stream.unwrap(
          Effect.gen(this, function* () {
            if (!this.isCurrent(generation, sessionId)) return Stream.empty
            const shape = desiredShapeForSnapshot(this.snapshot)
            this.lastRequestedShape = shape
            return this.client.StreamDisplayView({ sessionId, viewId, shape }).pipe(
              Stream.filter(isDisplayEvent),
              Stream.tap((event) =>
                Effect.gen(this, function* () {
                  if (!this.isCurrent(generation, sessionId)) return

                  hasReceivedEvent = true
                  const wasRecon = yield* Ref.getAndSet(wasReconnecting, false)
                  if (
                    wasRecon ||
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

                  yield* applyStreamEvent(
                    this.displaySync,
                    event,
                    resync,
                    sessionId,
                    viewId,
                    (payload) => {
                      if (!this.isCurrent(generation, sessionId)) return
                      if (payload.forkId !== null || payload.messages.length === 0) return
                      this.restoreQueuedMessages(payload.messages.map((message) => message.content).join("\n"))
                    },
                  )

                  this.syncDesiredShape()
                }),
              ),
            )
          }),
        )

      const reconnectSchedule = Schedule.exponential("100 millis").pipe(
        Schedule.upTo("5 seconds"),
        Schedule.jittered,
      )

      yield* buildStream().pipe(
        Stream.retry(
          reconnectSchedule.pipe(
            Schedule.tapInput(() =>
              Effect.gen(this, function* () {
                if (!this.isCurrent(generation, sessionId)) return
                yield* Ref.set(wasReconnecting, true)
                this.setSnapshot({
                  ...this.snapshot,
                  phase: hasReceivedEvent ? "reconnecting" : "opening",
                  hasReceivedDisplay: hasReceivedEvent || this.snapshot.hasReceivedDisplay,
                  connectionError: { message: "Reconnecting to daemon...", reconnecting: true, invariantViolation: false },
                })
              }),
            ),
          ),
        ),
        Stream.catchAllCause((cause) =>
          Cause.isInterruptedOnly(cause)
            ? Stream.empty
            : Stream.unwrap(
                Effect.gen(this, function* () {
                  if (!this.isCurrent(generation, sessionId)) return Stream.empty
                  yield* Ref.set(wasReconnecting, true)
                  this.setSnapshot({
                    ...this.snapshot,
                    phase: hasReceivedEvent ? "reconnecting" : "opening",
                    hasReceivedDisplay: hasReceivedEvent || this.snapshot.hasReceivedDisplay,
                    connectionError: { message: "Reconnecting to daemon...", reconnecting: true, invariantViolation: false },
                  })
                  return buildStream().pipe(Stream.retry(reconnectSchedule))
                }),
              ),
        ),
        Stream.runDrain,
      )
    }).pipe(
      Effect.catchAllCause((cause) =>
        Cause.isInterruptedOnly(cause)
          ? Effect.void
          : Effect.sync(() => {
              if (!this.isCurrent(generation, sessionId)) return
              const info = classifyStreamError(cause as Cause.Cause<StreamDisplayViewFailure>)
              this.setSnapshot({
                ...this.snapshot,
                phase: "failed",
                connectionError: {
                  message: info.message,
                  reconnecting: false,
                  invariantViolation: info.invariantViolation,
                },
              })
            }),
      ),
      Effect.scoped,
    )

    this.streamFiber = Effect.runFork(streamEffect)
  }

  private restoreQueuedMessages(text: string): void {
    this.onRestoreQueuedInputText?.(text)
  }

  private closeActiveView(options: { readonly closeRemote?: boolean } = {}): void {
    const closeRemote = options.closeRemote ?? true
    const { selectedSessionId, viewId } = this.snapshot
    this.interruptStream()
    this.streamGeneration++
    this.shapeRequestId++
    this.lastRequestedShape = EMPTY_DISPLAY_VIEW_SHAPE
    if (!closeRemote || !selectedSessionId || !viewId) return
    Effect.runFork(
      Queue.offer(this.commandQueue, {
        _tag: "close-view",
        sessionId: selectedSessionId,
        viewId,
      }),
    )
  }

  private interruptStream(): void {
    if (!this.streamFiber) return
    Effect.runFork(Fiber.interrupt(this.streamFiber))
    this.streamFiber = null
  }

  private resetAcceptedStore(): void {
    this.displaySync.resetAccepted({ shape: EMPTY_DISPLAY_VIEW_SHAPE, state: EMPTY_DISPLAY_STATE })
  }

  private isCurrent(generation: number, sessionId: string): boolean {
    return this.streamGeneration === generation && this.snapshot.selectedSessionId === sessionId
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
      return right._tag === "ready" && left.forkId === right.forkId && left.timeline === right.timeline
    case "empty":
      return right._tag === "empty" && left.forkId === right.forkId && left.timeline === right.timeline
    case "unavailable":
      return right._tag === "unavailable" && left.forkId === right.forkId && left.reason === right.reason
    case "error":
      return right._tag === "error" && left.forkId === right.forkId && left.message === right.message
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
