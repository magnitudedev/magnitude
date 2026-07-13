import { Deferred, Effect, Exit, Option, Queue, Scope, Stream } from "effect"
import {
  compilePatchMap,
  diffDecoded,
} from "@magnitudedev/utils/patch"
import {
  DisplayViewSnapshot as DisplayViewSnapshotSchema,
  SessionOperationFailed,
  type DisplayViewShape,
  type DisplayViewSnapshot,
  type DisplayViewPatchEvent,
  type DisplayViewRestoreQueuedMessagesEvent,
  type DisplayViewStateEvent,
  type SessionError,
  type StreamEvent,
} from "@magnitudedev/protocol"

export interface DisplayViewSource {
  readonly on: {
    readonly restoreQueuedMessages: Stream.Stream<{
      readonly forkId: string | null
      readonly messages: readonly { readonly id: string; readonly content: string; readonly taskMode: boolean }[]
    }>
  }
  readonly displayView: {
    readonly stream: (viewId: string) => Stream.Stream<DisplayViewSnapshot, SessionError>
    readonly snapshot: (viewId: string) => Effect.Effect<DisplayViewSnapshot, SessionError>
    readonly setShape: (viewId: string, shape: DisplayViewShape) => Effect.Effect<void, SessionError>
    readonly close: (viewId: string) => Effect.Effect<void, SessionError>
  }
}

/**
 * The stream mirrors the agent's accepted DisplayView-shaped DisplayState and
 * only normalizes it into decoded-level patch events for transport.
 */
export interface DisplayViewStreamInput {
  readonly source: DisplayViewSource
  readonly viewId: string
}

export interface DisplayViewStreamHandle {
  readonly stream: Stream.Stream<StreamEvent, SessionError>
  readonly takeSnapshot: Effect.Effect<void>
  readonly setShape: (shape: DisplayViewShape) => Effect.Effect<void, SessionError>
  readonly release: Effect.Effect<void>
  readonly close: Effect.Effect<void, SessionError>
}

// Compile the patch map once at module level.
const patchMap = compilePatchMap(DisplayViewSnapshotSchema)

/**
 * Create a display view stream that emits state/patch events.
 *
 * Pure Effect: the accepted display-view stream is the source of
 * truth. Snapshot requests pull the current accepted view explicitly.
 * Snapshot requests are merged in as full-state emissions; ordinary display
 * changes are normalized to decoded-level patch events.
 */
export const makeDisplayViewStream = (input: DisplayViewStreamInput): Effect.Effect<DisplayViewStreamHandle> =>
  Effect.gen(function* () {
    const snapshots = yield* Queue.unbounded<"snapshot">()
    const closeSignal = yield* Deferred.make<void>()

    const liveStates = input.source.displayView.stream(input.viewId).pipe(
      Stream.map((snapshot) => ({ _tag: "live" as const, snapshot })),
    )

    // Explicit snapshot requests are resync points: always a full state.
    const snapshotStates = Stream.fromQueue(snapshots).pipe(
      Stream.mapEffect(() => input.source.displayView.snapshot(input.viewId)),
      Stream.map((snapshot) => ({ _tag: "requested" as const, snapshot })),
    )

    // Decoded-level diff over the full accepted snapshot ({shape, state}).
    // The server diffs decoded values directly — no full Schema.encode needed.
    // Leaf values are encoded via sub-schemas for transport.
    const displayEvents = Stream.merge(liveStates, snapshotStates).pipe(
      Stream.mapAccumEffect(
        Option.none<DisplayViewSnapshot>(),
        (prev, next) =>
          Effect.gen(function* () {
            if (next._tag === "requested" || Option.isNone(prev)) {
              const event: DisplayViewStateEvent = {
                _tag: "state",
                shape: next.snapshot.shape,
                state: next.snapshot.state,
              }
              return [Option.some(next.snapshot), Option.some<StreamEvent>(event)] as const
            }

            const ops = yield* diffDecoded(prev.value, next.snapshot, patchMap).pipe(
              Effect.mapError((error) => new SessionOperationFailed({
                operation: "displayViewDiff",
                reason: error instanceof Error ? error.message : String(error),
              })),
            )

            if (ops.length === 0) {
              return [prev, Option.none<StreamEvent>()] as const
            }

            const event: DisplayViewPatchEvent = { _tag: "patch", ops }
            return [Option.some(next.snapshot), Option.some<StreamEvent>(event)] as const
          }),
      ),
      Stream.filterMap((event) => event),
    )

    const restoreQueuedMessagesEvents = input.source.on.restoreQueuedMessages.pipe(
      Stream.map((value): DisplayViewRestoreQueuedMessagesEvent => ({
        _tag: "restore_queued_messages",
        forkId: value.forkId,
        messages: [...value.messages],
      })),
    )

    const rawStream = Stream.merge(displayEvents, restoreQueuedMessagesEvents).pipe(
      Stream.interruptWhenDeferred(closeSignal),
    )
    const streamScope = yield* Scope.make()
    const stream = yield* rawStream.pipe(
      Stream.share({ capacity: "unbounded", replay: 1 }),
      Scope.extend(streamScope),
    )

    const closeStreamScope = Scope.close(streamScope, Exit.void).pipe(
      Effect.fork,
      Effect.asVoid,
    )
    const release = Deferred.succeed(closeSignal, undefined).pipe(
      Effect.zipRight(closeStreamScope),
      Effect.asVoid,
    )

    return {
      stream,
      takeSnapshot: Queue.offer(snapshots, "snapshot").pipe(Effect.asVoid),
      setShape: (shape) => input.source.displayView.setShape(input.viewId, shape),
      release,
      close: release.pipe(Effect.zipRight(input.source.displayView.close(input.viewId))),
    }
  })
