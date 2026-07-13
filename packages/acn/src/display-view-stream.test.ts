import { describe, expect, it } from "vitest"
import { Deferred, Duration, Effect, Option, Queue, Schema, Stream } from "effect"
import {
  StreamEvent as StreamEventSchema,
  type DisplayState,
  type DisplayViewShape,
} from "@magnitudedev/protocol"
import { makeDisplayViewStream, type DisplayViewStreamInput } from "./display-view-stream"

const shape: DisplayViewShape = {
  timelines: {
    root: { kind: "tail", limit: 200, live: true, presentation: "default" },
  },
}
const smallerShape: DisplayViewShape = {
  timelines: {
    root: { kind: "tail", limit: 50, live: true, presentation: "default" },
  },
}

const noSourceEvents = {
  restoreQueuedMessages: Stream.empty,
}

const emptyWindow = {
  start: 0,
  end: 0,
  totalCount: 0,
  hasMoreBefore: false,
  hasMoreAfter: false,
}

const defaultPresentation = {
  mode: "default" as const,
  entries: [],
  statusSlot: { kind: "none" as const },
}

const stateWithTaskParent = (parentId: Option.Option<string>): DisplayState => ({
  session: { sessionId: "s1", title: null, cwd: "/tmp" },
  timelines: {
    root: {
      mode: "idle",
      messages: { byId: {}, order: [] },
      streamingMessageId: null,
      window: emptyWindow,
      presentation: defaultPresentation,
    },
  },
  agents: {},
  actors: {},
  tasks: {
    byId: {
      root: {
        rowId: "row-root",
        kind: "task",
        taskId: "root",
        title: "Root",
        status: "pending",
        parentId,
        depth: 0,
        updatedAt: 1,
        assignee: { kind: "none" },
      },
    },
    order: ["root"],
    summary: { totalCount: 1, completedCount: 0, incompleteCount: 1 },
  },
})

describe("makeDisplayViewStream", () => {
  it("emits patches over encoded DisplayState values", async () => {
    const initial = stateWithTaskParent(Option.none())
    const next = stateWithTaskParent(Option.some("parent"))
    const source: DisplayViewStreamInput["source"] = {
      on: noSourceEvents,
      displayView: {
        stream: (viewId: string) => {
          expect(viewId).toBe("view-1")
          return Stream.fromIterable([
            { shape, state: initial },
            { shape, state: next },
          ])
        },
        snapshot: (viewId: string) => {
          expect(viewId).toBe("view-1")
          return Effect.succeed({ shape, state: next })
        },
        setShape: (_viewId: string, _shape: DisplayViewShape) => Effect.void,
        close: (_viewId: string) => Effect.void,
      },
    }

    const events = await Effect.runPromise(Effect.gen(function* () {
      const handle = yield* makeDisplayViewStream({ source, viewId: "view-1" })
      const chunk = yield* handle.stream.pipe(Stream.take(2), Stream.runCollect)
      return Array.from(chunk)
    }))

    if (events.length !== 2) {
      throw new Error(`Expected 2 stream events, received ${events.length}`)
    }
    const stateEvent = events[0]
    const patchEvent = events[1]

    expect(stateEvent._tag).toBe("state")
    expect(stateEvent).toMatchObject({ shape })
    expect(patchEvent).toEqual({
      _tag: "patch",
      ops: [{ op: "add", path: "/state/tasks/byId/root/parentId", value: "parent" }],
    })
    expect(Schema.encodeSync(StreamEventSchema)(patchEvent)).toEqual(patchEvent)
  })

  it("patches shape changes like any other change — full state only on open", async () => {
    const state = stateWithTaskParent(Option.none())
    const next = stateWithTaskParent(Option.some("parent"))
    const source: DisplayViewStreamInput["source"] = {
      on: noSourceEvents,
      displayView: {
        stream: (viewId: string) => {
          expect(viewId).toBe("view-2")
          return Stream.fromIterable([
            { shape, state },
            { shape: smallerShape, state },
            { shape: smallerShape, state: next },
          ])
        },
        snapshot: (viewId: string) => {
          expect(viewId).toBe("view-2")
          return Effect.succeed({ shape: smallerShape, state: next })
        },
        setShape: (_viewId: string, _shape: DisplayViewShape) => Effect.void,
        close: (_viewId: string) => Effect.void,
      },
    }

    const events = await Effect.runPromise(Effect.gen(function* () {
      const handle = yield* makeDisplayViewStream({ source, viewId: "view-2" })
      const chunk = yield* handle.stream.pipe(Stream.take(3), Stream.runCollect)
      return Array.from(chunk)
    }))

    expect(events.map((event) => event._tag)).toEqual(["state", "patch", "patch"])
    expect(events[0]).toMatchObject({ shape })
    // Shape change arrives as a generic patch on the snapshot's /shape subtree
    expect(events[1]).toEqual({
      _tag: "patch",
      ops: [{ op: "replace", path: "/shape/timelines/root/limit", value: 50 }],
    })
    expect(events[2]).toEqual({
      _tag: "patch",
      ops: [{ op: "add", path: "/state/tasks/byId/root/parentId", value: "parent" }],
    })
  })

  it("patches worker timelines into an existing view", async () => {
    const state = stateWithTaskParent(Option.none())
    const withWorker: DisplayState = {
      ...state,
      timelines: {
        ...state.timelines,
        "worker-1": {
          mode: "idle",
          messages: {
            byId: {
              "worker-msg-1": {
                id: "worker-msg-1",
                type: "assistant_message",
                content: "worker loaded",
                timestamp: 10,
              },
            },
            order: ["worker-msg-1"],
          },
          streamingMessageId: null,
          window: {
            start: 0,
            end: 1,
            totalCount: 1,
            hasMoreBefore: false,
            hasMoreAfter: false,
          },
          presentation: defaultPresentation,
        },
      },
    }
    const workerShape: DisplayViewShape = {
      timelines: {
        ...shape.timelines,
        "worker-1": { kind: "tail", limit: 200, live: true, presentation: "default" },
      },
    }
    const source: DisplayViewStreamInput["source"] = {
      on: noSourceEvents,
      displayView: {
        stream: (viewId: string) => {
          expect(viewId).toBe("view-worker")
          return Stream.fromIterable([
            { shape, state },
            { shape: workerShape, state: withWorker },
          ])
        },
        snapshot: (viewId: string) => {
          expect(viewId).toBe("view-worker")
          return Effect.succeed({ shape: workerShape, state: withWorker })
        },
        setShape: (_viewId: string, _shape: DisplayViewShape) => Effect.void,
        close: (_viewId: string) => Effect.void,
      },
    }

    const events = await Effect.runPromise(Effect.gen(function* () {
      const handle = yield* makeDisplayViewStream({ source, viewId: "view-worker" })
      const chunk = yield* handle.stream.pipe(Stream.take(2), Stream.runCollect)
      return Array.from(chunk)
    }))

    expect(events[0]).toMatchObject({ _tag: "state", shape })
    expect(events[1]._tag).toBe("patch")
    if (events[1]._tag !== "patch") {
      throw new Error("Expected patch event")
    }
    expect(events[1].ops).toEqual(expect.arrayContaining([
      {
        op: "add",
        path: "/shape/timelines/worker-1",
        value: { kind: "tail", limit: 200, live: true, presentation: "default" },
      },
      {
        op: "add",
        path: "/state/timelines/worker-1",
        value: {
          mode: "idle",
          messages: {
            byId: {
              "worker-msg-1": {
                id: "worker-msg-1",
                type: "assistant_message",
                content: "worker loaded",
                timestamp: 10,
              },
            },
            order: ["worker-msg-1"],
          },
          streamingMessageId: null,
          window: {
            start: 0,
            end: 1,
            totalCount: 1,
            hasMoreBefore: false,
            hasMoreAfter: false,
          },
          presentation: defaultPresentation,
        },
      },
    ]))
    expect(events[1].ops).toHaveLength(2)
    expect(Schema.encodeSync(StreamEventSchema)(events[1])).toEqual(events[1])
  })

  it("relays queued-message restore events without a display snapshot", async () => {
    const source: DisplayViewStreamInput["source"] = {
      on: {
        restoreQueuedMessages: Stream.succeed({
          forkId: null,
          messages: [{ id: "queued-1", content: "please keep this", taskMode: false }],
        }),
      },
      displayView: {
        stream: () => Stream.never,
        snapshot: () => Effect.succeed({ shape, state: stateWithTaskParent(Option.none()) }),
        setShape: (_viewId: string, _shape: DisplayViewShape) => Effect.void,
        close: (_viewId: string) => Effect.void,
      },
    }

    const events = await Effect.runPromise(Effect.gen(function* () {
      const handle = yield* makeDisplayViewStream({ source, viewId: "view-restore" })
      const chunk = yield* handle.stream.pipe(Stream.take(1), Stream.runCollect)
      return Array.from(chunk)
    }))

    expect(events).toEqual([{
      _tag: "restore_queued_messages",
      forkId: null,
      messages: [{ id: "queued-1", content: "please keep this", taskMode: false }],
    }])
    expect(Schema.encodeSync(StreamEventSchema)(events[0])).toEqual(events[0])
  })

  it("release terminates the stream wrapper without closing the view", async () => {
    const initial = stateWithTaskParent(Option.none())
    const closed: string[] = []
    const source: DisplayViewStreamInput["source"] = {
      on: noSourceEvents,
      displayView: {
        stream: () =>
          Stream.fromIterable([
            { shape, state: initial },
          ]).pipe(Stream.concat(Stream.never)),
        snapshot: () => Effect.succeed({ shape, state: initial }),
        setShape: (_viewId: string, _shape: DisplayViewShape) => Effect.void,
        close: (viewId: string) =>
          Effect.sync(() => {
            closed.push(viewId)
          }),
      },
    }

    await Effect.runPromise(Effect.gen(function* () {
      const handle = yield* makeDisplayViewStream({ source, viewId: "view-release" })
      const events = yield* Queue.unbounded<unknown>()
      const drained = yield* Deferred.make<void>()

      yield* handle.stream.pipe(
        Stream.tap((event) => Queue.offer(events, event)),
        Stream.runDrain,
        Effect.ensuring(Deferred.succeed(drained, undefined)),
        Effect.fork
      )

      yield* Queue.take(events)
      yield* handle.release
      yield* Deferred.await(drained).pipe(
        Effect.timeoutFail({
          duration: Duration.millis(100),
          onTimeout: () => "display view stream wrapper did not release"
        })
      )
    }))

    expect(closed).toEqual([])
  })

  it("close terminates the stream wrapper and closes the view", async () => {
    const initial = stateWithTaskParent(Option.none())
    const closed: string[] = []
    const source: DisplayViewStreamInput["source"] = {
      on: noSourceEvents,
      displayView: {
        stream: () =>
          Stream.fromIterable([
            { shape, state: initial },
          ]).pipe(Stream.concat(Stream.never)),
        snapshot: () => Effect.succeed({ shape, state: initial }),
        setShape: (_viewId: string, _shape: DisplayViewShape) => Effect.void,
        close: (viewId: string) =>
          Effect.sync(() => {
            closed.push(viewId)
          }),
      },
    }

    await Effect.runPromise(Effect.gen(function* () {
      const handle = yield* makeDisplayViewStream({ source, viewId: "view-close" })
      const events = yield* Queue.unbounded<unknown>()
      const drained = yield* Deferred.make<void>()

      yield* handle.stream.pipe(
        Stream.tap((event) => Queue.offer(events, event)),
        Stream.runDrain,
        Effect.ensuring(Deferred.succeed(drained, undefined)),
        Effect.fork
      )

      yield* Queue.take(events)
      yield* handle.close
      yield* Deferred.await(drained).pipe(
        Effect.timeoutFail({
          duration: Duration.millis(100),
          onTimeout: () => "display view stream wrapper did not close"
        })
      )
    }))

    expect(closed).toEqual(["view-close"])
  })
})
