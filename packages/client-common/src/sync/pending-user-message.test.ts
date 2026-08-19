import { afterEach, describe, expect, it, vi } from "vitest"
import type { DisplayState, DisplayViewShape } from "@magnitudedev/sdk"
import { EMPTY_DISPLAY_STATE } from "../state/empty-display-state"
import {
  appendMessageToTimeline,
  createDisplayViewStore,
  emptyTimeline,
} from "./display-view-store"
import { EMPTY_DISPLAY_VIEW_SHAPE, timelineTail, INITIAL_ROOT_PAGE_SIZE } from "./display-view-shape"
import { presentPendingUserMessage } from "./pending-user-message"

afterEach(() => vi.restoreAllMocks())

const shapeWithRoot = (): DisplayViewShape => ({
  timelines: { root: timelineTail(INITIAL_ROOT_PAGE_SIZE) },
})

const stateWithRoot = (mode: "idle" | "streaming" = "idle"): DisplayState => ({
  ...EMPTY_DISPLAY_STATE,
  session: { sessionId: "session-1", title: null, cwd: "/project" },
  timelines: { root: { ...emptyTimeline(), mode } },
})

describe("presentPendingUserMessage", () => {
  it("creates the minimum disposable display for a new session", () => {
    vi.spyOn(Date, "now").mockReturnValue(123)
    const store = createDisplayViewStore(EMPTY_DISPLAY_STATE, EMPTY_DISPLAY_VIEW_SHAPE)

    presentPendingUserMessage(store, {
      messageId: "m1",
      content: "hey",
      taskMode: true,
      activeSessionId: null,
      draftSessionId: "draft:owner-1",
      cwd: "/project",
    })

    const snapshot = store.getSnapshot()
    expect(snapshot.state.session).toEqual({
      sessionId: "draft:owner-1",
      title: null,
      cwd: "/project",
    })
    expect(snapshot.shape.timelines.root).toEqual(timelineTail(INITIAL_ROOT_PAGE_SIZE))
    expect(snapshot.state.timelines.root?.messages.order).toEqual(["m1"])
    expect(snapshot.state.timelines.root?.messages.byId.m1).toEqual({
      id: "m1",
      type: "user_message",
      content: "hey",
      timestamp: 123,
      taskMode: true,
      attachments: [],
    })
  })

  it("uses the accepted timeline mode for temporary queue presentation", () => {
    const store = createDisplayViewStore(stateWithRoot("streaming"), shapeWithRoot())

    presentPendingUserMessage(store, {
      messageId: "queued-1",
      content: "next",
      taskMode: false,
      activeSessionId: "session-1",
      draftSessionId: "unused",
      cwd: "/project",
    })

    expect(store.getSnapshot().state.timelines.root?.messages.byId["queued-1"]).toMatchObject({
      type: "queued_user_message",
      content: "next",
    })
  })

  it("presents attachment snapshots immediately", () => {
    const store = createDisplayViewStore(stateWithRoot(), shapeWithRoot())

    presentPendingUserMessage(store, {
      messageId: "with-files",
      content: "review these",
      taskMode: false,
      activeSessionId: "session-1",
      draftSessionId: "unused",
      cwd: "/project",
      attachments: [
        { type: "mention_file", path: "$M/attachments/notes.md" },
        {
          type: "image",
          path: "$M/attachments/diagram.png",
          filename: "diagram.png",
          mediaType: "image/png",
          width: 320,
          height: 180,
        },
      ],
    })

    expect(store.getSnapshot().state.timelines.root?.messages.byId["with-files"]).toMatchObject({
      attachments: [
        { type: "mention_file", path: "$M/attachments/notes.md" },
        { type: "image", filename: "diagram.png", width: 320, height: 180 },
      ],
    })
  })

  it("survives accepted-state resets and reconciles only to the exact message id", () => {
    const store = createDisplayViewStore(EMPTY_DISPLAY_STATE, EMPTY_DISPLAY_VIEW_SHAPE)
    presentPendingUserMessage(store, {
      messageId: "m1",
      content: "visible text",
      taskMode: false,
      activeSessionId: null,
      draftSessionId: "draft:owner-1",
      cwd: "/project",
    })

    store.resetAccepted({ shape: EMPTY_DISPLAY_VIEW_SHAPE, state: EMPTY_DISPLAY_STATE })
    expect(store.getSnapshot().state.timelines.root?.messages.order).toEqual(["m1"])

    const unrelated = appendMessageToTimeline(emptyTimeline(), {
      id: "other",
      type: "user_message",
      content: "other",
      timestamp: 1,
      taskMode: false,
      attachments: [],
    })
    store.accept({
      shape: shapeWithRoot(),
      state: { ...stateWithRoot(), timelines: { root: unrelated } },
    })
    expect(store.getSnapshot().state.timelines.root?.messages.order).toEqual(["other", "m1"])

    const acknowledged = appendMessageToTimeline(unrelated, {
      id: "m1",
      type: "user_message",
      content: "authoritative text",
      timestamp: 2,
      taskMode: false,
      attachments: [],
    })
    store.accept({
      shape: shapeWithRoot(),
      state: { ...stateWithRoot(), timelines: { root: acknowledged } },
    })

    expect(store.getSnapshot().state.timelines.root?.messages.order).toEqual(["other", "m1"])
    expect(store.getSnapshot().state.timelines.root?.messages.byId.m1).toMatchObject({
      content: "authoritative text",
      timestamp: 2,
    })
  })

  it("removes only the rejected pending message", () => {
    const store = createDisplayViewStore(stateWithRoot(), shapeWithRoot())
    const first = presentPendingUserMessage(store, {
      messageId: "m1",
      content: "first",
      taskMode: false,
      activeSessionId: "session-1",
      draftSessionId: "unused",
      cwd: "/project",
    })
    presentPendingUserMessage(store, {
      messageId: "m2",
      content: "second",
      taskMode: false,
      activeSessionId: "session-1",
      draftSessionId: "unused",
      cwd: "/project",
    })

    first.remove()

    expect(store.getSnapshot().state.timelines.root?.messages.order).toEqual(["m2"])
  })

  it("keeps a second activation message when the first full snapshot acknowledges only the first", () => {
    const store = createDisplayViewStore(EMPTY_DISPLAY_STATE, EMPTY_DISPLAY_VIEW_SHAPE)
    for (const id of ["m1", "m2"] as const) {
      presentPendingUserMessage(store, {
        messageId: id,
        content: id,
        taskMode: false,
        activeSessionId: null,
        draftSessionId: "draft:owner-1",
        cwd: "/project",
      })
    }

    const firstAccepted = appendMessageToTimeline(emptyTimeline(), {
      id: "m1",
      type: "user_message",
      content: "authoritative m1",
      timestamp: 10,
      taskMode: false,
      attachments: [],
    })
    store.accept({
      shape: shapeWithRoot(),
      state: { ...stateWithRoot(), timelines: { root: firstAccepted } },
    })

    expect(store.getSnapshot().state.timelines.root?.messages.order).toEqual(["m1", "m2"])
    expect(store.getSnapshot().state.timelines.root?.messages.byId.m1).toMatchObject({
      content: "authoritative m1",
    })
  })
})
