import { describe, expect, it } from "vitest"
import { Layer } from "effect"
import type { DaemonSpawnerTag, DisplayTimeline } from "@magnitudedev/sdk"
import {
  createDisplayViewStore,
  displayShapeFor,
  EMPTY_DISPLAY_VIEW_SHAPE,
  INITIAL_ROOT_PAGE_SIZE,
  WORKER_TIMELINE_LIMIT,
} from "../sync/index"
import { EMPTY_DISPLAY_STATE } from "../state/empty-display-state"
import {
  desiredShapeForSnapshot,
  DisplayViewControllerCore,
  timelineStatusFor,
} from "./controller"

const daemonSpawnerLayer = Layer.empty as unknown as Layer.Layer<DaemonSpawnerTag, never, never>

const makeController = (): DisplayViewControllerCore =>
  new DisplayViewControllerCore({
    daemonSpawnerLayer,
    displaySync: createDisplayViewStore(
      EMPTY_DISPLAY_STATE,
      EMPTY_DISPLAY_VIEW_SHAPE,
    ),
  })

const emptyTimeline = (): DisplayTimeline => ({
  mode: "idle",
  messages: { byId: {}, order: [] },
  window: { start: 0, end: 0, totalCount: 0, hasMoreBefore: false, hasMoreAfter: false },
  presentation: { mode: "default", entries: [], statusSlot: { kind: "none" } },
  streamingMessageId: null,
})

describe("DisplayViewControllerCore", () => {
  it("derives worker shape from the visible fork stack", () => {
    const controller = makeController()

    controller.pushFork("worker-a")
    const afterPush = controller.getSnapshot()

    expect(afterPush.expandedForkStack).toEqual(["worker-a"])
    expect(desiredShapeForSnapshot(afterPush).timelines["worker-a"]).toEqual({
      kind: "tail",
      limit: WORKER_TIMELINE_LIMIT,
      live: true,
      presentation: "default",
    })

    controller.popFork()
    const afterPop = controller.getSnapshot()

    expect(afterPop.expandedForkStack).toEqual([])
    expect(Object.keys(desiredShapeForSnapshot(afterPop).timelines)).toEqual(["root"])
  })

  it("replaces worker shape when fork stack changes instead of retaining old workers", () => {
    const controller = makeController()

    controller.setForkStack(["worker-a", "worker-b"])
    expect(Object.keys(desiredShapeForSnapshot(controller.getSnapshot()).timelines).sort()).toEqual([
      "root",
      "worker-a",
      "worker-b",
    ])

    controller.setForkStack(["worker-b"])
    expect(Object.keys(desiredShapeForSnapshot(controller.getSnapshot()).timelines).sort()).toEqual([
      "root",
      "worker-b",
    ])
  })

  it("derives timeline status from desired shape and accepted shape", () => {
    const desired = displayShapeFor(INITIAL_ROOT_PAGE_SIZE, ["worker-a"])
    const acceptedRootOnly = displayShapeFor(INITIAL_ROOT_PAGE_SIZE, [])
    const acceptedWorker = displayShapeFor(INITIAL_ROOT_PAGE_SIZE, ["worker-a"])

    expect(timelineStatusFor("session-a", desired, acceptedRootOnly, undefined, "worker-a")).toEqual({
      _tag: "pending",
      forkId: "worker-a",
    })

    expect(
      timelineStatusFor("session-a", acceptedRootOnly, acceptedWorker, emptyTimeline(), "worker-a"),
    ).toEqual({ _tag: "none" })

    expect(
      timelineStatusFor("session-a", desired, acceptedWorker, emptyTimeline(), "worker-a"),
    ).toMatchObject({ _tag: "empty", forkId: "worker-a" })
  })
})
