import { describe, expect, it } from "vitest"
import { Effect, Layer } from "effect"
import { RpcClient } from "@effect/rpc"
import type { DisplayTimeline } from "@magnitudedev/sdk"
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

const protocolLayer: Layer.Layer<RpcClient.Protocol, never, never> = Layer.scoped(
  RpcClient.Protocol,
  RpcClient.Protocol.make(() =>
    Effect.succeed({
      send: () => Effect.void,
      supportsAck: false,
      supportsTransferables: false,
    }),
  ),
)

const makeController = (): Promise<DisplayViewControllerCore> =>
  Effect.runPromise(
    DisplayViewControllerCore.make({
      displaySync: createDisplayViewStore(EMPTY_DISPLAY_STATE, EMPTY_DISPLAY_VIEW_SHAPE),
    }).pipe(Effect.scoped, Effect.provide(protocolLayer)),
  )

const emptyTimeline = (): DisplayTimeline => ({
  mode: "idle",
  messages: { byId: {}, order: [] },
  window: { start: 0, end: 0, totalCount: 0, hasMoreBefore: false, hasMoreAfter: false },
  presentation: { mode: "default", entries: [], statusSlot: { kind: "none" } },
  streamingMessageId: null,
})

describe("DisplayViewControllerCore", () => {
  it("derives worker shape from the visible fork stack", async () => {
    const controller = await makeController()

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

  it("replaces worker shape when fork stack changes instead of retaining old workers", async () => {
    const controller = await makeController()

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
