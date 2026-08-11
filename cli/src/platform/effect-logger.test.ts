import { describe, expect, test } from "vitest"
import { Effect } from "effect"
import type { PushNotification } from "@magnitudedev/client-common"
import { makeCliEffectLoggingLayer } from "./effect-logger"

const runLog = async (
  debug: boolean,
  effect: Effect.Effect<void>,
): Promise<readonly PushNotification[]> => {
  const notifications: PushNotification[] = []
  await Effect.runPromise(effect.pipe(
    Effect.provide(makeCliEffectLoggingLayer({
      debug,
      publishNotification: (notification) => notifications.push(notification),
    })),
  ))
  return notifications
}

describe("CLI Effect logger", () => {
  test("surfaces errors as toasts", async () => {
    const notifications = await runLog(false, Effect.logError("RPC failed"))

    expect(notifications).toContainEqual(expect.objectContaining({
      message: "RPC failed",
      priority: "error",
    }))
  })

  test("does not surface warnings normally", async () => {
    const notifications = await runLog(false, Effect.logWarning("retrying"))

    expect(notifications).toEqual([])
  })

  test("surfaces warnings in debug mode", async () => {
    const notifications = await runLog(true, Effect.logWarning("retrying"))

    expect(notifications).toContainEqual(expect.objectContaining({
      message: "retrying",
      priority: "warning",
    }))
  })
})
