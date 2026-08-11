import { describe, expect, test } from "vitest"
import { Option } from "effect"
import { Registry } from "@effect-atom/atom-react"
import {
  NotificationIdSchema,
  NotificationStateSchema,
  notificationAreaStateAtom,
  pushNotificationAtom,
  resolveActiveNotificationState,
} from "./notification-area-state"

const notification = (
  id: string,
  priority: "activity" | "notice" | "warning" | "error",
  createdAt: number,
) => NotificationStateSchema.make({
  id: NotificationIdSchema.make(id),
  message: id,
  compactMessage: Option.none(),
  priority,
  action: Option.none(),
  createdAt,
})

describe("notification area state", () => {
  test("shows the highest-priority notification and the newest equal-priority notification", () => {
    const warning = notification("warning", "warning", 1)
    const newestWarning = notification("newest-warning", "warning", 2)

    expect(resolveActiveNotificationState(
      { notificationStates: [warning, newestWarning] },
      [notification("activity", "activity", 0)],
    )).toBe(newestWarning)
  })

  test("retains concurrent ephemeral occurrences and dismisses each exact identity", async () => {
    const registry = Registry.make()
    const unmount = registry.mount(notificationAreaStateAtom)
    registry.get(notificationAreaStateAtom)

    registry.set(pushNotificationAtom, {
      message: "first",
      priority: "notice",
      action: Option.none(),
      dismissAfterMilliseconds: 20,
    })
    registry.set(pushNotificationAtom, {
      message: "second",
      priority: "warning",
      action: Option.none(),
      dismissAfterMilliseconds: 100,
    })

    await new Promise((resolve) => setTimeout(resolve, 5))
    expect(registry.get(notificationAreaStateAtom).notificationStates.map(
      ({ message }) => message,
    )).toEqual(["first", "second"])

    await new Promise((resolve) => setTimeout(resolve, 60))
    expect(registry.get(notificationAreaStateAtom).notificationStates.map(
      ({ message }) => message,
    )).toEqual(["second"])

    unmount()
    registry.dispose()
  })
})
