import { Option } from "effect"
import { describe, expect, test } from "vitest"
import {
  NotificationIdSchema,
  NotificationStateSchema,
} from "@magnitudedev/client-common"
import { notificationAreaLabel } from "./notification-area"

describe("notification area presentation", () => {
  test("uses compact low-memory guidance in the two-row presentation", () => {
    const state = NotificationStateSchema.make({
      id: NotificationIdSchema.make("selected-local-model-low-memory"),
      message: "Low memory: close memory-intensive apps (need 2.4 GB) to load model",
      compactMessage: Option.some("Low memory: Free 2.4 GB to load"),
      priority: "warning",
      action: Option.none(),
      createdAt: 0,
    })

    expect(notificationAreaLabel(state)).toBe(
      "! Low memory: close memory-intensive apps (need 2.4 GB) to load model",
    )
    expect(notificationAreaLabel(state, true)).toBe(
      "! Low memory: Free 2.4 GB to load",
    )
  })
})
