import { describe, expect, test } from "vitest"
import { Option, Schema } from "effect"
import { Registry } from "@effect-atom/atom-react"
import {
  AcnRecovering,
  AcnLifecycleStateSchema,
  ModelSlotConfiguredLocal,
  ModelSlotUnassigned,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/sdk"
import {
  deriveAcnRecoveryNotificationState,
  deriveSelectedModelResidencyNotificationState,
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
  test("projects active service recovery without blocking the application", () => {
    const state = new AcnRecovering({
      occurrence: 1,
      lifecycle: Schema.decodeUnknownSync(AcnLifecycleStateSchema)({
        _tag: "Starting",
        phase: "LaunchingLocalInference",
      }),
    })
    expect(deriveAcnRecoveryNotificationState(state)).toMatchObject({
      id: "acn-recovery",
      message: "Starting local inference…",
      priority: "activity",
    })
  })
  test("identifies model discovery during service recovery", () => {
    const state = new AcnRecovering({
      occurrence: 1,
      lifecycle: Schema.decodeUnknownSync(AcnLifecycleStateSchema)({
        _tag: "Starting",
        phase: "DiscoveringLocalModels",
      }),
    })
    expect(deriveAcnRecoveryNotificationState(state)).toMatchObject({
      id: "acn-recovery",
      message: "Discovering local models…",
      priority: "activity",
    })
  })
  test("projects the authoritative selected-model load request into the footer", () => {
    const providerId = ProviderIdSchema.make("local")
    const providerModelId = ProviderModelIdSchema.make("model")
    const base = {
      slotId: PRIMARY_SLOT_ID,
      selection: {
        providerId,
        providerModelId,
        reasoningEffort: ReasoningEffortSchema.make("none"),
      },
      descriptor: {
        providerId,
        providerModelId,
        displayName: "Model",
        variantLabel: Option.none(),
      },
      availability: { _tag: "Available" as const },
      actions: ["Stop" as const],
    }
    const slots = (residency: ModelSlotConfiguredLocal["residency"]) => ({
      slots: {
        primary: new ModelSlotConfiguredLocal({ ...base, residency }),
        secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
      },
      recentModels: { primary: [], secondary: [] },
      favoriteModels: [],
    })

    expect(deriveSelectedModelResidencyNotificationState(slots({
      _tag: "Requested",
    }), PRIMARY_SLOT_ID)).toMatchObject({
      message: "Waiting to load the selected model",
      priority: "activity",
    })
    expect(deriveSelectedModelResidencyNotificationState(slots({
      _tag: "Failed",
      failure: { code: "load_failed", message: "Could not load", retryable: true },
    }), PRIMARY_SLOT_ID)).toMatchObject({
      message: "Could not load",
      priority: "error",
    })
  })

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
