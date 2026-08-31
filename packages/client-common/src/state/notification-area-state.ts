import { Atom } from "@effect-atom/atom-react"
import { createId } from "@magnitudedev/generate-id"
import type {
  AcnRecoveryState,
  LocalModelsState,
  ModelSlotsState,
  ProviderModelId,
  SlotId,
} from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import { localModelProviderModelId, localModelServingState } from "../local-models/projection"
import { formatMemorySize } from "../utils/format-bytes"

export const NotificationIdSchema = Schema.NonEmptyString.pipe(
  Schema.brand("NotificationId"),
)
export type NotificationId = typeof NotificationIdSchema.Type

export const NotificationPrioritySchema = Schema.Literal(
  "activity",
  "notice",
  "warning",
  "error",
)
export type NotificationPriority = typeof NotificationPrioritySchema.Type

export const NotificationActionSchema = Schema.Literal("openCatalog")
export type NotificationAction = typeof NotificationActionSchema.Type

export const NotificationStateSchema = Schema.Struct({
  id: NotificationIdSchema,
  message: Schema.String,
  compactMessage: Schema.optionalWith(Schema.String, {
    as: "Option",
    exact: true,
  }),
  priority: NotificationPrioritySchema,
  action: Schema.optionalWith(NotificationActionSchema, {
    as: "Option",
    exact: true,
  }),
  createdAt: Schema.Number.pipe(Schema.finite(), Schema.nonNegative()),
})
export type NotificationState = typeof NotificationStateSchema.Type

export const NotificationAreaStateSchema = Schema.Struct({
  notificationStates: Schema.Array(NotificationStateSchema),
})
export type NotificationAreaState = typeof NotificationAreaStateSchema.Type

export const PushNotificationSchema = Schema.Struct({
  message: Schema.String,
  priority: NotificationPrioritySchema,
  action: Schema.optionalWith(NotificationActionSchema, {
    as: "Option",
    exact: true,
  }),
  dismissAfterMilliseconds: Schema.Number.pipe(
    Schema.finite(),
    Schema.positive(),
  ),
})
export type PushNotification = typeof PushNotificationSchema.Type

const notificationAreaSourceStateAtom = Atom.keepAlive(
  Atom.make<NotificationAreaState>({ notificationStates: [] }),
)

export const pushNotificationAtom = Atom.fn<PushNotification>()(
  (input, get) => Effect.gen(function* () {
    const notificationState = NotificationStateSchema.make({
      id: NotificationIdSchema.make(createId()),
      message: input.message,
      compactMessage: Option.none(),
      priority: input.priority,
      action: input.action,
      createdAt: Date.now(),
    })
    yield* Effect.sync(() => {
      const current = get.registry.get(notificationAreaSourceStateAtom)
      get.registry.set(notificationAreaSourceStateAtom, {
        notificationStates: [...current.notificationStates, notificationState],
      })
    })
    yield* Effect.sleep(input.dismissAfterMilliseconds)
    yield* Effect.sync(() => {
      const latest = get.registry.get(notificationAreaSourceStateAtom)
      get.registry.set(notificationAreaSourceStateAtom, {
        notificationStates: latest.notificationStates.filter(
          ({ id }) => id !== notificationState.id,
        ),
      })
    })
  }),
  { concurrent: true },
)

export const dismissNotificationAtom = Atom.fnSync<NotificationId>()((id, get) => {
  const current = get.registry.get(notificationAreaSourceStateAtom)
  get.set(notificationAreaSourceStateAtom, {
    notificationStates: current.notificationStates.filter(
      (notificationState) => notificationState.id !== id,
    ),
  })
})

/**
 * Reading the public state also mounts the two operations that own its
 * event-driven updates and dismissal fibers.
 */
export const notificationAreaStateAtom = Atom.make((get) => {
  get(pushNotificationAtom)
  get(dismissNotificationAtom)
  return get(notificationAreaSourceStateAtom)
})

const persistentNotificationState = (
  id: string,
  message: string,
  priority: NotificationPriority,
  action: Option.Option<NotificationAction> = Option.none(),
  compactMessage: Option.Option<string> = Option.none(),
): NotificationState => NotificationStateSchema.make({
  id: NotificationIdSchema.make(id),
  message,
  compactMessage,
  priority,
  action,
  createdAt: 0,
})

const recoveryActivity = (state: AcnRecoveryState): string | null => {
  if (state._tag !== "Recovering") return null
  const lifecycle = state.lifecycle
  if (lifecycle._tag === "Checking") return "Reconnecting to Magnitude service…"
  if (lifecycle._tag === "Ready" || lifecycle._tag === "Failed") return null
  if (lifecycle._tag === "Installing") {
    if (lifecycle.phase === "StartingMagnitude") return "Starting Magnitude service…"
    const phase = lifecycle.phase === "DownloadingDaemon"
      ? "Downloading Magnitude service"
      : "Downloading inference engine"
    const percentage = Option.match(lifecycle.detail, {
      onNone: () => Math.floor(lifecycle.overallProgress * 100),
      onSome: (progress) => Math.floor(
        Math.max(0, Math.min(1, progress.completed / progress.totalBytes)) * 100,
      ),
    })
    return `${phase} · ${percentage}%`
  }
  if (typeof lifecycle.phase !== "string") {
    const backend = lifecycle.phase.backend
    return `Preparing ${backend._tag} backend for ${backend.hardwareLabel}…`
  }
  switch (lifecycle.phase) {
    case "PreparingAcn": return "Preparing Magnitude service…"
    case "WaitingForOwner": return "Waiting for previous Magnitude service…"
    case "ResolvingLocalInference": return "Preparing local inference…"
    case "LaunchingLocalInference": return "Starting local inference…"
    case "DiscoveringLocalModels": return "Discovering local models…"
  }
}

export const deriveAcnRecoveryNotificationState = (
  state: AcnRecoveryState,
): NotificationState | null => {
  const message = recoveryActivity(state)
  return message === null
    ? null
    : persistentNotificationState("acn-recovery", message, "activity")
}

export const deriveModelDownloadNotificationState = (
  modelsState: LocalModelsState | null,
): NotificationState | null => {
  if (modelsState === null) return null
  const count = modelsState.models.filter((model) => model._tag === "Catalog"
    && (model.acquisitionState._tag === "Installing" || model.acquisitionState._tag === "Updating")).length
  if (count === 0) return null
  return persistentNotificationState(
    "local-model-download",
    `${count} ${count === 1 ? "model" : "models"} downloading`,
    "activity",
    Option.some("openCatalog"),
  )
}

export const deriveSelectedModelLowMemoryNotificationState = (
  modelsState: LocalModelsState | null,
  slotsState: ModelSlotsState | null,
): NotificationState | null => deriveSelectedModelLowMemoryNotificationStateByProviderModelId(
  modelsState,
  slotsState?.slots.primary._tag === "ConfiguredLocal"
    ? slotsState.slots.primary.selection.providerModelId
    : null,
)

export const deriveSelectedModelResidencyNotificationState = (
  slotsState: ModelSlotsState | null,
  slotId: SlotId,
): NotificationState | null => {
  if (slotsState === null) return null
  const slot = slotsState.slots[slotId === "primary" ? "primary" : "secondary"]
  if (slot._tag !== "ConfiguredLocal") return null
  switch (slot.residency._tag) {
    case "Requested":
      return persistentNotificationState(
        `model-residency-${slotId}`,
        "Waiting to load the selected model",
        "activity",
      )
    case "Failed":
      return persistentNotificationState(
        `model-residency-${slotId}`,
        slot.residency.failure.message,
        "error",
      )
    case "Unloaded":
    case "Loading":
    case "Ready":
    case "Stopping":
      return null
  }
}

export const deriveSelectedModelLowMemoryNotificationStateByProviderModelId = (
  modelsState: LocalModelsState | null,
  selectedProviderModelId: ProviderModelId | null,
): NotificationState | null => {
  if (modelsState === null || selectedProviderModelId === null) return null
  const selectedModel = modelsState.models.find((model) =>
    Option.contains(
      localModelProviderModelId(model),
      selectedProviderModelId,
    ))
  if (selectedModel === undefined) return null
  const serving = Option.getOrUndefined(localModelServingState(selectedModel))
  if (serving?._tag !== "Assessed" || serving.assessment._tag !== "Fits") return null
  const currentHeadroomState =
    serving.assessment.memory.currentHeadroomState
  if (currentHeadroomState._tag !== "Insufficient") return null
  const additionalMemory = formatMemorySize(
    currentHeadroomState.minimumAdditionalAvailableBytes,
    { rounding: "up" },
  )
  return persistentNotificationState(
    "selected-local-model-low-memory",
    `Low memory: close memory-intensive apps (need ${additionalMemory}) to load model`,
    "warning",
    Option.none(),
    Option.some(`Low memory: Free ${additionalMemory} to load`),
  )
}

export const deriveLocalModelPersistentNotificationStates = (
  modelsState: LocalModelsState,
  selectedProviderModelId: ProviderModelId | null,
): readonly NotificationState[] => [
  deriveModelDownloadNotificationState(modelsState),
  deriveSelectedModelLowMemoryNotificationStateByProviderModelId(
    modelsState,
    selectedProviderModelId,
  ),
].filter((state): state is NotificationState => state !== null)

export const notificationStatesEquivalent = Schema.equivalence(
  Schema.Array(NotificationStateSchema),
)

const priorityRank: Record<NotificationPriority, number> = {
  activity: 0,
  notice: 1,
  warning: 2,
  error: 3,
}

export const resolveActiveNotificationState = (
  notificationAreaState: NotificationAreaState,
  persistentNotificationStates: readonly (NotificationState | null)[],
): NotificationState | null => {
  const candidates = [
    ...persistentNotificationStates.filter(
      (state): state is NotificationState => state !== null,
    ),
    ...notificationAreaState.notificationStates,
  ]
  let activeState: NotificationState | null = null
  for (const candidate of candidates) {
    if (activeState === null
      || priorityRank[candidate.priority] > priorityRank[activeState.priority]
      || (candidate.priority === activeState.priority
        && candidate.createdAt >= activeState.createdAt)) {
      activeState = candidate
    }
  }
  return activeState
}
