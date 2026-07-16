import { Context, Data, Effect, Layer, PubSub, Stream } from "effect"
import type { LocalInferenceUsageSelection } from "@magnitudedev/protocol"
import {
  MagnitudeStorage,
  type ConfigStorageShape,
  type DurableLocalModelBinding,
  type LocalInferenceConfig,
  type MagnitudeConfig,
  type SlotId,
  type SlotModelConfig,
} from "@magnitudedev/storage"

export class ModelConfigurationError extends Data.TaggedError("ModelConfigurationError")<{
  readonly operation: string
  readonly reason: string
  readonly cause?: unknown
}> {}

export interface LocalModelConfigurationApi {
  readonly get: Effect.Effect<LocalInferenceConfig, ModelConfigurationError>
  readonly updateUsage: (usage: LocalInferenceUsageSelection) => Effect.Effect<void, ModelConfigurationError>
  readonly updateSlots: (
    slots: Partial<Record<SlotId, SlotModelConfig>>,
  ) => Effect.Effect<void, ModelConfigurationError>
  readonly reconcileSlots: (
    candidates: readonly LocalSlotCandidate[],
  ) => Effect.Effect<boolean, ModelConfigurationError>
  readonly recordUse: (slotId: SlotId | "selected", providerModelId: string) => Effect.Effect<void, ModelConfigurationError>
  readonly activateLocal: (binding: DurableLocalModelBinding) => Effect.Effect<void, ModelConfigurationError>
  readonly disableLocal: Effect.Effect<void, ModelConfigurationError>
  readonly changes: Stream.Stream<void>
}

export interface LocalSlotCandidate {
  readonly providerModelId: string
  readonly availability: "available" | "disabled"
  readonly ownership: "managed" | "external"
  readonly residency: "loaded" | "sleeping" | "unloaded" | "loading" | "failed" | "unknown"
  readonly productRank: number
  readonly externalPriority: number
}

export class LocalModelConfiguration extends Context.Tag("LocalModelConfiguration")<
  LocalModelConfiguration,
  LocalModelConfigurationApi
>() {}

const configurationError = (operation: string, cause: unknown): ModelConfigurationError =>
  new ModelConfigurationError({
    operation,
    reason: cause instanceof Error ? cause.message : String(cause),
    cause,
  })

const localSlot = (binding: DurableLocalModelBinding): SlotModelConfig => ({
  providerId: "llamacpp",
  providerModelId: binding.providerModelId,
})

const RECENCY_LIMIT = 32
const moveToFront = (items: readonly string[], providerModelId: string): readonly string[] =>
  [providerModelId, ...items.filter((item) => item !== providerModelId)].slice(0, RECENCY_LIMIT)

const sameSlot = (left: SlotModelConfig | undefined, right: SlotModelConfig | undefined): boolean =>
  left?.providerId === right?.providerId
  && left?.providerModelId === right?.providerModelId
  && left?.reasoningEffort === right?.reasoningEffort

export const reconcileLocalModelSlots = (
  current: MagnitudeConfig,
  candidates: readonly LocalSlotCandidate[],
): { readonly config: MagnitudeConfig; readonly changed: boolean } => {
  const models = current.models ?? {}
  const slots = { ...(models.slots ?? {}) }
  const intent = { ...(models.localSlotIntent ?? {}) }
  const available = candidates.filter((candidate) => candidate.availability === "available")
  let changed = false

  for (const slotId of ["primary", "secondary"] as const) {
    const existing = slots[slotId]
    if (existing?.providerId && existing.providerId !== "llamacpp") {
      if (intent[slotId] !== "cloud") { intent[slotId] = "cloud"; changed = true }
      continue
    }
    if (intent[slotId] !== "local" && existing?.providerId !== "llamacpp") continue
    if (intent[slotId] !== "local") { intent[slotId] = "local"; changed = true }
    const recency = models.localModelRecency?.[slotId] ?? []
    const recencyIndex = (id: string): number => {
      const index = recency.indexOf(id)
      return index < 0 ? Number.MAX_SAFE_INTEGER : index
    }
    const rank = (left: LocalSlotCandidate, right: LocalSlotCandidate): number =>
      recencyIndex(left.providerModelId) - recencyIndex(right.providerModelId)
      || Number(right.providerModelId === existing?.providerModelId) - Number(left.providerModelId === existing?.providerModelId)
      || left.externalPriority - right.externalPriority
      || left.productRank - right.productRank
      || left.providerModelId.localeCompare(right.providerModelId)
    const resident = (candidate: LocalSlotCandidate): boolean => candidate.residency === "loaded" || candidate.residency === "sleeping"
    const externalLoaded = available.filter((candidate) => candidate.ownership === "external" && resident(candidate)).sort(rank)
    const managedLoaded = available.filter((candidate) => candidate.ownership === "managed" && resident(candidate)).sort(rank)
    const currentCandidate = available.find((candidate) => candidate.providerModelId === existing?.providerModelId)
    const recentCandidate = recency.map((id) => available.find((candidate) => candidate.providerModelId === id)).find((candidate): candidate is LocalSlotCandidate => candidate !== undefined)
    const bestManaged = available.filter((candidate) => candidate.ownership === "managed").sort(rank)[0]
    const selected = externalLoaded[0] ?? managedLoaded[0] ?? currentCandidate ?? recentCandidate ?? bestManaged
    const next = selected
      ? { ...existing, providerId: "llamacpp", providerModelId: selected.providerModelId }
      : undefined
    if (!sameSlot(existing, next)) {
      if (next) slots[slotId] = next
      else delete slots[slotId]
      changed = true
    }
  }
  if (!changed) return { config: current, changed: false }
  return { config: { ...current, models: { ...models, slots, localSlotIntent: intent } }, changed: true }
}

type LocalModelConfigurationStorage = Pick<
  ConfigStorageShape,
  "getLocalInferenceConfig" | "updateModelConfig" | "update"
>

export const makeLocalModelConfiguration = (
  storage: LocalModelConfigurationStorage,
): Effect.Effect<LocalModelConfigurationApi> => Effect.gen(function* () {
  const changes = yield* PubSub.unbounded<void>()
  const publish = PubSub.publish(changes, undefined).pipe(Effect.asVoid)

  const get = storage.getLocalInferenceConfig().pipe(
    Effect.map((config) => config ?? {}),
    Effect.mapError((cause) => configurationError("read local model configuration", cause)),
  )

  return LocalModelConfiguration.of({
    get,

    updateSlots: (updates) => storage.update((current) => {
      const models = current.models ?? {}
      if (Object.keys(updates).length === 0) {
        return { ...current, models: { ...models, slots: {}, localSlotIntent: {} } }
      }
      const slots = { ...(models.slots ?? {}) }
      const intent = { ...(models.localSlotIntent ?? {}) }
      const recency = { ...(models.localModelRecency ?? {}) }
      for (const slotId of ["primary", "secondary"] as const) {
        const slot = updates[slotId]
        if (!slot) continue
        if (slot.providerId || slot.providerModelId || slot.reasoningEffort) slots[slotId] = slot
        else delete slots[slotId]
        if (slot.providerId === "llamacpp") {
          intent[slotId] = "local"
          if (slot.providerModelId) recency[slotId] = moveToFront(recency[slotId] ?? [], slot.providerModelId)
        } else if (slot.providerId) {
          intent[slotId] = "cloud"
        }
      }
      return { ...current, models: { ...models, slots, localSlotIntent: intent, localModelRecency: recency } }
    }).pipe(
      Effect.mapError((cause) => configurationError("update model slots", cause)),
      Effect.asVoid,
      Effect.zipRight(publish),
    ),

    reconcileSlots: (candidates) => Effect.gen(function* () {
      let changed = false
      yield* storage.update((current) => {
        const result = reconcileLocalModelSlots(current, candidates)
        changed = result.changed
        return result.config
      }).pipe(Effect.mapError((cause) => configurationError("reconcile local model slots", cause)))
      if (changed) yield* publish
      return changed
    }),

    recordUse: (slotId, providerModelId) => storage.update((current) => {
      const models = current.models ?? {}
      const recency = { ...(models.localModelRecency ?? {}) }
      const targets = slotId === "selected"
        ? (["primary", "secondary"] as const).filter((candidate) => {
            const slot = models.slots?.[candidate]
            return slot?.providerId === "llamacpp" && slot.providerModelId === providerModelId
          })
        : [slotId]
      for (const target of targets) recency[target] = moveToFront(recency[target] ?? [], providerModelId)
      return { ...current, models: { ...models, localModelRecency: recency } }
    }).pipe(
      Effect.mapError((cause) => configurationError("record local model use", cause)),
      Effect.asVoid,
    ),

    updateUsage: (usage) => storage.update((current) => {
      const previous = current.localInference?.usage
      const changed = previous?.localModelRole !== usage.localModelRole
        || previous.sessionConcurrency !== usage.sessionConcurrency
      if (!changed) return current
      const existing = current.models?.slots ?? {}
      const primary = existing.primary?.providerId === "llamacpp" ? undefined : existing.primary
      const secondary = existing.secondary?.providerId === "llamacpp" ? undefined : existing.secondary
      const localSlotIntent = { ...(current.models?.localSlotIntent ?? {}) }
      if (existing.primary?.providerId === "llamacpp") delete localSlotIntent.primary
      if (existing.secondary?.providerId === "llamacpp") delete localSlotIntent.secondary
      return {
        ...current,
        localInference: { usage },
        models: {
          ...current.models,
          slots: {
            ...(primary ? { primary } : {}),
            ...(secondary ? { secondary } : {}),
          },
          localSlotIntent,
        },
      }
    }).pipe(
      Effect.mapError((cause) => configurationError("update local inference usage", cause)),
      Effect.zipRight(publish),
    ),

    activateLocal: (binding) => storage.update((current) => {
      const usage = current.localInference?.usage
      if (!usage) return current
      const existingSlots = current.models?.slots ?? {}
      const slots = usage.localModelRole === "main"
        ? { ...existingSlots, primary: localSlot(binding) }
        : { ...existingSlots, secondary: localSlot(binding) }
      return {
        ...current,
        localInference: { ...current.localInference, binding },
        models: {
          ...current.models,
          slots,
          localSlotIntent: {
            ...(current.models?.localSlotIntent ?? {}),
            [usage.localModelRole === "main" ? "primary" : "secondary"]: "local",
          },
          localModelRecency: {
            ...(current.models?.localModelRecency ?? {}),
            [usage.localModelRole === "main" ? "primary" : "secondary"]: moveToFront(
              current.models?.localModelRecency?.[usage.localModelRole === "main" ? "primary" : "secondary"] ?? [],
              binding.providerModelId,
            ),
          },
        },
      }
    }).pipe(
      Effect.mapError((cause) => configurationError("activate local model", cause)),
      Effect.flatMap((config) => config.localInference?.usage && config.localInference.binding
        ? Effect.void
        : Effect.fail(new ModelConfigurationError({
          operation: "activate local model",
          reason: "Choose local inference usage before activating a model.",
        }))),
      Effect.zipRight(publish),
    ),

    disableLocal: storage.update((current) => {
      const existing = current.models?.slots ?? {}
      const primary = existing.primary?.providerId === "llamacpp" ? undefined : existing.primary
      const secondary = existing.secondary?.providerId === "llamacpp" ? undefined : existing.secondary
      const slots = {
        ...(primary ? { primary } : {}),
        ...(secondary ? { secondary } : {}),
      }
      const { binding: _, ...localInference } = current.localInference ?? {}
      const localSlotIntent = { ...(current.models?.localSlotIntent ?? {}) }
      if (existing.primary?.providerId === "llamacpp") delete localSlotIntent.primary
      if (existing.secondary?.providerId === "llamacpp") delete localSlotIntent.secondary
      return {
        ...current,
        localInference,
        models: { ...current.models, slots, localSlotIntent },
      }
    }).pipe(
      Effect.mapError((cause) => configurationError("disable local model", cause)),
      Effect.zipRight(publish),
    ),

    changes: Stream.fromPubSub(changes),
  })
})

export const LocalModelConfigurationLive: Layer.Layer<
  LocalModelConfiguration,
  never,
  MagnitudeStorage
> = Layer.effect(
  LocalModelConfiguration,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    return yield* makeLocalModelConfiguration(storage.config)
  }),
)
