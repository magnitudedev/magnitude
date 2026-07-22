import { Array as Arr, Effect, Option, Predicate } from "effect"
import { Generated } from "@magnitudedev/icn"
import type { IcnInventoryService } from "@magnitudedev/icn/inventory"
import type { LocalModelConfigurationApi } from "../model-configuration"

const PROFILE_PARALLEL_SEQUENCES = 1

export type ServingConfigurationInventory = Pick<
  IcnInventoryService,
  "get" | "configureModelServing"
>

/**
 * Applies ACN's durable product selection to ICN's authoritative model record.
 * This is the only translation from a selected product profile to ICN serving
 * configuration; callers may invoke it repeatedly because the operation is idempotent.
 */
export const reconcileSelectedServingConfiguration = (
  inventory: ServingConfigurationInventory,
  configuration: LocalModelConfigurationApi,
  currentModels: Option.Option<readonly Generated.Model[]> = Option.none(),
) => Effect.gen(function* () {
  const models = Option.isSome(currentModels)
    ? currentModels.value
    : (yield* inventory.get).state.data
  const config = yield* configuration.get
  const selected = Option.fromNullable(config.selectedProfile)
  if (Option.isNone(selected)) return models

  const providerModelId = Option.fromNullable(selected.value.providerModelId)
  if (Option.isNone(providerModelId)) return models

  const model = Arr.findFirst(models, (candidate) => candidate.id === providerModelId.value)
  if (Option.isNone(model)) return models
  if (Option.exists(Option.filter(model.value.serving_configuration, Predicate.isNotNull), (serving) =>
    serving.profile.context_length === selected.value.contextTokens
    && serving.profile.parallel_sequences === PROFILE_PARALLEL_SEQUENCES)) return models

  const updated = yield* inventory.configureModelServing({
    path: { model_id: model.value.id },
    payload: {
      context_length: selected.value.contextTokens,
      parallel_sequences: PROFILE_PARALLEL_SEQUENCES,
    },
  })
  return models.map((candidate) => candidate.id === updated.id ? updated : candidate)
})
