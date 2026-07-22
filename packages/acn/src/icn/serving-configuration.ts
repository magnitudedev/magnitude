import { Array as Arr, Effect, Layer, Option, Predicate, Stream } from "effect"
import { Generated, IcnClient, IcnInventory } from "@magnitudedev/icn"
import { LocalModelConfiguration } from "../model-configuration"

const PROFILE_PARALLEL_SEQUENCES = 1

/**
 * Applies ACN's durable product selection to ICN's authoritative model record.
 * This is the only translation from a selected product profile to ICN serving
 * configuration; callers may invoke it repeatedly because the operation is idempotent.
 */
export const reconcileSelectedServingConfiguration = (
  currentModels: Option.Option<readonly Generated.Model[]> = Option.none(),
) => Effect.gen(function* () {
  const client = yield* IcnClient
  const inventory = yield* IcnInventory
  const configuration = yield* LocalModelConfiguration
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

  const updated = yield* client.models.configureModelServing({
    path: { model_id: model.value.id },
    payload: {
      context_length: selected.value.contextTokens,
      parallel_sequences: PROFILE_PARALLEL_SEQUENCES,
    },
  })
  return models.map((candidate) => candidate.id === updated.id ? updated : candidate)
})

/**
 * Keeps ACN's durable local-model selection applied to ICN's authoritative
 * serving configuration. The initial reconciliation completes before this
 * layer is published; subsequent inventory and configuration changes are
 * reconciled serially for the lifetime of the ACN scope.
 */
export const makeServingConfigurationReconciliation = () => Layer.scopedDiscard(
  Effect.gen(function* () {
    const inventory = yield* IcnInventory
    const configuration = yield* LocalModelConfiguration

    yield* reconcileSelectedServingConfiguration()

    yield* Stream.merge(inventory.changes, configuration.changes).pipe(
      Stream.mapEffect(() => reconcileSelectedServingConfiguration(), { concurrency: 1 }),
      Stream.runDrain,
      Effect.forkScoped,
    )
  }),
)
