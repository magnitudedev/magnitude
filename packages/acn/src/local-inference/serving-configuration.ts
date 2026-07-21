import { Effect, Option } from "effect"
import { IcnApiClient, Generated } from "@magnitudedev/icn"
import { LOCAL_MODEL_CATALOG_OVERLAY } from "./catalog"
import type { LocalModelConfigurationApi } from "./model-configuration"

const PROFILE_PARALLEL_SEQUENCES = 1

const legacyModelForSelection = (
  models: readonly Generated.Model[],
  catalogModelId: string,
): Generated.Model | undefined => {
  const artifact = LOCAL_MODEL_CATALOG_OVERLAY.models
    .flatMap((model) => model.artifacts)
    .find((candidate) => candidate.id === catalogModelId)
  if (!artifact) return undefined
  return models.find((model) => {
    if (model.source.type !== "hugging_face" || model.source.repository !== artifact.repository) return false
    const components = model.location.type === "file"
      ? [model.location.component]
      : model.location.components
    return components.some((component) => component.path.includes(artifact.filenameIncludes))
  })
}

/**
 * Applies ACN's durable product selection to ICN's authoritative model record.
 * This is the only translation from a selected product profile to ICN serving
 * configuration; callers may invoke it repeatedly because the operation is idempotent.
 */
export const reconcileSelectedServingConfiguration = (
  client: IcnApiClient,
  configuration: LocalModelConfigurationApi,
  inventory?: readonly Generated.Model[],
) => Effect.gen(function* () {
  const models = inventory ?? (yield* client.models.listModels({})).data
  const config = yield* configuration.get
  let selected = config.selectedProfile
  if (!selected) return models

  if (!selected.providerModelId) {
    const legacyModel = legacyModelForSelection(models, selected.catalogModelId)
    if (!legacyModel) return models
    selected = { ...selected, providerModelId: legacyModel.id }
    yield* configuration.selectProfile(selected)
  }

  const model = models.find((candidate) => candidate.id === selected.providerModelId)
  if (!model) return models
  const serving = Option.getOrUndefined(model.serving_configuration)
  if (
    serving?.profile.context_length === selected.contextTokens
    && serving.profile.parallel_sequences === PROFILE_PARALLEL_SEQUENCES
  ) return models

  const updated = yield* client.models.configureModelServing({
    path: { model_id: model.id },
    payload: {
      context_length: selected.contextTokens,
      parallel_sequences: PROFILE_PARALLEL_SEQUENCES,
    },
  })
  return models.map((candidate) => candidate.id === updated.id ? updated : candidate)
})
