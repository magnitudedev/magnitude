import { Array as Arr, Data, Effect, Option, Stream } from "effect"
import { IcnClient, IcnInventory, IcnRecipes, type Generated } from "@magnitudedev/icn"
import {
  LocalModelConfiguration,
  type ModelSlotsConfiguration,
} from "../model-configuration"
import { reconcileSelectedServingConfiguration } from "./serving-configuration"

const PROFILE_PARALLEL_SEQUENCES = 1
const EMPTY_MODEL_SLOTS: ModelSlotsConfiguration = {}

export class ModelRecipeNotFound extends Data.TaggedError("ModelRecipeNotFound")<{
  readonly configurationId: string
}> {}

export class InventoryModelNotFound extends Data.TaggedError("InventoryModelNotFound")<{
  readonly modelId: string
}> {}

export const downloadLocalModel = (configurationId: string) => Effect.gen(function* () {
  const client = yield* IcnClient
  const recipes = yield* IcnRecipes
  const configuration = yield* LocalModelConfiguration
  const recommendation = yield* recipes.resolve(configurationId)
  if (Option.isNone(recommendation)) {
    return yield* new ModelRecipeNotFound({ configurationId })
  }

  const selected = {
    configurationId,
    catalogModelId: recommendation.value.catalogModelId,
    contextTokens: recommendation.value.contextTokens,
  }
  yield* configuration.selectProfile(selected)

  const request: Generated.DownloadModelRequestSchema = {
    source: {
      type: "hugging_face",
      repository: recommendation.value.repo,
      revision: recommendation.value.revision,
    },
    components: recommendation.value.files.map((file, index) => ({
      path: file.path,
      role: file.role,
      expected_sha256: Option.fromNullable(file.sha256),
      shard_index: index === 0 ? Option.none() : Option.some(index),
    })),
    relationships: [],
    serving_profile: {
      context_length: recommendation.value.contextTokens,
      parallel_sequences: PROFILE_PARALLEL_SEQUENCES,
    },
  }
  const response = yield* client.models.downloadModel({ payload: request })
  yield* response.events.pipe(
    Stream.runForEach((event) => event.type === "ready"
        ? configuration.selectProfile({
            ...selected,
            providerModelId: event.model.id,
          })
        : Effect.void),
  )
})

export const activateLocalModel = (modelId: string) => Effect.gen(function* () {
  const client = yield* IcnClient
  const inventory = yield* IcnInventory
  const models = (yield* inventory.get).state.data
  const model = Arr.findFirst(models, (candidate) => candidate.id === modelId)
  if (Option.isNone(model)) return yield* new InventoryModelNotFound({ modelId })

  yield* reconcileSelectedServingConfiguration(Option.some(models))
  const response = yield* client.models.loadModel({ path: { model_id: model.value.id } })
  yield* response.events.pipe(Stream.runDrain)
})

export const deleteLocalModel = (modelId: string) => Effect.gen(function* () {
  const client = yield* IcnClient
  const inventory = yield* IcnInventory
  const model = Arr.findFirst(
    (yield* inventory.get).state.data,
    (candidate) => candidate.id === modelId,
  )
  if (Option.isNone(model)) return yield* new InventoryModelNotFound({ modelId })

  yield* client.models.deleteModel({
    path: { model_id: model.value.id },
    urlParams: { dry_run: Option.none() },
  })
})

export const restartLocalInference = Effect.gen(function* () {
  const client = yield* IcnClient
  const inventory = yield* IcnInventory
  const loaded = Arr.findFirst(
    (yield* inventory.get).state.data,
    (model) => model.residency.type === "loaded",
  )
  if (Option.isNone(loaded)) return

  yield* client.models.unloadModel({ path: { model_id: loaded.value.id } })
  const response = yield* client.models.loadModel({ path: { model_id: loaded.value.id } })
  yield* response.events.pipe(Stream.runDrain)
})

export const disableLocalInference = Effect.gen(function* () {
  const client = yield* IcnClient
  const inventory = yield* IcnInventory
  const configuration = yield* LocalModelConfiguration
  const models = yield* configuration.getModels
  const slots = Option.fromNullable(models.slots).pipe(
    Option.getOrElse(() => EMPTY_MODEL_SLOTS),
  )
  const updates = Object.fromEntries(
    (["primary", "secondary"] as const)
      .filter((slotId) => Option.exists(
        Option.fromNullable(slots[slotId]),
        (slot) => slot.providerId === "local",
      ))
      .map((slotId) => [slotId, {}]),
  )
  if (Object.keys(updates).length > 0) yield* configuration.updateSlots(updates)

  const loaded = (yield* inventory.get).state.data.filter(
    (model) => model.residency.type === "loaded",
  )
  yield* Effect.forEach(
    loaded,
    (model) => client.models.unloadModel({ path: { model_id: model.id } }),
    { discard: true },
  )
})
