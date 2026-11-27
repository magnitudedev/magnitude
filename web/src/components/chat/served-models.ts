import { Option } from "effect"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import type { InferenceModel, LocalModelsState } from "@magnitudedev/sdk"
import {
  activeLocalModel,
  formatLocalModelDisplayName,
  localModelContextLimit,
  localModelProviderModelId,
  localModelServingProfile,
  readyToChat,
  useAgentClient,
  useLocalModels,
} from "@magnitudedev/client-common"

/**
 * Whether Chat's startup model and settings data is still arriving: before servable models, local
 * model discovery, or saved per-model defaults have settled. Failures settle immediately so errors
 * never stay behind a skeleton.
 */
export const chatModelsPending = ({
  servable,
  local,
  defaults,
  models,
}: {
  readonly servable: Result.Result<unknown, unknown>
  readonly local: Result.Result<LocalModelsState, unknown>
  readonly defaults: Result.Result<unknown, unknown>
  readonly models: readonly InferenceModel[]
}): boolean => {
  if (Result.isFailure(servable) || Result.isFailure(defaults)) return false
  if (Result.isInitial(servable) || Result.isInitial(defaults) || Result.isInitial(local)) return true
  if (!Result.isSuccess(local)) return false
  if (local.value.preparation?.discovery?.complete === false) return true
  return (
    Result.isSuccess(servable) &&
    servable.waiting &&
    models.length === 0 &&
    local.value.models.some(readyToChat)
  )
}

/**
 * The models chats can use: every served model that reads and writes text, with what this computer knows about the
 * ones it runs: their names with quant, the context it serves them at, and the model's own when that is more. A new
 * chat starts on the model loaded here, so its first message doesn't swap that model out, else on the first.
 */
export function useChatModels() {
  const client = useAgentClient()
  const result = useAtomValue(client.Models.ListServableModels({})).result
  const local = useLocalModels()
  const models: readonly InferenceModel[] = Result.isSuccess(result)
    ? result.value.data.filter((model) =>
        model.architecture.input_modalities.includes("text") && model.architecture.output_modalities.includes("text"))
    : []
  const localFor = (id: string) => Result.isSuccess(local)
    ? local.value.models.find((model) => Option.contains(localModelProviderModelId(model), id))
    : undefined
  const nameOf = (model: InferenceModel) => {
    const found = localFor(model.id)
    return found ? formatLocalModelDisplayName(found) : model.name
  }
  const contextFor = (model: InferenceModel) => {
    const found = localFor(model.id)
    return found
      ? Option.getOrElse(Option.map(localModelServingProfile(found), (profile) => profile.contextLength), () => model.context_length)
      : model.context_length
  }
  /** The model's own context when this computer serves it at less. */
  const contextLimitFor = (model: InferenceModel) => {
    const found = localFor(model.id)
    const limit = found ? Option.getOrUndefined(localModelContextLimit(found))?.contextLength : undefined
    return limit !== undefined && limit > contextFor(model) ? limit : undefined
  }
  const active = Result.isSuccess(local) ? Option.flatMap(activeLocalModel(local.value), ({ model }) => localModelProviderModelId(model)) : Option.none<string>()
  const preferred = Option.getOrUndefined(Option.filter(active, (id) => models.some((model) => model.id === id))) ?? models[0]?.id
  return { result, local, models, preferred, localFor, nameOf, contextFor, contextLimitFor }
}
