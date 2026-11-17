import { Option } from "effect"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import type { InferenceModel } from "@magnitudedev/sdk"
import {
  formatLocalModelDisplayName,
  localModelContextLimit,
  localModelProviderModelId,
  localModelServingProfile,
  useAgentClient,
  useLocalModels,
} from "@magnitudedev/client-common"

/**
 * The models chats can use: every served model that reads and writes text, with what this computer knows about the
 * ones it runs: their names with quant, the context it serves them at, and the model's own when that is more.
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
  return { result, models, localFor, nameOf, contextFor, contextLimitFor }
}
