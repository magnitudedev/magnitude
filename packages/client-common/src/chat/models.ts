import { Option } from "effect"
import { type InferenceModel, type ReasoningEffort, ReasoningEffortSchema, type ResponseReasoningEffort } from "@magnitudedev/sdk"

const label = (effort: string) =>
  effort === "none" || effort === "off" ? "Off" : effort.charAt(0).toUpperCase() + effort.slice(1)

/** The reasoning levels a model offers. Display labels never change the engine's declared wire values. */
export const reasoningLevels = (
  model: InferenceModel | undefined
): readonly { readonly value: ReasoningEffort; readonly label: string }[] => {
  const reasoning = model ? Option.getOrNull(model.reasoning) : null
  return (
    reasoning?.supported_efforts
      .filter((effort) => !reasoning.mandatory || (effort !== "none" && effort !== "off"))
      .map((value) => ({ value: ReasoningEffortSchema.make(value), label: label(value) })) ?? []
  )
}

/** The level a request to this model asks for: the selection when the model offers it, else its default. */
export const reasoningEffortFor = (
  model: InferenceModel | undefined,
  selected: Option.Option<ResponseReasoningEffort>
): Option.Option<ResponseReasoningEffort> => {
  const levels = reasoningLevels(model)
  if (!levels.length) return Option.none()
  if (Option.exists(selected, (value) => levels.some((level) => level.value === value))) return selected
  const reasoning = model ? Option.getOrNull(model.reasoning) : null
  return Option.some(levels.find((level) => level.value === reasoning?.default_effort)?.value ?? levels[0]!.value)
}
