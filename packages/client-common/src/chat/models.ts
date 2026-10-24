import { Option } from "effect"
import type { ChatModels } from "@magnitudedev/sdk"
type ChatModel = (typeof ChatModels.Type.data)[number]
const label = (effort: string) =>
  effort === "none" || effort === "off"
    ? "Off"
    : effort.charAt(0).toUpperCase() + effort.slice(1)
/** Display labels never change the engine's declared wire values. */
export const chatReasoningLevels = (
  model: ChatModel | undefined
): readonly { readonly value: string; readonly label: string }[] => {
  const reasoning = model ? Option.getOrNull(model.reasoning) : null
  return (
    reasoning?.supported_efforts
      .filter(
        (effort) =>
          !reasoning.mandatory || (effort !== "none" && effort !== "off")
      )
      .map((value) => ({ value, label: label(value) })) ?? []
  )
}
export const chatReasoningEffort = (
  model: ChatModel | undefined,
  selected: Option.Option<string>
): Option.Option<string> => {
  const levels = chatReasoningLevels(model)
  if (!levels.length) return Option.none()
  if (
    Option.exists(selected, (value) =>
      levels.some((level) => level.value === value)
    )
  )
    return selected
  const reasoning = model ? Option.getOrNull(model.reasoning) : null
  return Option.some(
    levels.find((level) => level.value === reasoning?.default_effort)?.value ??
      levels[0]!.value
  )
}
