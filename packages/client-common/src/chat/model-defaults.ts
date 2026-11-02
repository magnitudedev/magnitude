import { Equal, Option } from "effect"
import {
  type ChatModelDefaults,
  type InferenceModel,
  type RequestedResponseSettings,
  type SavedChatModelDefaults,
  ServingModelId,
} from "@magnitudedev/sdk"
import { defaultResponseSettings, type ChatRequestSettings } from "./draft"
import { reasoningEffortFor } from "./models"

/** A setting a model's defaults hold: every requested setting and the reasoning level. */
export type ChatDefaultsKey = keyof RequestedResponseSettings | "reasoningEffort"

/** Where a chat's settings stand against the model's starting point. */
export type ChatDefaultsStatus = "standard" | "changedFromStandard" | "yours" | "changedFromYours"

const settingKeys = Object.keys(defaultResponseSettings) as ReadonlyArray<keyof RequestedResponseSettings>

/** The settings a new chat with this model starts from: its saved defaults, else the standard settings. */
export const chatStartingSettings = (
  model: InferenceModel,
  saved: SavedChatModelDefaults
): ChatRequestSettings => {
  const defaults = saved[ServingModelId.make(model.id)]
  return {
    ...(defaults?.settings ?? defaultResponseSettings),
    model: Option.some(ServingModelId.make(model.id)),
    reasoningEffort: reasoningEffortFor(model, defaults?.reasoningEffort ?? Option.none()),
  }
}

/** These settings as the defaults to save for this model. */
export const chatModelDefaults = (settings: ChatRequestSettings, model: InferenceModel): ChatModelDefaults => ({
  reasoningEffort: reasoningEffortFor(model, settings.reasoningEffort),
  settings: Object.fromEntries(settingKeys.map((key) => [key, settings[key]])) as RequestedResponseSettings,
})

const same = (a: unknown, b: unknown): boolean =>
  Array.isArray(a) && Array.isArray(b)
    ? a.length === b.length && a.every((value, index) => Equal.equals(value, b[index]))
    : Equal.equals(a, b)

/** The settings in which this chat differs from where a new chat with its model starts. */
export const changedChatDefaults = (
  settings: ChatRequestSettings,
  model: InferenceModel,
  saved: SavedChatModelDefaults
): ReadonlySet<ChatDefaultsKey> => {
  const start = chatStartingSettings(model, saved)
  const changed = new Set<ChatDefaultsKey>(settingKeys.filter((key) => !same(settings[key], start[key])))
  if (!Equal.equals(reasoningEffortFor(model, settings.reasoningEffort), start.reasoningEffort)) {
    changed.add("reasoningEffort")
  }
  return changed
}

export const chatDefaultsStatus = (
  model: InferenceModel,
  saved: SavedChatModelDefaults,
  changed: ReadonlySet<ChatDefaultsKey>
): ChatDefaultsStatus =>
  ServingModelId.make(model.id) in saved
    ? changed.size
      ? "changedFromYours"
      : "yours"
    : changed.size
      ? "changedFromStandard"
      : "standard"
