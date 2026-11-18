import { useRef, useState } from "react"
import { Option, Schema } from "effect"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { type InferenceModel, RequestedResponseSettings, type SavedChatModelDefaults, ServingModelId } from "@magnitudedev/sdk"
import {
  changedChatDefaults,
  chatModelDefaults,
  chatStartingSettings,
  reasoningEffortFor,
  reasoningLevels,
  type ChatRequestSettings,
  useAgentClient,
} from "@magnitudedev/client-common"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select"
import { SettingsGroup, SettingsRow } from "../settings-row"
import { ErrorNotice } from "../error-notice"
import { ChatSettingsFields } from "./settings"
import { useChatModels } from "./served-models"
import { errorText } from "./shared"

const SAVE_DELAY_MS = 400

/**
 * Settings → Chat: each model's settings, which every chat with that model uses from its next message. A chat
 * itself chooses only its model and thinking level.
 */
export function ChatModelSettings() {
  const client = useAgentClient()
  const { result, models, nameOf, contextFor, contextLimitFor } = useChatModels()
  const savedResult = useAtomValue(client.Chat.ListChatModelDefaults({})).result
  const [chosen, setChosen] = useState<string | undefined>(undefined)
  const model = models.find(entry => entry.id === chosen) ?? models[0]
  if (Result.isFailure(result) || Result.isFailure(savedResult)) return <ErrorNotice title="Couldn’t read chat settings" description={errorText(Result.isFailure(result) ? result : savedResult) ?? "Try again."} />
  if (!Result.isSuccess(result) || !Result.isSuccess(savedResult)) return null
  if (!model) return <p className="text-sm text-slate-500">Download a model to set its chat settings.</p>
  return <>
    <SettingsGroup>
      <SettingsRow label="Model" hint="Each model keeps its own settings. Every chat with it uses them from its next message."
        control={<Select items={models.map(entry => ({ value: entry.id, label: nameOf(entry) }))} value={model.id} onValueChange={next => setChosen(next ?? undefined)}>
          <SelectTrigger aria-label="Model to set" className="w-64 text-[13px]"><SelectValue /></SelectTrigger>
          <SelectContent>{models.map(entry => <SelectItem key={entry.id} value={entry.id}>{nameOf(entry)}</SelectItem>)}</SelectContent>
        </Select>} />
    </SettingsGroup>
    <ModelSettings key={model.id} model={model} saved={savedResult.value} context={contextFor(model)} contextLimit={contextLimitFor(model)} />
  </>
}

/** One model's settings, saved a moment after each change; Reset returns it to its standard settings. */
function ModelSettings({ model, saved, context, contextLimit }: { model: InferenceModel; saved: SavedChatModelDefaults; context: number; contextLimit: number | undefined }) {
  const client = useAgentClient()
  const save = useAtomSet(client.Chat.SaveChatModelDefaults)
  const clear = useAtomSet(client.Chat.ClearChatModelDefaults)
  const saving = useAtomValue(client.Chat.SaveChatModelDefaults)
  const [edits, setEdits] = useState<Partial<ChatRequestSettings>>({})
  const pending = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const shown: ChatRequestSettings = { ...chatStartingSettings(model, saved), ...edits }
  const valid = (settings: ChatRequestSettings) => Schema.is(Schema.typeSchema(RequestedResponseSettings))(settings)
  const standard = changedChatDefaults(shown, model, {})
  const customized = ServingModelId.make(model.id) in saved || standard.size > 0
  const patch = (change: Partial<ChatRequestSettings>) => {
    const next = { ...edits, ...change }
    setEdits(next)
    clearTimeout(pending.current)
    const settings = { ...chatStartingSettings(model, saved), ...next }
    if (valid(settings)) pending.current = setTimeout(() => save({ model: ServingModelId.make(model.id), defaults: chatModelDefaults(settings, model) }), SAVE_DELAY_MS)
  }
  const reset = () => {
    clearTimeout(pending.current)
    setEdits({})
    clear({ model: ServingModelId.make(model.id) })
  }
  const levels = reasoningLevels(model)
  const effort = Option.getOrUndefined(reasoningEffortFor(model, shown.reasoningEffort))
  return <SettingsGroup>
    <SettingsRow label={customized ? "Customized" : "Standard settings"}
      hint={customized ? "Changed settings are marked with a dot." : "The settings this model's makers recommend."}
      alert={Result.isFailure(saving) && !saving.waiting ? <ErrorNotice title="These settings weren’t saved" description={errorText(saving) ?? "Try again."} /> : undefined}
      control={customized && <button type="button" className="text-xs font-medium text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100" onClick={reset}>Reset to standard</button>} />
    {levels.length > 1 && <SettingsRow label="Thinking" hint="Where new chats start; each chat can change it beside Send."
      control={<div role="radiogroup" aria-label="Default thinking level" className="model-picker-levels">
        {levels.map(level => <button key={level.value} type="button" role="radio" aria-checked={level.value === effort} onClick={() => patch({ reasoningEffort: Option.some(level.value) })}>{level.label}</button>)}
      </div>} />}
    <div className="px-4 pb-5">
      {!valid(shown) && <p role="alert" className="mt-3 text-xs text-red-600 dark:text-red-400">Some values are out of range, so they haven't been saved.</p>}
      <ChatSettingsFields settings={shown} patch={patch} context={context || undefined} contextLimit={contextLimit}
        maxOutput={model.top_provider.max_completion_tokens || undefined} changed={standard} />
    </div>
  </SettingsGroup>
}
