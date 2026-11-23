import { useRef, useState, type KeyboardEvent } from "react"
import { Option } from "effect"
import { Popover } from "@base-ui/react/popover"
import type { InferenceModel, ReasoningEffort, ResponseReasoningEffort } from "@magnitudedev/sdk"
import { reasoningEffortFor, reasoningLevels } from "@magnitudedev/client-common"
import { CheckIcon } from "@phosphor-icons/react"
import { NamedLogo, labForName } from "../model-logo"

/** One model the picker offers: what it is called, the context it is served at here, and what else to know about it. */
export interface PickerModel {
  readonly model: InferenceModel
  readonly name: string
  readonly context: number
  readonly note: string | null
}

const contextLabel = (tokens: number) => tokens >= 1024 ? `${Math.round(tokens / 1024)}K` : `${tokens}`

/** What the trigger says after the model's name: its thinking level, or nothing for a model without thinking. */
export const thinkingSuffix = (model: InferenceModel | undefined, selected: Option.Option<ResponseReasoningEffort>) => {
  const levels = reasoningLevels(model)
  if (levels.length === 0) return null
  const effort = Option.getOrUndefined(reasoningEffortFor(model, selected))
  const label = levels.find(level => level.value === effort)?.label ?? levels[0]!.label
  return label === "Off" ? "Thinking off" : label
}

/**
 * The composer's model and thinking control: one pill naming the model and its thinking level, opening a popover
 * with the thinking levels on top and the models below. Choosing a model selects it and closes the popover.
 */
export function ModelPicker({ models, model, name, selected, onModel, onEffort, onManage }: {
  models: readonly PickerModel[]
  model: InferenceModel | undefined
  name: string
  selected: Option.Option<ResponseReasoningEffort>
  onModel: (model: InferenceModel) => void
  onEffort: (effort: ReasoningEffort) => void
  onManage: () => void
}) {
  const [open, setOpen] = useState(false)
  const list = useRef<HTMLDivElement>(null)
  const levels = reasoningLevels(model)
  const effort = Option.getOrUndefined(reasoningEffortFor(model, selected))
  const suffix = thinkingSuffix(model, selected)
  const step = (by: number) => {
    const index = levels.findIndex(level => level.value === effort)
    const next = levels[Math.min(levels.length - 1, Math.max(0, index + by))]
    if (next && next.value !== effort) onEffort(next.value)
  }
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault()
      step(event.key === "ArrowLeft" ? -1 : 1)
      return
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return
    event.preventDefault()
    const rows = [...(list.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? [])]
    const index = rows.indexOf(document.activeElement as HTMLButtonElement)
    rows[index === -1 ? 0 : (index + (event.key === "ArrowDown" ? 1 : rows.length - 1)) % rows.length]?.focus()
  }
  return <Popover.Root open={open} onOpenChange={setOpen}>
    <Popover.Trigger disabled={models.length === 0} aria-label={`Model and thinking: ${name}${suffix ? `, ${suffix}` : ""}`}
      className="model-picker-trigger">
      {model && <NamedLogo name={model.name} lab={labForName(model.name)} className="size-3 shrink-0" />}
      <span className="truncate">{name}</span>
      {suffix && <span className="shrink-0 text-slate-400 dark:text-slate-500">{suffix}</span>}
    </Popover.Trigger>
    <Popover.Portal>
      <Popover.Positioner side="top" align="end" sideOffset={6} className="isolate z-50">
        <Popover.Popup className="chat-popover w-80" onKeyDown={onKeyDown}>
          {levels.length > 0 && <div className="model-picker-thinking">
            <span>Thinking</span>
            <div role="radiogroup" aria-label="Thinking level" className="model-picker-levels">
              {levels.map(level => <button key={level.value} type="button" role="radio" aria-checked={level.value === effort}
                onClick={() => onEffort(level.value)}>{level.label}</button>)}
            </div>
          </div>}
          <div ref={list} role="listbox" aria-label="Model" className="chat-popover-list">
            {models.map(entry => {
              const current = entry.model.id === model?.id
              return <button key={entry.model.id} type="button" role="option" aria-selected={current}
                onClick={() => { onModel(entry.model); setOpen(false) }}
                onKeyDown={event => { if (event.key === "Enter") { event.preventDefault(); onModel(entry.model); setOpen(false) } }}>
                <span className="chat-popover-row">
                  <NamedLogo name={entry.model.name} lab={labForName(entry.model.name)} className="size-3 shrink-0" />
                  <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                  <span className="shrink-0 text-xs text-slate-400">{contextLabel(entry.context)}</span>
                  <span className="flex size-4 shrink-0 items-center justify-center">{current && <CheckIcon className="size-3.5" />}</span>
                </span>
                {entry.note && <span className="model-picker-row-note">{entry.note}</span>}
              </button>
            })}
          </div>
          <button type="button" className="model-picker-manage" onClick={() => { setOpen(false); onManage() }}>Manage models…</button>
        </Popover.Popup>
      </Popover.Positioner>
    </Popover.Portal>
  </Popover.Root>
}
