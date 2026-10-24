import { Option } from "effect"
import type { ChatModels } from "@magnitudedev/sdk"
import {
  chatReasoningEffort,
  chatReasoningLevels,
} from "@magnitudedev/client-common"
import { ChartBarIcon } from "@phosphor-icons/react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select"
import { ActionTooltip } from "../ui/tooltip"

export function ChatThinkingSelect({
  model,
  selected,
  onChange,
}: {
  model: (typeof ChatModels.Type.data)[number] | undefined
  selected: Option.Option<string>
  onChange: (effort: string) => void
}) {
  const levels = chatReasoningLevels(model)
  const effort = Option.getOrUndefined(chatReasoningEffort(model, selected))
  const label = levels.find((level) => level.value === effort)?.label ?? "Off"
  const disabled = levels.length < 2
  const explanation = !model
    ? "Select a model to configure thinking."
    : !levels.length || label === "Off"
    ? "This model does not support thinking."
    : `This model always uses ${label.toLowerCase()} thinking.`

  return (
    <Select
      value={effort ?? ""}
      disabled={disabled}
      onValueChange={(next) => {
        if (next) onChange(next)
      }}
    >
      <ActionTooltip
        label={explanation}
        disabled={!disabled}
        trigger={
          // Disabled buttons do not receive focus or pointer events. The wrapper
          // keeps their explanation available to mouse and keyboard users.
          <span
            tabIndex={disabled ? 0 : undefined}
            aria-label={disabled ? `Thinking level: ${label}` : undefined}
            className="inline-flex rounded-md outline-none focus-visible:ring-2 focus-visible:ring-blue-500/30"
          >
            <SelectTrigger
              aria-label="Thinking level"
              variant="inline"
              className="h-8 px-2 text-xs text-slate-500 hover:bg-slate-100 disabled:pointer-events-none dark:text-slate-400 dark:hover:bg-slate-800"
            >
              <SelectValue>
                <ChartBarIcon aria-hidden="true" className="size-3" />
                {label}
              </SelectValue>
            </SelectTrigger>
          </span>
        }
      />
      <SelectContent
        side="top"
        align="start"
        alignItemWithTrigger={false}
        className="w-36"
      >
        {levels.map((level) => (
          <SelectItem key={level.value} value={level.value}>
            {level.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
