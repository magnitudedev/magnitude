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
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip"

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
    : `${label} is the only thinking option for this model.`

  return (
    <Tooltip disabled={!disabled}>
      <TooltipTrigger
        delay={0}
        closeOnClick={false}
        render={
          // The disabled select cannot receive pointer events. Its focusable
          // wrapper owns the explanation and the disabled cursor instead.
          <span
            tabIndex={disabled ? 0 : undefined}
            aria-label={disabled ? `Thinking level: ${label}` : undefined}
            aria-description={disabled ? explanation : undefined}
            className={`inline-flex rounded-md outline-none focus-visible:ring-2 focus-visible:ring-blue-500/30 ${disabled ? "cursor-not-allowed" : ""}`}
          >
            <Select
              value={effort ?? ""}
              disabled={disabled}
              onValueChange={(next) => {
                if (next) onChange(next)
              }}
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
          </span>
        }
      />
      <TooltipContent>{explanation}</TooltipContent>
    </Tooltip>
  )
}
