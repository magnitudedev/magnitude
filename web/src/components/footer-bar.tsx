/**
 * FooterBar — spec §9.7
 *
 * Compact runtime controls rendered inside the bottom of the composer.
 */
import { useState } from "react"
import { type ReasoningEffortOption } from "@magnitudedev/client-common"
import type { ContextUsageDisplay, ProviderModelId, ReasoningEffort } from "@magnitudedev/sdk"
import { ContextUsageIndicator } from "./context-usage-indicator"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Button } from "@/components/ui/button"

export interface FooterModelOption {
  readonly value: ProviderModelId
  readonly label: string
}

export interface FooterBarProps {
  /** Context usage info from timeline */
  context: ContextUsageDisplay | null
  /** Token cap (max context window tokens), if known */
  tokenCap?: number | null
  /** Bash mode active */
  bashMode?: boolean
  /** Current model label */
  model?: string | null
  /** Thinking level label (e.g. "High", "Medium") */
  thinkingLevel?: string | null
  /** Resident model allocation label. */
  memoryLabel?: string | null
  /** Next Esc will kill all workers */
  nextEscWillKillAll?: boolean
  /** Transcript mode active */
  transcriptMode?: boolean
  /** Installed models available for primary-slot selection. */
  modelOptions?: readonly FooterModelOption[]
  /** Currently selected provider-model identity. */
  selectedModelId?: ProviderModelId | null
  /** Applies an installed model to the primary model slot. */
  onModelSelect?: (providerModelId: ProviderModelId) => void
  /** Click handler for resident memory (opens Hardware) */
  onMemoryClick?: () => void
  /** Reasoning efforts supported by the selected model. */
  thinkingOptions?: readonly ReasoningEffortOption[]
  /** Currently selected reasoning effort. */
  thinkingEffort?: ReasoningEffort | null
  /** Applies a reasoning effort to the primary model slot. */
  onThinkingSelect?: (effort: ReasoningEffort) => void
}

interface FooterDropdownOption<Value extends string> {
  readonly value: Value
  readonly label: string
}

function FooterDropdown<Value extends string>({
  label,
  value,
  options,
  open,
  tone,
  onOpenChange,
  onSelect,
}: {
  readonly label: string
  readonly value: Value | null
  readonly options: readonly FooterDropdownOption<Value>[]
  readonly open: boolean
  readonly tone: "model" | "reasoning"
  readonly onOpenChange: (open: boolean) => void
  readonly onSelect: (value: Value) => void
}): React.ReactNode {
  const enabled = options.length > 0
  return (
    <Select
      value={value ?? undefined}
      open={open}
      onOpenChange={onOpenChange}
      onValueChange={(nextValue) => onSelect(nextValue as Value)}
      disabled={!enabled}
    >
      <SelectTrigger
        aria-label={`${tone === "model" ? "Model" : "Thinking level"}: ${label}`}
        variant="inline"
        showIcon={false}
        className={`h-auto min-w-0 px-1.5 py-1 text-[14px] leading-5 ${
          tone === "reasoning"
            ? "text-violet-700 enabled:hover:bg-slate-100 dark:text-violet-400 dark:enabled:hover:bg-slate-750"
            : "text-slate-700 enabled:hover:bg-slate-100 dark:text-slate-300 dark:enabled:hover:bg-slate-750"
        }`}
      >
        <SelectValue>{label}</SelectValue>
      </SelectTrigger>
      <SelectContent
        side="top"
        sideOffset={9}
        align="start"
        className={`max-h-[min(320px,calc(100vh-48px))] border border-slate-300 p-1.5 dark:border-slate-600 dark:bg-slate-750 ${
          tone === "model" ? "w-max min-w-[260px]" : "w-[168px]"
        }`}
      >
        {options.map((option) => (
          <SelectItem
            key={option.value}
            value={option.value}
            className={
              tone === "reasoning"
                ? "text-[12px] leading-[1.35] data-selected:bg-violet-200 data-selected:text-violet-700 dark:data-selected:bg-violet-700 dark:data-selected:text-violet-200"
                : "text-[12px] leading-[1.35] data-selected:bg-blue-200 data-selected:text-slate-900 dark:data-selected:bg-blue-700 dark:data-selected:text-slate-50"
            }
          >
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export function FooterBar({
  context,
  tokenCap,
  bashMode,
  model,
  thinkingLevel,
  memoryLabel,
  nextEscWillKillAll,
  transcriptMode,
  modelOptions = [],
  selectedModelId,
  onModelSelect,
  onMemoryClick,
  thinkingOptions = [],
  thinkingEffort,
  onThinkingSelect,
}: FooterBarProps): React.ReactNode {
  const [openDropdown, setOpenDropdown] = useState<"model" | "reasoning" | null>(null)
  return (
    <div className="flex min-h-7 shrink-0 items-center bg-transparent pr-9 font-sans">
      <div className="flex min-w-0 flex-wrap items-center gap-1">
        {bashMode && (
          <span className="text-[11px] text-orange-700 dark:text-orange-500 shrink-0">
            Bash mode
          </span>
        )}
        {nextEscWillKillAll && (
          <span className="text-[11px] text-orange-700 dark:text-orange-500 shrink-0">
            Press Esc again to interrupt all workers
          </span>
        )}
        {transcriptMode && (
          <span className="font-sans text-[11px] text-blue-700 dark:text-blue-500 bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-750 rounded-[4px] [padding:1px_6px]">
            Transcript Mode
          </span>
        )}

        {!bashMode && model && (
          <FooterDropdown
            label={model}
            value={selectedModelId ?? null}
            options={onModelSelect ? modelOptions : []}
            open={openDropdown === "model"}
            tone="model"
            onOpenChange={(open) => setOpenDropdown(open ? "model" : null)}
            onSelect={(providerModelId) => onModelSelect?.(providerModelId)}
          />
        )}
        {!bashMode && thinkingLevel && (
          <FooterDropdown
            label={thinkingLevel}
            value={thinkingEffort ?? null}
            options={onThinkingSelect ? thinkingOptions : []}
            open={openDropdown === "reasoning"}
            tone="reasoning"
            onOpenChange={(open) => setOpenDropdown(open ? "reasoning" : null)}
            onSelect={(effort) => onThinkingSelect?.(effort)}
          />
        )}
        {!bashMode && (
          <>
            {memoryLabel && (
              <Button
                type="button"
                onClick={onMemoryClick}
                disabled={!onMemoryClick}
                variant="ghost"
                className="h-auto whitespace-nowrap px-1.5 py-1 text-[12px] leading-none text-slate-500"
              >
                {memoryLabel}
              </Button>
            )}
            <span className="inline-flex px-1.5 py-1">
              <ContextUsageIndicator
                context={context}
                tokenCap={tokenCap}
                size={18}
                strokeWidth={2}
                tooltip="popover"
                tooltipPlacement="above-center"
              />
            </span>
          </>
        )}
      </div>
    </div>
  )
}
