/**
 * FooterBar — spec §9.7
 *
 * Compact runtime controls rendered inside the bottom of the composer.
 */
import { useId, useRef, useState } from "react"
import { type ReasoningEffortOption } from "@magnitudedev/client-common"
import type {
  ContextUsageDisplay,
  ProviderModelId,
  ReasoningEffort,
} from "@magnitudedev/sdk"
import { ContextUsageIndicator } from "./context-usage-indicator"

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
  const menuId = useId()
  const triggerRef = useRef<HTMLButtonElement>(null)
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([])
  const enabled = options.length > 0
  const selectedIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value)
  )

  const focusOption = (index: number): void => {
    optionRefs.current[index]?.focus()
  }
  const openAndFocus = (index: number): void => {
    onOpenChange(true)
    requestAnimationFrame(() => focusOption(index))
  }
  const closeAndFocusTrigger = (): void => {
    onOpenChange(false)
    triggerRef.current?.focus()
  }

  return (
    <span
      className="relative inline-flex min-w-0"
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) {
          onOpenChange(false)
        }
      }}
    >
      <button
        ref={triggerRef}
        type="button"
        aria-label={`${
          tone === "model" ? "Model" : "Thinking level"
        }: ${label}`}
        aria-controls={open ? menuId : undefined}
        aria-expanded={open}
        aria-haspopup="listbox"
        disabled={!enabled}
        onClick={() => onOpenChange(!open)}
        onKeyDown={(event) => {
          if (!enabled) return
          if (open) {
            if (event.key === "Escape") {
              event.preventDefault()
              onOpenChange(false)
            } else if (event.key === "ArrowUp") {
              event.preventDefault()
              focusOption(options.length - 1)
            } else if (event.key === "ArrowDown") {
              event.preventDefault()
              focusOption(selectedIndex)
            }
            return
          }
          if (event.key === "ArrowUp") {
            event.preventDefault()
            openAndFocus(options.length - 1)
          } else if (event.key === "ArrowDown") {
            event.preventDefault()
            openAndFocus(selectedIndex)
          }
        }}
        className={`inline-flex min-w-0 cursor-pointer items-center rounded px-1.5 py-1 border-0 bg-transparent font-sans text-[14px] leading-5 whitespace-nowrap no-underline transition-colors duration-100 disabled:cursor-default ${
          tone === "reasoning"
            ? "text-violet-700 enabled:hover:bg-slate-100 dark:text-violet-400 dark:enabled:hover:bg-slate-750"
            : "text-slate-700 enabled:hover:bg-slate-100 dark:text-slate-300 dark:enabled:hover:bg-slate-750"
        }`}
      >
        <span className="overflow-hidden text-ellipsis">
          {label}
        </span>
      </button>

      {open && enabled && (
        <span
          id={menuId}
          role="listbox"
          aria-label={tone === "model" ? "Installed models" : "Thinking level"}
          className={`absolute bottom-[calc(100%+9px)] left-0 z-50 box-border flex max-h-[min(320px,calc(100vh-48px))] max-w-[calc(100vw-24px)] flex-col gap-0.5 overflow-y-auto rounded-lg border border-slate-300 bg-white p-1.5 shadow-[0_8px_24px_rgba(0,0,0,.16)] dark:border-slate-600 dark:bg-slate-750 dark:shadow-[0_8px_24px_rgba(0,0,0,.36)] ${
            tone === "model" ? "w-max min-w-[260px]" : "w-[168px]"
          }`}
        >
          {options.map((option, index) => {
            const selected = option.value === value
            return (
              <button
                key={option.value}
                ref={(element) => {
                  optionRefs.current[index] = element
                }}
                type="button"
                role="option"
                aria-selected={selected}
                onClick={() => {
                  if (!selected) onSelect(option.value)
                  closeAndFocusTrigger()
                }}
                onKeyDown={(event) => {
                  if (event.key === "Escape") {
                    event.preventDefault()
                    closeAndFocusTrigger()
                  } else if (event.key === "ArrowDown") {
                    event.preventDefault()
                    focusOption((index + 1) % options.length)
                  } else if (event.key === "ArrowUp") {
                    event.preventDefault()
                    focusOption((index - 1 + options.length) % options.length)
                  } else if (event.key === "Home") {
                    event.preventDefault()
                    focusOption(0)
                  } else if (event.key === "End") {
                    event.preventDefault()
                    focusOption(options.length - 1)
                  }
                }}
                className={`w-full cursor-pointer rounded-md border-0 px-2.5 py-2 text-left font-sans text-[12px] leading-[1.35] ${
                  selected
                    ? tone === "reasoning"
                      ? "bg-violet-200 text-violet-700 dark:bg-violet-700 dark:text-violet-200"
                      : "bg-blue-200 text-slate-900 dark:bg-blue-700 dark:text-slate-50"
                    : "bg-transparent text-slate-700 hover:bg-slate-150 dark:text-slate-200 dark:hover:bg-slate-700"
                }`}
              >
                {option.label}
              </button>
            )
          })}
        </span>
      )}
    </span>
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
  const [openDropdown, setOpenDropdown] = useState<
    "model" | "reasoning" | null
  >(null)
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
              <button
                type="button"
                onClick={onMemoryClick}
                disabled={!onMemoryClick}
                className="cursor-pointer whitespace-nowrap rounded border-0 bg-transparent px-1.5 py-1 font-sans text-[12px] leading-none text-slate-500 no-underline transition-colors duration-100 enabled:hover:bg-slate-100 disabled:cursor-default dark:enabled:hover:bg-slate-750"
              >
                {memoryLabel}
              </button>
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
