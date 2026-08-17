/**
 * FooterBar — spec §9.7
 *
 * Bottom dock status row matching the CLI's model-runtime footer.
 */
import { useId, useRef, useState } from "react"
import { ChevronUp, LoaderCircle } from "lucide-react"
import {
  formatCwdForDisplay,
  type ReasoningEffortOption,
} from "@magnitudedev/client-common"
import type {
  ContextUsageDisplay,
  ProviderModelId,
  ReasoningEffort,
} from "@magnitudedev/sdk"
import { formatFooterContextUsage } from "./local-inference-format"

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
  /** Current agent-host working directory */
  cwd?: string | null
  /** Current model label */
  model?: string | null
  /** Current local-model residency presentation. */
  modelResidency?: "ready" | "loading" | "not-ready" | null
  /** Loading percentage, when projected by the selected slot. */
  modelLoadingPercentage?: number | null
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
        className={`group inline-flex min-w-0 cursor-pointer items-center gap-0.5 border-0 bg-transparent p-0 font-sans text-[13px] leading-[inherit] whitespace-nowrap no-underline disabled:cursor-default ${
          tone === "reasoning"
            ? "text-violet-700 enabled:hover:text-violet-600 dark:text-violet-500 dark:enabled:hover:text-violet-400"
            : "text-slate-900 enabled:hover:text-blue-700 dark:text-slate-200 dark:enabled:hover:text-blue-500"
        }`}
      >
        <span className="overflow-hidden text-ellipsis group-enabled:group-hover:underline group-enabled:group-hover:underline-offset-2">
          {label}
        </span>
        {enabled && (
          <ChevronUp
            size={12}
            strokeWidth={1.75}
            aria-hidden="true"
            className={`shrink-0 transition-transform duration-150 ${
              open ? "rotate-180" : ""
            }`}
          />
        )}
      </button>

      {open && enabled && (
        <span
          id={menuId}
          role="listbox"
          aria-label={tone === "model" ? "Installed models" : "Thinking level"}
          className={`absolute bottom-[calc(100%+8px)] left-0 z-50 box-border flex max-h-[min(320px,calc(100vh-48px))] max-w-[calc(100vw-24px)] flex-col gap-0.5 overflow-y-auto rounded-lg border border-slate-300 bg-slate-50 p-1.5 shadow-[0_10px_32px_rgba(0,0,0,.18)] dark:border-slate-750 dark:bg-slate-875 dark:shadow-[0_10px_32px_rgba(0,0,0,.45)] ${
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
                    : "bg-transparent text-slate-700 hover:bg-slate-150 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-slate-100"
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

function ModelResidencyIndicator({
  residency,
  loadingPercentage,
}: {
  readonly residency: NonNullable<FooterBarProps["modelResidency"]>
  readonly loadingPercentage?: number | null
}): React.ReactNode {
  if (residency === "loading") {
    const label =
      loadingPercentage === null || loadingPercentage === undefined
        ? "Model loading"
        : `Model loading · ${loadingPercentage}%`
    return (
      <span
        className="inline-flex w-3 h-4 items-center justify-center shrink-0 font-sans text-xs leading-none [&.ready]:text-green-700 dark:[&.ready]:text-green-500 [&.not-ready]:text-slate-500 [&.loading]:text-orange-700 dark:[&.loading]:text-orange-500 loading"
        aria-label={label}
        title={label}
      >
        <LoaderCircle className="animate-spin" size={12} aria-hidden="true" />
      </span>
    )
  }
  const ready = residency === "ready"
  const label = ready ? "Model ready" : "Model not ready"
  return (
    <span
      className={`inline-flex w-3 h-4 items-center justify-center shrink-0 font-sans text-xs leading-none [&.ready]:text-green-700 dark:[&.ready]:text-green-500 [&.not-ready]:text-slate-500 [&.loading]:text-orange-700 dark:[&.loading]:text-orange-500 ${
        ready ? "ready" : "not-ready"
      }`}
      aria-label={label}
      title={label}
    >
      <span aria-hidden="true">{ready ? "●" : "○"}</span>
    </span>
  )
}
export function FooterBar({
  context,
  tokenCap,
  bashMode,
  cwd,
  model,
  modelResidency,
  modelLoadingPercentage,
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
  const cwdText = cwd
    ? formatCwdForDisplay(cwd, {
        maxLen: 80,
        abbreviateHome: true,
      })
    : ""
  const contextLabel = formatFooterContextUsage(context, tokenCap)
  return (
    <div className="flex min-h-[26px] shrink-0 items-center justify-between bg-transparent px-0.5 font-sans">
      {/* Model controls mirror the CLI footer: model and reasoning on the
          left, environment on the right. */}
      <div className="flex items-center flex-wrap [gap:8px] min-w-0">
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

        {!bashMode && modelResidency && (
          <ModelResidencyIndicator
            residency={modelResidency}
            loadingPercentage={modelLoadingPercentage}
          />
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
                className="p-0 border-0 bg-transparent text-slate-900 dark:text-slate-200 font-sans text-[13px] leading-[inherit] no-underline whitespace-nowrap cursor-pointer disabled:cursor-default enabled:hover:text-blue-700 dark:enabled:hover:text-blue-500 enabled:hover:underline enabled:hover:underline-offset-2 !text-slate-500"
              >
                {memoryLabel}
              </button>
            )}
            <span
              className="!text-slate-500 font-sans text-xs leading-[normal] tabular-nums whitespace-nowrap data-[compacting=true]:animate-context-pulse"
              data-compacting={context?.isCompacting ?? false}
              title="Context usage"
            >
              {contextLabel}
            </span>
          </>
        )}
      </div>

      <div className="flex items-center justify-end min-w-0 [max-width:40%] [margin-left:12px]">
        {cwdText && (
          <span
            title={cwd ?? undefined}
            className="text-slate-600 dark:text-slate-400 font-mono text-[13px] min-w-0 overflow-hidden text-ellipsis whitespace-nowrap"
          >
            {cwdText}
          </span>
        )}
      </div>
    </div>
  )
}
