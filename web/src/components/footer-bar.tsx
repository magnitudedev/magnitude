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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { CaretDownIcon, RocketLaunchIcon } from "@phosphor-icons/react"

export interface FooterModelOption {
  readonly value: ProviderModelId
  readonly label: string
  readonly thinkingOptions: readonly ReasoningEffortOption[]
  readonly defaultThinkingEffort: ReasoningEffort
}

export type FooterModelOptionsState =
  | { readonly _tag: "Loading"; readonly options: readonly FooterModelOption[] }
  | { readonly _tag: "Ready"; readonly options: readonly FooterModelOption[] }
  | { readonly _tag: "Degraded"; readonly options: readonly FooterModelOption[] }
  | { readonly _tag: "Failed"; readonly options: readonly FooterModelOption[] }

export interface FooterBarProps {
  /** Context usage info from timeline */
  context: ContextUsageDisplay | null
  /** Whether the root chat has sent or restored at least one message. */
  showContext?: boolean
  /** Token cap (max context window tokens), if known */
  tokenCap?: number | null
  /** Bash mode active */
  bashMode?: boolean
  /** Current model label */
  model?: string | null
  /** Thinking level label (e.g. "High", "Medium") */
  thinkingLevel?: string | null
  /** Next Esc will kill all workers */
  nextEscWillKillAll?: boolean
  /** Transcript mode active */
  transcriptMode?: boolean
  /** Authoritative availability of installed models for primary-slot selection. */
  modelOptionsState?: FooterModelOptionsState
  /** Currently selected provider-model identity. */
  selectedModelId?: ProviderModelId | null
  /** Atomically applies an installed model and reasoning effort to the primary model slot. */
  onSelectionCommit?: (
    providerModelId: ProviderModelId,
    effort: ReasoningEffort,
  ) => void
  /** Reasoning efforts supported by the selected model. */
  thinkingOptions?: readonly ReasoningEffortOption[]
  /** Currently selected reasoning effort. */
  thinkingEffort?: ReasoningEffort | null
  /** Applies a reasoning effort to the currently selected primary model. */
  onThinkingSelect?: (effort: ReasoningEffort) => void
}

function ModelThinkingMenu({
  modelLabel,
  thinkingLabel,
  optionsState,
  selectedModelId,
  thinkingEffort,
  currentThinkingOptions,
  onSelectionCommit,
  onThinkingSelect,
}: {
  readonly modelLabel: string
  readonly thinkingLabel: string | null
  readonly optionsState: FooterModelOptionsState
  readonly selectedModelId: ProviderModelId | null
  readonly thinkingEffort: ReasoningEffort | null
  readonly currentThinkingOptions: readonly ReasoningEffortOption[]
  readonly onSelectionCommit?: (providerModelId: ProviderModelId, effort: ReasoningEffort) => void
  readonly onThinkingSelect?: (effort: ReasoningEffort) => void
}): React.ReactNode {
  const [open, setOpen] = useState(false)
  const [modelOpen, setModelOpen] = useState(false)
  const [thinkingOpen, setThinkingOpen] = useState(false)
  const [pendingModelId, setPendingModelId] = useState<ProviderModelId | null>(null)
  const draftModelId = pendingModelId ?? selectedModelId
  const draftModel = optionsState.options.find((option) => option.value === draftModelId) ?? null
  const draftThinkingOptions = draftModel?.thinkingOptions
    ?? (draftModelId === selectedModelId ? currentThinkingOptions : [])
  const draftThinkingLabel = draftModelId === selectedModelId
    ? thinkingLabel
      ?? draftThinkingOptions.find((option) => option.value === thinkingEffort)?.label
      ?? "None"
    : "Choose"
  const triggerModelLabel = selectedModelId === null
    ? optionsState._tag === "Loading"
      ? "Loading models…"
      : optionsState._tag === "Failed"
      ? "Models unavailable"
      : modelLabel
    : modelLabel
  const hasAvailabilityNotice = optionsState._tag !== "Ready"
  const canChooseModel = optionsState.options.length > 0
  const canChooseThinking = selectedModelId !== null && currentThinkingOptions.length > 0
  const menuAvailable = canChooseModel || canChooseThinking || hasAvailabilityNotice
  const availabilityLabel =
    optionsState._tag === "Loading"
      ? "Loading models…"
      : optionsState._tag === "Degraded"
      ? "Some installed models are unavailable."
      : optionsState._tag === "Failed"
      ? optionsState.options.length > 0
        ? "Some models could not be loaded."
        : "Unable to load models."
      : null

  const resetDraft = () => {
    setPendingModelId(null)
    setModelOpen(false)
    setThinkingOpen(false)
  }
  const chooseModel = (providerModelId: ProviderModelId) => {
    const option = optionsState.options.find((candidate) => candidate.value === providerModelId)
    if (option === undefined) return
    setPendingModelId(providerModelId)
    setModelOpen(false)
    if (option.thinkingOptions.length === 0) {
      onSelectionCommit?.(providerModelId, option.defaultThinkingEffort)
      setOpen(false)
      return
    }
    setThinkingOpen(true)
  }
  const chooseThinking = (effort: ReasoningEffort) => {
    if (draftModelId === null) return
    if (draftModelId === selectedModelId) onThinkingSelect?.(effort)
    else onSelectionCommit?.(draftModelId, effort)
    setOpen(false)
  }

  if (!menuAvailable && selectedModelId !== null) {
    return (
      <div
        className="inline-flex h-auto min-w-0 items-center gap-1.5 px-1.5 py-1 text-[14px] font-medium leading-5 text-slate-900 dark:text-slate-50"
        aria-label={`Model: ${triggerModelLabel}${thinkingLabel === null ? "" : `. Thinking level: ${thinkingLabel}`}`}
      >
        <RocketLaunchIcon className="size-3.5 shrink-0 text-slate-500 dark:text-slate-400" aria-hidden="true" />
        <span className="min-w-0 truncate">{triggerModelLabel}</span>
        {thinkingLabel !== null ? (
          <span className="shrink-0 text-slate-500 dark:text-slate-400">{thinkingLabel}</span>
        ) : null}
      </div>
    )
  }

  return (
    <DropdownMenu
      open={open}
      onOpenChange={(nextOpen) => {
        setOpen(nextOpen)
        if (!nextOpen) resetDraft()
      }}
      disabled={!menuAvailable}
    >
      <DropdownMenuTrigger
        render={(
          <Button
            type="button"
            variant="unstyled"
            size="unstyled"
            disabled={!menuAvailable}
            className="inline-flex h-auto min-w-0 items-center gap-1.5 rounded-md px-1.5 py-1 text-[14px] font-medium leading-5 text-slate-900 outline-none transition-colors hover:bg-slate-150 focus-visible:bg-slate-150 aria-expanded:bg-slate-150 dark:text-slate-50 dark:hover:bg-slate-750 dark:focus-visible:bg-slate-750 dark:aria-expanded:bg-slate-750"
            aria-label={`Model: ${triggerModelLabel}${thinkingLabel === null ? "" : `. Thinking level: ${thinkingLabel}`}`}
            aria-busy={optionsState._tag === "Loading" || undefined}
          >
            {optionsState._tag === "Loading" && selectedModelId === null ? (
              <Spinner className="size-3.5 shrink-0 text-slate-500" aria-hidden="true" />
            ) : (
              <RocketLaunchIcon className="size-3.5 shrink-0 text-slate-500 dark:text-slate-400" aria-hidden="true" />
            )}
            <span className="min-w-0 truncate">{triggerModelLabel}</span>
            {thinkingLabel !== null ? (
              <span className="shrink-0 text-slate-500 dark:text-slate-400">{thinkingLabel}</span>
            ) : null}
            <CaretDownIcon className="size-3.5 shrink-0 text-slate-500 dark:text-slate-400" aria-hidden="true" />
          </Button>
        )}
      />
      <DropdownMenuContent
        side="top"
        sideOffset={9}
        align="start"
        className="w-[296px] p-1.5"
      >
        <DropdownMenuSub
          open={modelOpen}
          onOpenChange={(nextOpen) => {
            setModelOpen(nextOpen)
            if (nextOpen) setThinkingOpen(false)
          }}
        >
          <DropdownMenuSubTrigger
            openOnHover
            disabled={optionsState.options.length === 0}
            className="grid grid-cols-[88px_minmax(0,1fr)_16px] text-[13px] [&>svg:last-child]:ml-0"
          >
            <span>Model</span>
            <span className="min-w-0 truncate text-left text-slate-500 dark:text-slate-400">
              {draftModel?.label ?? triggerModelLabel}
            </span>
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent side="right" className="max-h-[320px] min-w-[280px] overflow-y-auto p-1.5">
            <DropdownMenuRadioGroup value={draftModelId} onValueChange={(value) => chooseModel(value as ProviderModelId)}>
              {optionsState.options.map((option) => (
                <DropdownMenuRadioItem key={option.value} value={option.value} closeOnClick={false}>
                  <span className="truncate">{option.label}</span>
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        {draftThinkingOptions.length > 0 ? (
          <DropdownMenuSub
            open={thinkingOpen}
            onOpenChange={(nextOpen) => {
              setThinkingOpen(nextOpen)
              if (nextOpen) setModelOpen(false)
            }}
          >
            <DropdownMenuSubTrigger
              openOnHover
              className="grid grid-cols-[88px_minmax(0,1fr)_16px] text-[13px] [&>svg:last-child]:ml-0"
            >
              <span>Thinking</span>
              <span className="min-w-0 truncate text-left text-slate-500 dark:text-slate-400">{draftThinkingLabel}</span>
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent side="right" className="min-w-[168px] p-1.5">
              <DropdownMenuRadioGroup value={draftModelId === selectedModelId ? thinkingEffort : null} onValueChange={(value) => chooseThinking(value as ReasoningEffort)}>
                {draftThinkingOptions.map((option) => (
                  <DropdownMenuRadioItem key={option.value} value={option.value} closeOnClick>
                    {option.label}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        ) : null}
        {availabilityLabel ? (
          <div
            role="status"
            className="flex items-center gap-2 px-2 py-2 text-[12px] leading-[1.35] text-slate-500 dark:text-slate-300"
          >
            {optionsState._tag === "Loading" ? (
              <Spinner className="size-3.5 shrink-0" aria-hidden="true" />
            ) : null}
            <span>{availabilityLabel}</span>
          </div>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function FooterBar({
  context,
  showContext = false,
  tokenCap,
  bashMode,
  model,
  thinkingLevel,
  nextEscWillKillAll,
  transcriptMode,
  modelOptionsState = { _tag: "Ready", options: [] },
  selectedModelId,
  onSelectionCommit,
  thinkingOptions = [],
  thinkingEffort,
  onThinkingSelect,
}: FooterBarProps): React.ReactNode {
  return (
    <div className="flex min-h-7 shrink-0 items-center bg-transparent pr-12 font-sans">
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
      </div>
      {!bashMode && (
        <div className="ml-auto flex min-w-0 items-center gap-1">
          {showContext ? (
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
          ) : null}
          {model ? (
            <ModelThinkingMenu
              modelLabel={model}
              thinkingLabel={thinkingLevel ?? null}
              optionsState={modelOptionsState}
              selectedModelId={selectedModelId ?? null}
              thinkingEffort={thinkingEffort ?? null}
              currentThinkingOptions={thinkingOptions}
              onSelectionCommit={onSelectionCommit}
              onThinkingSelect={onThinkingSelect}
            />
          ) : null}
        </div>
      )}
    </div>
  )
}
