/**
 * FooterBar — spec §9.7
 *
 * Bottom dock status row matching the CLI's model-runtime footer.
 */
import { useState } from "react"
import { LoaderCircle } from "lucide-react"
import {
  formatCwdForDisplay,
  type ReasoningEffortOption,
} from "@magnitudedev/client-common"
import type { ContextUsageDisplay, ReasoningEffort } from "@magnitudedev/sdk"
import { formatFooterContextUsage } from "./local-inference-format"
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
  /** Click handler for model name (opens the Models settings tab) */
  onModelClick?: () => void
  /** Click handler for resident memory (opens Hardware) */
  onMemoryClick?: () => void
  /** Reasoning efforts supported by the selected model. */
  thinkingOptions?: readonly ReasoningEffortOption[]
  /** Currently selected reasoning effort. */
  thinkingEffort?: ReasoningEffort | null
  /** Applies a reasoning effort to the primary model slot. */
  onThinkingSelect?: (effort: ReasoningEffort) => void
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
  onModelClick,
  onMemoryClick,
  thinkingOptions = [],
  thinkingEffort,
  onThinkingSelect,
}: FooterBarProps): React.ReactNode {
  const [thinkingOpen, setThinkingOpen] = useState(false)
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
          <button
            type="button"
            onClick={onModelClick}
            disabled={!onModelClick}
            className="cursor-pointer border-0 bg-transparent p-0 font-sans text-[13px] leading-[inherit] whitespace-nowrap text-slate-900 no-underline disabled:cursor-default enabled:hover:text-blue-700 enabled:hover:underline enabled:hover:underline-offset-2 dark:text-slate-200 dark:enabled:hover:text-blue-500"
          >
            {model}
          </button>
        )}
        {!bashMode && thinkingLevel && (
          <button
            type="button"
            onClick={() => {
              if (thinkingOptions.length > 0 && onThinkingSelect) {
                setThinkingOpen((open) => !open)
              }
            }}
            aria-expanded={
              thinkingOptions.length > 0 ? thinkingOpen : undefined
            }
            aria-label={`Reasoning effort: ${thinkingLevel}`}
            disabled={thinkingOptions.length === 0 || !onThinkingSelect}
            className="p-0 border-0 bg-transparent text-slate-900 dark:text-slate-200 font-sans text-[13px] leading-[inherit] no-underline whitespace-nowrap cursor-pointer disabled:cursor-default enabled:hover:text-blue-700 dark:enabled:hover:text-blue-500 enabled:hover:underline enabled:hover:underline-offset-2 !text-violet-700 dark:!text-violet-500"
          >
            {thinkingLevel}
          </button>
        )}
        {!bashMode && thinkingOpen && (
          <div
            role="listbox"
            aria-label="Reasoning effort"
            className="flex items-center [gap:10px] min-w-0"
          >
            <span aria-hidden="true" className="text-slate-500">
              &gt;
            </span>
            {thinkingOptions.map((option) => {
              const selected = option.value === thinkingEffort
              return (
                <button
                  key={option.value}
                  type="button"
                  role="option"
                  aria-selected={selected}
                  onClick={() => {
                    if (!selected) onThinkingSelect?.(option.value)
                    setThinkingOpen(false)
                  }}
                  className={`${
                    selected
                      ? "text-violet-700 dark:text-violet-500"
                      : "text-slate-600 dark:text-slate-400"
                  } ${
                    selected
                      ? "[text-decoration:underline]"
                      : "[text-decoration:none]"
                  } border-0 bg-transparent p-0 font-sans text-[13px] whitespace-nowrap cursor-pointer`}
                >
                  {option.label}
                </button>
              )
            })}
          </div>
        )}
        {!bashMode && !thinkingOpen && (
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
