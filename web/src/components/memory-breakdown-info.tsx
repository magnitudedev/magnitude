import { formatMemorySize, localModelMemoryParts, localModelUnservedContext } from "@magnitudedev/client-common"
import { Option } from "effect"
import type { LocalModelAssessment } from "@magnitudedev/sdk"
import { InfoIcon } from "@phosphor-icons/react"
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip"

type FitsAssessment = Extract<LocalModelAssessment, { readonly _tag: "Fits" }>

/** A fitting model's info icon beside its memory, opening what that memory holds. */
export function MemoryBreakdownInfo({ assessment }: { assessment: FitsAssessment }) {
  return <Tooltip>
    <TooltipTrigger render={<button type="button" aria-label="Memory breakdown" className="inline-flex shrink-0 rounded-sm text-slate-400 hover:text-slate-600 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-500 dark:hover:text-slate-300"><InfoIcon aria-hidden="true" className="size-3" /></button>} />
    <TooltipContent className="max-w-md flex-col items-stretch gap-2 py-2.5"><MemoryBreakdownSummary assessment={assessment} /></TooltipContent>
  </Tooltip>
}

/** What a fitting model's memory holds: weights, context at the length it is served at, and overhead. */
export function MemoryBreakdownSummary({ assessment }: { assessment: FitsAssessment }) {
  const totalRequiredBytes = assessment.memory.totalRequiredBytes
  const colors = ["bg-slate-200", "bg-blue-400", "bg-slate-500"]
  const parts = localModelMemoryParts(assessment).map((part, index) => ({ ...part, color: colors[index] }))
  const unserved = localModelUnservedContext(assessment)
  return <>
    <span>This model takes up <span className="tabular-nums">{formatMemorySize(totalRequiredBytes)}</span> when loaded</span>
    <span aria-hidden="true" className="flex h-1.5 gap-px overflow-hidden rounded-full">
      {parts.map(part => <span key={part.label} className={part.color} style={{ width: `${totalRequiredBytes > 0 ? part.bytes / totalRequiredBytes * 100 : 0}%` }} />)}
    </span>
    <span className="flex flex-wrap items-center gap-x-1.5 whitespace-nowrap text-slate-300">
      {parts.map((part, index) => <span key={part.label} className="inline-flex items-center gap-1">
        {index > 0 && <span aria-hidden="true" className="pr-0.5 text-slate-500">·</span>}
        <span aria-hidden="true" className={`size-1.5 rounded-full ${part.color}`} />
        {part.label} <span className="tabular-nums text-white dark:text-slate-100">{formatMemorySize(part.bytes)}</span>
      </span>)}
      {Option.isSome(unserved) && <span className="text-slate-400">({unserved.value})</span>}
    </span>
  </>
}
