import { Option } from "effect"
import type { LocalInferenceHardware, ModelInstanceAllocation } from "@magnitudedev/sdk"
import { formatMemorySize, modelServingFacts } from "@magnitudedev/client-common"
import { ModelFacts } from "./model-facts"
import { SkeletonLine } from "./page-skeletons"
import { Skeleton } from "./ui/skeleton"

export function MemoryFigures({ allocation, hardware, reading = false, pendingLabel = "Available once loaded" }: {
  readonly allocation: Option.Option<ModelInstanceAllocation> | null
  /** The machine its memory domains are on; an expert bank's memory is named once it is known. */
  readonly hardware: Option.Option<LocalInferenceHardware>
  readonly reading?: boolean
  readonly pendingLabel?: string
}) {
  const domains = allocation === null ? [] : Option.match(allocation, { onNone: () => [], onSome: value => value.memoryDomains })
  const segments = [
    { label: "Weights", bytes: domains.reduce((sum, domain) => sum + domain.modelBytes + domain.auxiliaryBytes, 0), color: "weights" },
    { label: "Context", bytes: domains.reduce((sum, domain) => sum + domain.contextBytes, 0), color: "cache" },
    { label: "Overhead", bytes: domains.reduce((sum, domain) => sum + domain.computeBytes, 0), color: "runtime" },
  ]
  const total = segments.reduce((sum, segment) => sum + segment.bytes, 0)
  const serving = allocation === null ? null : Option.getOrNull(allocation)
  return <div className="settings-memory">
    <div className="settings-memory-bar" aria-hidden="true">
      {reading ? <Skeleton className="h-full w-full" /> : allocation !== null && segments.map(segment => <span key={segment.label} className={`settings-memory-segment ${segment.color}`} style={{ width: `${total > 0 ? segment.bytes / total * 100 : 0}%` }} />)}
    </div>
    <div className="settings-memory-legend">
      <span className="settings-memory-total"><span>Model memory</span>{allocation === null
        ? reading ? <SkeletonLine className="h-4 w-16" /> : <span className="settings-memory-pending">{pendingLabel}</span>
        : <strong data-memory-bytes={total}>{formatMemorySize(total)}</strong>}</span>
      {allocation !== null && <div className="settings-memory-parts">{segments.map(segment => <span key={segment.label} className="settings-memory-item">
        <span aria-hidden="true" className={`settings-memory-key ${segment.color}`} /><span>{segment.label}</span>
        <strong data-memory-category={segment.label} data-bytes={segment.bytes}>{formatMemorySize(segment.bytes)}</strong>
      </span>)}</div>}
    </div>
    {serving !== null && <ModelFacts facts={modelServingFacts(serving, hardware)} label="Serving"
      className="mt-5 grid grid-cols-1 gap-3 border-t border-slate-200 pt-4 md:grid-cols-2 dark:border-slate-750" />}
  </div>
}
