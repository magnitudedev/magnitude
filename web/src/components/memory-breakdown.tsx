import { Option } from "effect"
import type { ModelInstanceAllocation } from "@magnitudedev/sdk"
import { formatMemorySize } from "@magnitudedev/client-common"
import { SkeletonLine } from "./page-skeletons"
import { Skeleton } from "./ui/skeleton"

export function MemoryFigures({ allocation, reading = false, pendingLabel = "Available once loaded" }: {
  readonly allocation: Option.Option<ModelInstanceAllocation> | null
  readonly reading?: boolean
  readonly pendingLabel?: string
}) {
  const domains = allocation === null ? [] : Option.match(allocation, { onNone: () => [], onSome: value => value.memoryDomains })
  const segments = [
    { label: "Weights", bytes: domains.reduce((sum, domain) => sum + domain.modelBytes + domain.auxiliaryBytes, 0), color: "weights" },
    { label: "KV cache", bytes: domains.reduce((sum, domain) => sum + domain.contextBytes, 0), color: "cache" },
    { label: "Runtime", bytes: domains.reduce((sum, domain) => sum + domain.computeBytes, 0), color: "runtime" },
  ]
  const total = segments.reduce((sum, segment) => sum + segment.bytes, 0)
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
  </div>
}
