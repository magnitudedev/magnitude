import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { MemoryStick } from "lucide-react"
import type { ModelInstanceAllocation } from "@magnitudedev/sdk"
import { activeLocalModel, formatMemorySize, useLocalModels } from "@magnitudedev/client-common"

export function MemoryFigures({ allocation }: { readonly allocation: Option.Option<ModelInstanceAllocation> }) {
  const domains = Option.match(allocation, { onNone: () => [], onSome: value => value.memoryDomains })
  const segments = [
    { label: "Model weights", bytes: domains.reduce((sum, domain) => sum + domain.modelBytes, 0), color: "bg-blue-600 dark:bg-blue-400" },
    { label: "KV cache", bytes: domains.reduce((sum, domain) => sum + domain.contextBytes, 0), color: "bg-blue-300 dark:bg-blue-700" },
    { label: "Overhead", bytes: domains.reduce((sum, domain) => sum + domain.computeBytes + domain.auxiliaryBytes, 0), color: "bg-slate-600 dark:bg-slate-300" },
  ]
  const total = segments.reduce((sum, segment) => sum + segment.bytes, 0)
  return <>
    <p className="mt-4 font-heading text-3xl tabular-nums" data-memory-bytes={total}>{formatMemorySize(total)}</p>
    <div className="mt-5 flex h-3 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800" aria-hidden="true">
      {segments.map(segment => <span key={segment.label} className={segment.color} style={{ width: `${total > 0 ? segment.bytes / total * 100 : 0}%` }} />)}
    </div>
    <dl className="mt-4 grid grid-cols-3 gap-4">
      {segments.map(segment => <div key={segment.label}>
        <dt className="flex items-center gap-2 text-xs text-slate-500"><span className={`size-2 shrink-0 rounded-sm ${segment.color}`} />{segment.label}</dt>
        <dd className="mt-1 text-sm tabular-nums" data-memory-category={segment.label} data-bytes={segment.bytes}>{formatMemorySize(segment.bytes)}</dd>
      </div>)}
    </dl>
  </>
}

function ModelMemory() {
  const models = useLocalModels()
  if (!Result.isSuccess(models)) return <p className="mt-4 text-sm text-slate-500">{Result.isFailure(models) ? "Memory unavailable" : "Reading memory…"}</p>
  const active = Option.getOrNull(activeLocalModel(models.value))
  if (active && active.residency._tag !== "Ready" && !(active.residency._tag === "Stopping" && active.residency.allocation._tag === "Resident")) return <p className="mt-4 text-sm text-slate-500">Loading model…</p>
  const allocation = active?.residency._tag === "Ready" ? Option.some(active.residency.allocation)
    : active?.residency._tag === "Stopping" && active.residency.allocation._tag === "Resident" ? Option.some(active.residency.allocation.allocation)
    : Option.none()
  return <MemoryFigures allocation={allocation} />
}

export function MemoryBreakdown() {
  return <section aria-label="Memory usage" className="rounded-2xl border border-slate-200 bg-white p-6 dark:border-slate-750 dark:bg-slate-850">
    <div className="flex items-center gap-3"><MemoryStick className="size-5 text-blue-600 dark:text-blue-400" /><h2 className="font-heading text-lg">Memory</h2></div>
    <ModelMemory />
  </section>
}
