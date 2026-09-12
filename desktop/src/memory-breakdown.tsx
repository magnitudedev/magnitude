import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import {
  activeLocalModel,
  deriveHardwareMemoryView,
  formatMemorySize,
  useLocalInferenceHardware,
  useLocalModels,
  type HardwareMemoryDomainView,
} from "@magnitudedev/client-common"

export function MemoryDomain({ domain }: { readonly domain: HardwareMemoryDomainView }) {
  const segments = [
    { label: "Model weights", bytes: domain.modelBytes, color: "bg-blue-600 dark:bg-blue-400" },
    { label: "KV cache", bytes: domain.kvCacheBytes, color: "bg-blue-300 dark:bg-blue-700" },
    { label: "Engine overhead", bytes: domain.overheadBytes, color: "bg-slate-600 dark:bg-slate-300" },
    { label: "System & apps", bytes: domain.systemAndAppsBytes, color: "bg-slate-400 dark:bg-slate-600" },
    { label: "Free", bytes: domain.freeBytes, color: "bg-slate-100 dark:bg-slate-800" },
  ]
  const complete = segments.every(segment => segment.bytes !== null)
  return <div className="mt-6" data-memory-domain={domain.id}>
    <div className="flex flex-wrap items-baseline justify-between gap-2">
      <h3 className="text-sm font-medium">{domain.label}</h3>
      <span className="text-xs text-slate-500">{domain.usedBytes === null ? `${formatMemorySize(domain.totalBytes)} total` : `${formatMemorySize(domain.usedBytes)} of ${formatMemorySize(domain.totalBytes)} used`}</span>
    </div>
    {complete && <div className="mt-3 flex h-3 overflow-hidden rounded-full" aria-hidden="true">
      {segments.map(segment => <span key={segment.label} className={segment.color} style={{ width: `${domain.totalBytes > 0 ? (segment.bytes ?? 0) / domain.totalBytes * 100 : 0}%` }} />)}
    </div>}
    <dl className="mt-4 grid grid-cols-2 gap-x-5 gap-y-3 sm:grid-cols-3">
      {segments.map(segment => <div key={segment.label}>
        <dt className="flex items-center gap-2 text-xs text-slate-500"><span className={`size-2 rounded-sm ${segment.color}`} />{segment.label}</dt>
        <dd className="mt-1 text-sm tabular-nums" data-memory-category={segment.label} data-bytes={segment.bytes ?? undefined}>{segment.bytes === null ? "Unavailable" : formatMemorySize(segment.bytes)}</dd>
      </div>)}
    </dl>
    {domain.notice && <p className="mt-3 text-xs text-slate-500">{domain.notice}</p>}
  </div>
}

export function MemoryBreakdown() {
  const hardware = useLocalInferenceHardware()
  const models = useLocalModels()
  if (!Result.isSuccess(hardware) || !Result.isSuccess(models)) return <p className="mt-5 text-sm text-slate-500">{Result.isFailure(hardware) || Result.isFailure(models) ? "Memory breakdown unavailable." : "Reading memory breakdown…"}</p>
  const active = Option.getOrNull(activeLocalModel(models.value))
  if (active && active.residency._tag !== "Ready" && !(active.residency._tag === "Stopping" && active.residency.allocation._tag === "Resident")) return <p className="mt-5 text-sm text-slate-500">Model allocations will be available when loading completes.</p>
  const allocation = active?.residency._tag === "Ready" ? Option.some(active.residency.allocation)
    : active?.residency._tag === "Stopping" && active.residency.allocation._tag === "Resident" ? Option.some(active.residency.allocation.allocation)
    : Option.none()
  const memory = deriveHardwareMemoryView(hardware.value, allocation)
  return <div className="mt-6 border-t border-slate-200 pt-1 dark:border-slate-750">
    {memory.domains.map(domain => <MemoryDomain key={domain.id} domain={domain} />)}
    <p className="mt-5 text-xs leading-relaxed text-slate-500">Model weights, allocated KV cache and engine buffers come from the inference engine. System & apps includes other memory in each domain. These allocation figures use different accounting from the application footprint above.</p>
  </div>
}
