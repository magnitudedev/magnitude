import { useState, type ReactNode } from "react"
import { Skeleton } from "../ui/skeleton"
import { ArrowSquareOutIcon, CheckIcon, CubeIcon, GraphicsCardIcon, HardDrivesIcon, MemoryIcon } from "@phosphor-icons/react"
import { formatContext, formatMemoryCapacity, formatStorageSize } from "@magnitudedev/client-common"
import { cn } from "@/lib/utils"
import { modelLabs, type ModelLab } from "../model-logo"
import type { MachineView, PlacementTier, QuantView } from "./view"

export const cardClass = "rounded-xl border border-slate-200 bg-white dark:border-slate-750 dark:bg-slate-850"

export const formatSpeed = (speed: number) => speed < 1 ? "<1 tok/s" : `${Math.round(speed)} tok/s`

export const quantSource = (quant: Pick<QuantView, "publisher" | "name">) => quant.publisher ? `${quant.publisher} ${quant.name}` : quant.name

export function LabMark({ lab, className = "size-8" }: { lab: ModelLab | undefined; className?: string }) {
  return <span className={cn("flex shrink-0 items-center justify-center", className)}>
    {lab ? <img src={lab.src} alt="" className={cn("size-full object-contain", lab.theme)} /> : <CubeIcon className="size-full text-slate-400" />}
  </span>
}

export const labByName = (name: string | null) => name === null ? undefined : modelLabs.find(lab => lab.name === name)

/** What to say for a quant with no speed yet: it is still being measured, or it does not fit. */
export const fitLabel = (quant: QuantView) => quant.measuring ? "Measuring…" : quant.fits ? "Speed unavailable" : "Doesn’t fit"

/** An estimate, so it stays small and muted. */
export function Speed({ quant }: { quant: QuantView }) {
  if (!quant.fits || quant.speed === null) return <span className="text-xs text-slate-500">{fitLabel(quant)}</span>
  return <span title={`Estimated decode speed in a long agent conversation.${quant.phrase ? ` ${quant.phrase}.` : ""}`} className="whitespace-nowrap text-xs tabular-nums text-slate-500">~{formatSpeed(quant.speed)}</span>
}

/** Context, marked when this machine serves less than the model supports. */
export function ContextText({ quant, limit, modelName }: { quant: QuantView; limit: number | null; modelName: string }) {
  if (quant.servedContext === null) return null
  if (limit !== null && quant.servedContext < limit) {
    return <span title={`${modelName} supports ${formatContext(limit)}; ${formatContext(quant.servedContext)} is what fits on your machine.`} className="whitespace-nowrap text-orange-700 dark:text-orange-300">{formatContext(quant.servedContext)} context</span>
  }
  return <span className="whitespace-nowrap">{formatContext(quant.servedContext)} context</span>
}

export function RepositoryLink({ repository, children }: { repository: string; children?: ReactNode }) {
  return <a href={`https://huggingface.co/${repository}`} target="_blank" rel="noreferrer" title={repository} onClick={event => event.stopPropagation()}
    className="inline-flex min-w-0 items-center gap-0.5 hover:text-blue-700 hover:underline dark:hover:text-blue-400">
    <span className="truncate">{children ?? repository}</span><ArrowSquareOutIcon className="size-3 shrink-0" />
  </a>
}

const tierColor: Record<PlacementTier["kind"], string> = {
  Device: "bg-blue-600 dark:bg-blue-400",
  Unified: "bg-blue-600 dark:bg-blue-400",
  Host: "bg-blue-300 dark:bg-blue-700",
  Disk: "bg-slate-300 dark:bg-slate-600",
}
const tierPhrase: Record<PlacementTier["kind"], string> = {
  Device: "held in GPU memory",
  Unified: "held in memory",
  Host: "held in RAM",
  Disk: "read from SSD as needed",
}

/** How a quant runs: what stays in fast memory, and what is read from the next tier while it runs. */
export function PlacementBar({ quant }: { quant: QuantView }) {
  const tiers = quant.placement.filter(tier => tier.bytes > 0)
  const total = tiers.reduce((sum, tier) => sum + tier.bytes, 0)
  if (total === 0) return null
  return <div>
    <p className="mb-2 text-xs font-medium text-slate-700 dark:text-slate-300">How {quant.bits}-bit runs on your machine</p>
    <div className="flex gap-0.5">
      {tiers.map(tier => <div key={tier.kind} className={cn("h-2 min-w-1 rounded-full", tierColor[tier.kind])} style={{ width: `${tier.bytes / total * 100}%` }} />)}
    </div>
    <p className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-slate-500">
      {tiers.map(tier => <span key={tier.kind} className="inline-flex items-center gap-1.5 whitespace-nowrap"><span className={cn("size-2 rounded-full", tierColor[tier.kind])} /><span className="font-medium text-slate-700 dark:text-slate-300">{formatStorageSize(tier.bytes)}</span>{tierPhrase[tier.kind]}</span>)}
    </p>
  </div>
}

/** Rungs side by side, smallest to largest: bits, source, then the two numbers that decide it. */
export function QuantLadder({ quants, selected, recommended, owned, limit, onSelect }: {
  quants: readonly QuantView[]; selected: string | null; recommended: string | null; owned: ReadonlySet<string>; limit: number | null; onSelect: (key: string) => void
}) {
  return <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${quants.length}, minmax(0, 1fr))` }}>
    {quants.map(quant => {
      const active = selected === quant.key
      return <button key={quant.key} type="button" disabled={!quant.fits || quant.measuring} onClick={() => onSelect(quant.key)} title={quant.quality ?? undefined}
        className={cn("flex flex-col items-start rounded-lg border px-3 py-2.5 text-left", !quant.fits && !quant.measuring && "opacity-40", active ? "border-blue-600 bg-blue-50/60 ring-1 ring-blue-600/30 dark:border-blue-400 dark:bg-slate-800" : "border-slate-200 bg-white hover:border-slate-300 dark:border-slate-750 dark:bg-slate-850")}>
        <span className="flex w-full items-center justify-between gap-1 text-[13px] font-medium">{quant.bits}-bit
          {owned.has(quant.key) ? <Badge>Downloaded</Badge> : recommended === quant.key && <Badge tone="blue">Best</Badge>}</span>
        <span className="w-full truncate text-[11px] text-slate-500">{quantSource(quant)}</span>
        {quant.fits && quant.servedContext !== null && limit !== null && quant.servedContext < limit && <span className="text-[11px] text-orange-700 dark:text-orange-300">{formatContext(quant.servedContext)} context</span>}
        <span className="mt-2.5 flex w-full items-baseline justify-between gap-2 tabular-nums">
          <span className={cn("text-[13px] font-medium", quant.measuring && "font-normal text-slate-500")}>{quant.fits && quant.speed !== null ? `~${formatSpeed(quant.speed)}` : fitLabel(quant)}</span>
          <span className="text-xs text-slate-500">{formatStorageSize(quant.sizeBytes)}</span>
        </span>
      </button>
    })}
  </div>
}

/** Quants from different publishers, one row each in the order given: bits, publisher and quant name, then speed and size. */
export function QuantList({ quants, selected, recommended, owned, onSelect }: {
  quants: readonly QuantView[]; selected: string | null; recommended: string | null; owned: ReadonlySet<string>; onSelect: (key: string) => void
}) {
  return <div className="divide-y divide-slate-200 overflow-hidden rounded-lg border border-slate-200 bg-white dark:divide-slate-750 dark:border-slate-750 dark:bg-slate-850">
    {quants.map(quant => {
      const active = selected === quant.key
      return <button key={quant.key} type="button" disabled={!quant.fits || quant.measuring} onClick={() => onSelect(quant.key)}
        className={cn("grid w-full grid-cols-[52px_minmax(0,1fr)_auto_72px_64px] items-center gap-3 px-3 py-2 text-left text-[13px]", active ? "bg-blue-50/60 dark:bg-slate-800" : "hover:bg-slate-50 dark:hover:bg-slate-800/50", !quant.fits && !quant.measuring && "opacity-40")}>
        <span className={cn("font-medium", active && "text-blue-700 dark:text-blue-400")}>{quant.bits}-bit</span>
        <span className="min-w-0 truncate text-xs text-slate-500">{quant.repository ? <RepositoryLink repository={quant.repository}>{quantSource(quant)}</RepositoryLink> : quantSource(quant)}</span>
        <span>{owned.has(quant.key) ? <Badge>Downloaded</Badge> : recommended === quant.key && <Badge tone="blue">Best</Badge>}</span>
        <span className="text-right text-xs tabular-nums text-slate-700 dark:text-slate-300">{quant.fits && quant.speed !== null ? `~${formatSpeed(quant.speed)}` : fitLabel(quant)}</span>
        <span className="text-right text-xs tabular-nums text-slate-500">{formatStorageSize(quant.sizeBytes)}</span>
      </button>
    })}
  </div>
}

export function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "blue" | "amber" }) {
  const tones = {
    slate: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
    blue: "bg-blue-50 text-blue-700 dark:bg-slate-800 dark:text-blue-400",
    amber: "border border-orange-300 text-orange-700 dark:bg-slate-800 dark:text-orange-300",
  }
  return <span className={cn("inline-flex items-center gap-1 whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-medium leading-4", tones[tone])}>{children}</span>
}

export function Section({ title, aside, children, className = "mt-8" }: { title: ReactNode; aside?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={className}>
    <div className="mb-3 flex min-h-7 items-center justify-between gap-3">
      <h2 className="font-heading text-sm font-semibold text-slate-900 dark:text-slate-100">{title}</h2>
      {aside && <span className="flex items-center gap-3 text-xs text-slate-500">{aside}</span>}
    </div>
    {children}
  </section>
}

/** The machine's photo: a placeholder until it has loaded, and nothing if it can't load. */
function MachinePhoto({ photo }: { photo: NonNullable<MachineView["photo"]> }) {
  const [state, setState] = useState<"loading" | "loaded" | "failed">("loading")
  const frame = photo.framing
  if (!frame || state === "failed") return null
  return <div className="relative aspect-[4/3] w-16 shrink-0">
    {state === "loading" && <Skeleton className="absolute inset-0" />}
    <svg role="img" aria-label={photo.subject} viewBox={`${frame.x} ${frame.y} ${frame.width} ${frame.height}`} className={cn("h-full w-full", state === "loading" && "invisible")} preserveAspectRatio="xMidYMid meet">
      <image href={photo.src} width={frame.sourceWidth} height={frame.sourceHeight} onLoad={() => setState("loaded")} onError={() => setState("failed")} />
    </svg>
  </div>
}

/** The machine everything is measured against: what it is and its numbers. */
export function MachineCard({ machine }: { machine: MachineView }) {
  const gpu = machine.gpus[machine.servingGpu]
  const several = machine.gpus.length > 1
  const parts = [
    ...(machine.unifiedMemoryBytes !== null ? [{ icon: MemoryIcon, name: "Memory", value: formatMemoryCapacity(machine.unifiedMemoryBytes), note: null }] : []),
    ...(machine.unifiedMemoryBytes === null && gpu ? [{ icon: GraphicsCardIcon, name: "GPU memory", value: formatMemoryCapacity(gpu.memoryBytes), note: several ? `GPU ${machine.servingGpu + 1} · ${gpu.name}` : null }] : []),
    ...(machine.unifiedMemoryBytes === null ? [{ icon: MemoryIcon, name: "RAM", value: formatMemoryCapacity(machine.ramBytes), note: null }] : []),
    ...(machine.freeDiskBytes !== null ? [{ icon: HardDrivesIcon, name: "Free disk", value: formatStorageSize(machine.freeDiskBytes), note: null }] : []),
  ]
  return <div className={cn(cardClass, "flex flex-wrap items-center gap-x-4 gap-y-4 px-5 py-4")}>
    {machine.photo && <MachinePhoto photo={machine.photo} />}
    <div className="min-w-56 flex-1">
      <h2 className="font-heading text-base font-semibold">Your machine</h2>
      <p className="truncate text-[14px] text-slate-700 dark:text-slate-300">{machine.name}{(several || machine.chip) && <span className="text-slate-500"> · {several ? `${machine.gpus.length} GPUs` : machine.chip}</span>}</p>
      {several && <p className="mt-0.5 text-xs text-slate-500">{machine.gpus.map(entry => entry.name).join(" + ")} · models run on GPU {machine.servingGpu + 1}</p>}
    </div>
    <div className={cn("grid shrink-0 items-start gap-8", parts.length >= 3 ? "grid-cols-3" : "grid-cols-2")}>
      {parts.map(part => <div key={part.name}>
        <p className="flex items-center gap-1.5 text-xs text-slate-500"><part.icon className="size-3.5" />{part.name}</p>
        <p className="mt-0.5 font-heading text-xl font-semibold tabular-nums">{part.value}</p>
        {part.note && <p className="mt-0.5 text-[11px] text-slate-500">{part.note}</p>}
      </div>)}
    </div>
  </div>
}

/** A selectable model card whose header summarizes the chosen quant and whose body opens in place. */
export function ModelCardShell({ lab, title, note, detail, quant, active, onSelect, children }: {
  lab: ModelLab | undefined; title: string; note?: string | undefined; detail: ReactNode; quant: QuantView; active: boolean; onSelect: () => void; children?: ReactNode
}) {
  return <div className={cn("overflow-hidden rounded-xl border bg-white dark:bg-slate-850", active ? "border-blue-600 ring-1 ring-blue-600/30 dark:border-blue-400" : "border-slate-200 hover:border-slate-300 dark:border-slate-750 dark:hover:border-slate-700")}>
    <div role="button" tabIndex={0} aria-expanded={active} onClick={onSelect} onKeyDown={event => { if (event.key === "Enter") onSelect() }} className="grid w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-4 px-4 py-3 text-left">
      <LabMark lab={lab} className="size-10" />
      <span className="min-w-0">
        <span className="flex items-center gap-2 text-[14px] font-medium"><span className="truncate">{title}</span>{note && <span className="shrink-0 text-xs font-normal text-slate-500">{note}</span>}</span>
        <span className="mt-0.5 block min-w-0 truncate text-xs text-slate-500">{detail}</span>
      </span>
      <span className="flex items-center gap-4">
        <span className="grid grid-cols-[84px_64px] gap-4 text-right">
          <span><span className="block text-[15px] font-medium tabular-nums text-slate-900 dark:text-slate-100">{quant.fits && quant.speed !== null ? `~${formatSpeed(quant.speed)}` : "—"}</span><span className="block whitespace-nowrap text-[11px] text-slate-500">{quant.phrase ?? fitLabel(quant)}</span></span>
          <span><span className="block text-[15px] font-medium tabular-nums text-slate-900 dark:text-slate-100">{formatStorageSize(quant.sizeBytes)}</span><span className="block text-[11px] text-slate-500">on disk</span></span>
        </span>
        <span className={cn("flex size-5 items-center justify-center rounded-full border", active ? "border-blue-600 bg-blue-600 text-white dark:border-blue-400 dark:bg-blue-400 dark:text-slate-925" : "border-slate-300 dark:border-slate-700")}>{active && <CheckIcon className="size-3" />}</span>
      </span>
    </div>
    {children}
  </div>
}
