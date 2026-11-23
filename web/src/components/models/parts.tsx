import { useState, type ReactNode } from "react"
import { Skeleton } from "../ui/skeleton"
import { ActionTooltip } from "../ui/tooltip"
import { ArrowSquareOutIcon, CheckIcon, CircleNotchIcon, CubeIcon, DesktopTowerIcon, GraphicsCardIcon, HardDrivesIcon, InfoIcon, LaptopIcon, MemoryIcon, MonitorIcon } from "@phosphor-icons/react"
import { formatContext, formatMemoryCapacity, formatStorageSize } from "@magnitudedev/client-common"
import { cn } from "@/lib/utils"
import { modelLabs, type ModelLab } from "../model-logo"
import type { MachineView, PlacementTier, QuantView } from "./view"

export const cardClass = "rounded-xl border border-slate-200 bg-white dark:border-slate-750 dark:bg-slate-850"

export const formatSpeed = (speed: number) => speed < 1 ? "<1 tok/s" : `${Math.round(speed)} tok/s`

export const bitWidthLabel = (bits: number): string | null => bits > 0 ? `${bits}-bit` : null

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

/** A repository's page on Hugging Face as an icon alone, beside a name that does something else when clicked. */
export function RepositoryIconLink({ repository }: { repository: string }) {
  return <a href={`https://huggingface.co/${repository}`} target="_blank" rel="noreferrer" title={repository} aria-label={`Open ${repository} on Hugging Face`}
    className="relative z-10 inline-flex shrink-0 items-center rounded p-0.5 hover:text-blue-700 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:text-blue-400">
    <ArrowSquareOutIcon className="size-3" />
  </a>
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
  Lookup: "border border-dashed border-slate-400 dark:border-slate-500",
}
const tierPhrase: Record<PlacementTier["kind"], string> = {
  Device: "held in GPU",
  Unified: "held in memory",
  Host: "held in RAM",
  Disk: "streamed from SSD",
  Lookup: "lookup table on SSD",
}
const tierHint: Partial<Record<PlacementTier["kind"], string>> = {
  Lookup: "Too large to keep in memory. Each token reads only a tiny part of it, so it barely affects speed.",
}

/**
 * How a quant runs: what stays in fast memory, what streams from the next tier while it runs, and a lookup table too
 * large to cache, which stays on disk and costs a few rows per token.
 */
export function PlacementBar({ quant }: { quant: QuantView }) {
  const tiers = quant.placement.filter(tier => tier.bytes > 0)
  const total = tiers.reduce((sum, tier) => sum + tier.bytes, 0)
  if (total === 0) return null
  const bits = bitWidthLabel(quant.bits)
  return <div>
    <p className="mb-2 text-xs font-medium text-slate-700 dark:text-slate-300">{bits ? `How ${bits} runs on your machine` : "How this model runs on your machine"}</p>
    <div className="flex gap-0.5">
      {tiers.map(tier => <div key={tier.kind} className={cn("h-2 min-w-1 rounded-full", tierColor[tier.kind])} style={{ width: `${tier.bytes / total * 100}%` }} />)}
    </div>
    <p className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-slate-500">
      {tiers.map(tier => {
        const hint = tierHint[tier.kind]
        return <span key={tier.kind} className="inline-flex items-center gap-1.5 whitespace-nowrap">
          <span className={cn("size-2 rounded-full", tierColor[tier.kind], "border-solid")} />
          <span className="font-medium text-slate-700 dark:text-slate-300">{formatStorageSize(tier.bytes)}</span>{tierPhrase[tier.kind]}
          {hint && <ActionTooltip label={hint} trigger={<button type="button" aria-label={hint} className="inline-flex rounded-sm text-slate-400 hover:text-slate-600 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-500 dark:hover:text-slate-300"><InfoIcon aria-hidden="true" className="size-3" /></button>} />}
        </span>
      })}
    </p>
  </div>
}

const MAX_LADDER_RUNGS = 4

/**
 * At most `max` rungs around the recommended choice (up to two faster below it and one smarter above, since anything
 * above the pick is already below the speed target), keeping any owned or currently selected rung in view.
 */
export function visibleLadderRungs(
  quants: readonly QuantView[],
  recommended: string | null,
  owned: ReadonlySet<string>,
  selected: string | null = null,
  max = MAX_LADDER_RUNGS,
): readonly QuantView[] {
  if (quants.length <= max) return quants
  const recommendedIndex = recommended === null ? -1 : quants.findIndex(quant => quant.key === recommended)
  const center = recommendedIndex === -1 ? 0 : recommendedIndex
  const start = Math.max(0, Math.min(center - 2, quants.length - max))
  const picked = new Set(Array.from({ length: max }, (_, offset) => start + offset))
  const pinned = new Set<number>([center])
  const selectedIndex = selected === null ? -1 : quants.findIndex(quant => quant.key === selected)
  if (selectedIndex !== -1) pinned.add(selectedIndex)
  quants.forEach((quant, index) => { if (owned.has(quant.key)) pinned.add(index) })
  for (const mustKeep of pinned) {
    if (picked.has(mustKeep)) continue
    let drop: number | null = null
    for (const candidate of picked) {
      if (pinned.has(candidate)) continue
      if (drop === null) { drop = candidate; continue }
      const candidateDistance = Math.abs(candidate - center)
      const dropDistance = Math.abs(drop - center)
      if (candidateDistance > dropDistance || (candidateDistance === dropDistance && candidate > drop)) drop = candidate
    }
    if (drop === null) break
    picked.delete(drop)
    picked.add(mustKeep)
  }
  return [...picked].sort((left, right) => left - right).map(index => quants[index]!)
}

/** Rungs side by side, smallest to largest: at most four around the pick, with bits, source, speed and size. */
export function QuantLadder({ quants, selected, recommended, owned, limit, onSelect }: {
  quants: readonly QuantView[]; selected: string | null; recommended: string | null; owned: ReadonlySet<string>; limit: number | null; onSelect: (key: string) => void
}) {
  const visible = visibleLadderRungs(quants, recommended, owned, selected)
  return <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${visible.length}, minmax(0, 1fr))` }}>
    {visible.map(quant => {
      const active = selected === quant.key
      const noRoom = quant.missingDiskBytes > 0
      const metric = noRoom
        ? `Needs ${formatStorageSize(quant.missingDiskBytes)} more disk`
        : quant.measuring
          ? <span aria-label="Measuring" title="Measuring speed on this computer" className="inline-flex items-center gap-1.5 text-slate-400 dark:text-slate-500"><CircleNotchIcon className="size-3.5 motion-safe:animate-spin" /></span>
          : quant.fits && quant.speed !== null ? `~${formatSpeed(quant.speed)}` : fitLabel(quant)
      return <button key={quant.key} type="button" disabled={!quant.fits || quant.measuring || noRoom} onClick={() => onSelect(quant.key)} title={quant.quality ?? undefined}
        className={cn("flex min-w-0 flex-col items-start rounded-lg border px-3 py-2.5 text-left", ((!quant.fits && !quant.measuring) || noRoom) && "opacity-40", active ? "border-blue-600 bg-blue-50/60 ring-1 ring-blue-600/30 dark:border-blue-400 dark:bg-slate-800" : "border-slate-200 bg-white hover:border-slate-300 dark:border-slate-750 dark:bg-slate-850")}>
        <span className="flex w-full items-center justify-between gap-1 whitespace-nowrap text-[13px] font-medium"><span>{bitWidthLabel(quant.bits) ?? quant.name}</span>
          {owned.has(quant.key) ? <Badge>Downloaded</Badge> : recommended === quant.key && <Badge tone="blue">Best</Badge>}</span>
        <span className="w-full truncate text-[11px] text-slate-500">{quantSource(quant)}</span>
        {quant.fits && quant.servedContext !== null && limit !== null && quant.servedContext < limit && <span className="w-full truncate text-[11px] text-orange-700 dark:text-orange-300">{formatContext(quant.servedContext)} context</span>}
        <span className="mt-2.5 flex w-full items-baseline justify-between gap-2 tabular-nums">
          <span className={cn("whitespace-nowrap text-[13px] font-medium", (quant.measuring || noRoom) && "font-normal text-slate-500", noRoom && "truncate text-[11px]")}>{metric}</span>
          <span className="whitespace-nowrap text-xs text-slate-500">{formatStorageSize(quant.sizeBytes)}</span>
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
      const bits = bitWidthLabel(quant.bits)
      const source = quantSource(quant)
      // The whole row chooses the size, from a button laid over it; only the icon beside the name opens Hugging Face.
      return <div key={quant.key} className={cn("relative grid w-full grid-cols-[52px_minmax(0,1fr)_auto_72px_64px] items-center gap-3 px-3 py-2 text-left text-[13px]", active ? "bg-blue-50/60 dark:bg-slate-800" : "hover:bg-slate-50 dark:hover:bg-slate-800/50", !quant.fits && !quant.measuring && "opacity-40")}>
        <button type="button" aria-label={bits ? `${bits} ${source}` : source} aria-pressed={active} disabled={!quant.fits || quant.measuring} onClick={() => onSelect(quant.key)}
          className="absolute inset-0 cursor-pointer focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-blue-500 disabled:cursor-default" />
        <span className={cn("font-medium", active && "text-blue-700 dark:text-blue-400")}>{bits ?? "—"}</span>
        <span className="flex min-w-0 items-center gap-1 text-xs text-slate-500"><span className="truncate">{source}</span>{quant.repository && <RepositoryIconLink repository={quant.repository} />}</span>
        <span>{owned.has(quant.key) ? <Badge>Downloaded</Badge> : recommended === quant.key && <Badge tone="blue">Best</Badge>}</span>
        <span className="text-right text-xs tabular-nums text-slate-700 dark:text-slate-300">{quant.fits && quant.speed !== null ? `~${formatSpeed(quant.speed)}` : fitLabel(quant)}</span>
        <span className="text-right text-xs tabular-nums text-slate-500">{formatStorageSize(quant.sizeBytes)}</span>
      </div>
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

/** Single-line summary of the machine for the Models page: name, memory tiers and free disk, separated by spacing. */
export function MachineStrip({ machine, formFactor }: { machine: MachineView; formFactor?: string }) {
  const gpu = machine.gpus[machine.servingGpu]
  const several = machine.gpus.length > 1
  const DeviceIcon = formFactor === "Laptop" ? LaptopIcon : formFactor === "Server" ? HardDrivesIcon : formFactor === "Desktop" ? DesktopTowerIcon : MonitorIcon
  const specs = [
    ...(machine.unifiedMemoryBytes !== null ? [{ icon: MemoryIcon, value: formatMemoryCapacity(machine.unifiedMemoryBytes), label: "memory" }] : []),
    ...(machine.unifiedMemoryBytes === null && gpu ? [{ icon: GraphicsCardIcon, value: formatMemoryCapacity(gpu.memoryBytes), label: several ? `GPU ${machine.servingGpu + 1}` : "GPU" }] : []),
    ...(machine.unifiedMemoryBytes === null ? [{ icon: MemoryIcon, value: formatMemoryCapacity(machine.ramBytes), label: "RAM" }] : []),
    ...(machine.freeDiskBytes !== null ? [{ icon: HardDrivesIcon, value: formatStorageSize(machine.freeDiskBytes), label: "free disk" }] : []),
  ]
  return <div aria-label={`Your machine: ${machine.name}`} className="mb-5 flex flex-wrap items-center gap-x-5 gap-y-1.5 text-xs text-slate-500">
    <span className="inline-flex min-w-0 items-center gap-1.5">
      <DeviceIcon aria-hidden="true" className="size-3.5 shrink-0 text-slate-400 dark:text-slate-500" />
      <span>Your machine:</span>
      <span className="truncate font-medium text-slate-700 dark:text-slate-200">{machine.name}</span>
      {(several || machine.chip) && <span className="truncate text-slate-500">{several ? `${machine.gpus.length} GPUs` : machine.chip}</span>}
    </span>
    {specs.map(spec => <span key={spec.label} className="inline-flex items-center gap-1.5 whitespace-nowrap tabular-nums">
      <spec.icon aria-hidden="true" className="size-3.5 shrink-0 text-slate-400 dark:text-slate-500" />
      <span className="font-medium text-slate-700 dark:text-slate-200">{spec.value}</span>
      <span>{spec.label}</span>
    </span>)}
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
    {/* The card opens from a button laid over it, so the repository link in its line is beside the button, not inside it. */}
    <div className="relative grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-4 px-4 py-3 text-left [&_a]:relative [&_a]:z-10">
      <button type="button" aria-expanded={active} aria-label={title} onClick={onSelect} className="absolute inset-0 cursor-pointer rounded-[inherit] focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-blue-500" />
      <LabMark lab={lab} className="size-10" />
      <span className="min-w-0">
        <span className="flex items-center gap-2 text-[14px] font-medium"><span className="max-w-full shrink-0 truncate">{title}</span>{note && <span className="min-w-0 truncate text-xs font-normal text-slate-500">{note}</span>}</span>
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
