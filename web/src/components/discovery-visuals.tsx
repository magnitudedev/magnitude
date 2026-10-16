import { ErrorNotice } from "./error-notice"
import { HardwarePending } from "./page-skeletons"
import { pageLayout } from "./page-layout"
import { Option } from "effect"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { MemoryIcon, CircuitryIcon, CpuIcon, InfoIcon } from "@phosphor-icons/react"
import { useAgentClient, localModelRadarAxes, useLocalInferenceHardware } from "@magnitudedev/client-common"
import { ActionTooltip } from "./ui/tooltip"
import { type HardwarePhoto } from "./hardware-photos"
import { hardwareDetails } from "./hardware-details"
import type { MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
import type { CatalogLocalModel, LocalInferenceHardware } from "@magnitudedev/sdk"

import { readLanguage, translate, type MessageKey } from "../i18n"

const tr = (key: Parameters<typeof translate>[1]) => translate(readLanguage(), key)
const hardwareLabelKeys: Record<string, MessageKey> = {
  "System RAM": "systemRam", "Unified memory": "unifiedMemory", "CPU cores": "cpuCores",
  "Laptop": "laptop", "Gaming laptop": "gamingLaptop", "Gaming tablet": "gamingTablet",
  "Mobile workstation": "mobileWorkstation", "Mini PC": "miniPc", "AI mini PC": "aiMiniPc",
  "Desktop": "desktop", "Workstation": "workstation", "All-in-one": "allInOne",
  "Server": "server", "Graphics card": "graphicsCard", "Unknown": "unknown", "Unknown processor": "unknownProcessor", "Chip": "chip", "CPU": "cpuLabel", "GPU": "gpuLabel",
  "Memory bandwidth (spec)": "memoryBandwidthSpec", "CPU core options (spec)": "cpuCoreOptionsSpec", "Neural Engine cores (spec)": "neuralEngineCoresSpec", "GPU cores (spec)": "gpuCoresSpec", "CUDA cores (spec)": "cudaCoresSpec", "GPU compute units (spec)": "gpuComputeUnitsSpec", "CPU cores / processor (spec)": "cpuCoresProcessorSpec", "Xe cores (spec)": "xeCoresSpec", "Stream processors (spec)": "streamProcessorsSpec", "GPU architecture (spec)": "gpuArchitectureSpec",
}
const localizeHardwareLabel = (label: string) => hardwareLabelKeys[label] ? tr(hardwareLabelKeys[label]!) : label
const radarDetailKeys: Record<string, MessageKey> = {
  "Not assessed": "notAssessed", "Very high": "veryHigh", "High": "high",
  "Medium": "medium", "Reduced": "reduced", "Spacious": "spacious",
  "Roomy": "roomy", "Comfortable": "comfortable", "Snug": "snug",
  "Tight": "tight", "None": "none",
}
function localizeRadarDetail(detail: string): string {
  const label = Object.keys(radarDetailKeys).find(candidate => detail === candidate || detail.startsWith(`${candidate} (`))
  return label ? tr(radarDetailKeys[label]!) + detail.slice(label.length) : detail
}
function localizeHardwareDetail(detail: string): string {
  const normalized = detail.replace(/ \(spec\)/g, "")
  const exact: Record<string, MessageKey> = {
    "Shares unified memory": "sharesUnifiedMemory",
    "Shares system RAM": "sharesSystemRam",
    "Local acceleration": "localAcceleration",
  }
  if (exact[normalized]) return tr(exact[normalized]!)
  if (normalized.endsWith(" VRAM")) return normalized.replace(/ VRAM$/, ` ${tr("vramLabel")}`)
  if (normalized.endsWith(" acceleration")) return normalized.replace(/ acceleration$/, ` ${tr("acceleration")}`)
  for (const [label, key] of Object.entries(hardwareLabelKeys)) {
    const shortLabel = label.replace(/ \(spec\)/g, "")
    if (normalized === shortLabel) return tr(key)
    if (normalized.endsWith(` ${shortLabel}`)) {
      return normalized.slice(0, -shortLabel.length) + tr(key)
    }
  }
  return normalized
}

/** An estimated speed's small slate info icon, explaining what the estimate leaves out. */
export function SpeedInfo() {
  return <ActionTooltip label={tr("speedEstimateTooltip")} trigger={<button type="button" aria-label={tr("speedEstimateTooltip")} className="inline-flex rounded-sm text-slate-400 hover:text-slate-600 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-500 dark:hover:text-slate-300"><InfoIcon aria-hidden="true" className="size-3" /></button>} />
}

export function ModelRadar({ model }: { model: CatalogLocalModel }) {
  const axes = localModelRadarAxes(model)
  if (Option.isNone(axes)) return <p className="py-12 text-center text-sm text-slate-500">{model.servingState._tag === "Assessing" ? tr("waitingAssessment") : tr("noPerformanceProfile")}</p>
  const point = (index: number, radius: number) => {
    const angle = -Math.PI / 2 + index * Math.PI * 2 / 5
    return [180 + Math.cos(angle) * radius, 138 + Math.sin(angle) * radius]
  }
  const polygon = (radius: number) => axes.value.map((_, index) => point(index, radius).join(",")).join(" ")
  const profilePath = axes.value.map((axis, index) => `${index === 0 ? "M" : "L"} ${point(index, Option.getOrElse(axis.value, () => 0) * 80).join(" ")}`).join(" ") + " Z"
  return <div className={pageLayout.modelRadar}>
    <svg viewBox="0 0 360 270" role="img" aria-label={tr("capabilityProfile").replace("{name}", model.presentation.displayName)} className="block h-full w-full text-blue-600 dark:text-blue-400">
      {[20,40,60,80].map(radius => <polygon key={radius} points={polygon(radius)} fill="none" className="stroke-slate-200 dark:stroke-slate-700" strokeWidth="0.8" />)}
      {axes.value.map((axis,index) => <line key={axis.label} x1="180" y1="138" x2={point(index,80)[0]} y2={point(index,80)[1]} className="stroke-slate-200 dark:stroke-slate-700" strokeWidth="0.8" />)}
      <path d={profilePath} style={{ d: `path("${profilePath}")` }} className="motion-safe:transition-[d] motion-safe:duration-300 motion-safe:ease-out" fill="currentColor" fillOpacity="0.13" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      {axes.value.map((axis,index) => {
        const [x,y] = [[180,20],[290,83],[258,238],[102,238],[70,83]][index]!
        const axisKeys: Record<string, MessageKey> = { INTELLIGENCE: "intelligence", SPEED: "speed", SPECULATION: "speculation", FIT: "fit", FIDELITY: "fidelity" }
        const label = tr(axisKeys[axis.label] ?? "notAssessed")
        // Speed's label carries its estimate marker and info icon: an HTML row centered on the
        // axis, occupying the label line's box above the detail's baseline.
        return <g key={axis.label}>
          {axis.label === "SPEED"
            ? <foreignObject x={x - 70} y={y - 11} width="140" height="14">
                <div className="flex h-full items-center justify-center gap-1 text-[11px] leading-none text-slate-500 dark:text-slate-400">{label} (est.)<SpeedInfo /></div>
              </foreignObject>
            : <text x={x} y={y} textAnchor="middle" className="fill-slate-500 dark:fill-slate-400" fontSize="11">{label}</text>}
          <text x={x} y={y + 18} textAnchor="middle" className="fill-slate-800 dark:fill-slate-200" fontSize="13" fontWeight="500">{localizeRadarDetail(axis.detail)}</text>
        </g>
      })}
    </svg>
  </div>
}

export function HardwareOverview() {
  const client = useAgentClient()
  const machine = useAtomValue(client.Configuration.GetServerMachine({})).result
  const identity = Result.map(machine, value => value.identity)
  if (Result.isInitial(identity)) return <HardwarePending identifying />
  return <HardwareCard identity={Result.isSuccess(identity) ? identity.value : null} />
}
export function HardwarePhotograph({ photo }: { photo: HardwarePhoto }) {
  const frame = photo.framing
  return <figure className={`${pageLayout.hardwarePhoto} aspect-[4/3] p-[6%]`}>
    <svg role="img" aria-label={photo.subject} viewBox={`${frame.x} ${frame.y} ${frame.width} ${frame.height}`} className="h-full w-full" preserveAspectRatio="xMidYMid meet">
      <image href={photo.src} width={frame.sourceWidth} height={frame.sourceHeight} />
    </svg>
  </figure>
}
function HardwareCard({ identity }: { identity: MachineIdentityObservation | null }) {
  const hardware = useLocalInferenceHardware()
  if (Result.isInitial(hardware)) return <HardwarePending />
  if (Result.isFailure(hardware)) return <ErrorNotice title={tr("couldNotReadHardware")} description={tr("hardwareInfoReturns")} className="my-6" />
  if (!Result.isSuccess(hardware)) return <div className="my-6 rounded-2xl border border-slate-200 p-6 text-sm text-slate-500 dark:border-slate-750">{tr("gettingToKnowMachine")}</div>
  return <HardwareSummary identity={identity} value={hardware.value} />
}
export function HardwareSummary({ identity, value }: { identity: MachineIdentityObservation | null; value: LocalInferenceHardware }) {
  const presentation = hardwareDetails(identity, value)
  return <section aria-label={tr("yourHardware")} className={pageLayout.hardware}>
    <div className={`grid items-center gap-6 ${Option.isSome(presentation.photo) ? pageLayout.hardwareGrid : ""}`}>
      {Option.isSome(presentation.photo) && <div className="w-full max-w-[260px]"><HardwarePhotograph photo={presentation.photo.value} /></div>}
      <div className="min-w-0">
        <p className="text-xs font-medium uppercase tracking-widest text-slate-500 dark:text-slate-400">{localizeHardwareLabel(presentation.category)}{presentation.cpuInference ? ` · ${tr("cpuInference")}` : ""}</p>
        <h2 title={Option.isNone(presentation.deviceId) ? Option.getOrElse(presentation.name, () => tr("yourComputer")) : undefined} className={`mt-2 font-heading text-xl ${Option.isSome(presentation.deviceId) ? "break-words" : "truncate"}`}>{Option.getOrElse(presentation.name, () => tr("yourComputer"))}</h2>
        <div className="mt-4 grid grid-cols-[repeat(auto-fit,minmax(min(100%,220px),1fr))] gap-x-6 gap-y-4">
          {presentation.groups.map(group => {
            const Icon = group.label === "Memory" ? MemoryIcon : group.label.startsWith("GPU") ? CircuitryIcon : CpuIcon
            return <div key={group.label} className="min-w-0">
              <div className="flex items-center gap-2">
                <Icon aria-hidden="true" className="size-4 shrink-0 text-blue-600 dark:text-blue-400" />
                <p className="m-0 text-xs font-medium leading-tight text-slate-500 dark:text-slate-400">{group.label === "Memory" ? tr("memory") : localizeHardwareLabel(group.label)}</p>
              </div>
              <div className="min-w-0">
                <p title={group.truncateName ? group.name : undefined} className={`mb-0 mt-1 text-sm font-semibold leading-snug ${group.truncateName ? "truncate" : "break-words"}`}>{localizeHardwareLabel(group.name)}</p>
                <div className="mt-1 flex flex-col gap-0.5">{group.details.map(detail => <p key={detail} className="m-0 text-xs leading-snug text-slate-500 dark:text-slate-400">{localizeHardwareDetail(detail)}</p>)}</div>
              </div>
            </div>
          })}
        </div>
      </div>
    </div>
  </section>
}
