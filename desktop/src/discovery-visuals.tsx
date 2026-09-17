import { HardwarePending } from "./page-skeletons"
import { pageLayout } from "./page-layout"
import { Option } from "effect"
import { useMemo } from "react"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { MemoryIcon, CircuitryIcon, CpuIcon } from "@phosphor-icons/react"
import { DesktopSession, useAgentClient, localModelRadarAxes, useLocalInferenceHardware } from "@magnitudedev/client-common"
import { type HardwarePhoto } from "./hardware-photos"
import { hardwareDetails } from "./hardware-details"
import type { MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
import type { CatalogLocalModel, LocalInferenceHardware } from "@magnitudedev/sdk"

export function ModelRadar({ model }: { model: CatalogLocalModel }) {
  const axes = localModelRadarAxes(model)
  if (Option.isNone(axes)) return <p className="py-12 text-center text-sm text-slate-500">{model.servingState._tag === "Assessing" ? "Waiting for model assessment" : "No performance profile is available for this configuration."}</p>
  const point = (index: number, radius: number) => {
    const angle = -Math.PI / 2 + index * Math.PI * 2 / 5
    return [180 + Math.cos(angle) * radius, 138 + Math.sin(angle) * radius]
  }
  const polygon = (radius: number) => axes.value.map((_, index) => point(index, radius).join(",")).join(" ")
  const profilePath = axes.value.map((axis, index) => `${index === 0 ? "M" : "L"} ${point(index, Option.getOrElse(axis.value, () => 0) * 80).join(" ")}`).join(" ") + " Z"
  return <div className={pageLayout.modelRadar}>
    <svg viewBox="0 0 360 270" role="img" aria-label={`${model.presentation.displayName} capability profile`} className="block h-full w-full text-blue-600 dark:text-blue-400">
      <title>{axes.value.map(axis => `${axis.label}: ${axis.detail}`).join("; ")}</title>
      {[20,40,60,80].map(radius => <polygon key={radius} points={polygon(radius)} fill="none" className="stroke-slate-200 dark:stroke-slate-700" strokeWidth="0.8" />)}
      {axes.value.map((axis,index) => <line key={axis.label} x1="180" y1="138" x2={point(index,80)[0]} y2={point(index,80)[1]} className="stroke-slate-200 dark:stroke-slate-700" strokeWidth="0.8" />)}
      <path d={profilePath} style={{ d: `path("${profilePath}")` }} className="motion-safe:transition-[d] motion-safe:duration-300 motion-safe:ease-out" fill="currentColor" fillOpacity="0.13" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      {axes.value.map((axis,index) => {
        const [x,y] = [[180,20],[290,83],[258,238],[102,238],[70,83]][index]!
        return <text key={axis.label} x={x} y={y} textAnchor="middle">
          <tspan x={x} className="fill-slate-500 dark:fill-slate-400" fontSize="11">{axis.label.charAt(0)+axis.label.slice(1).toLowerCase()}</tspan>
          <tspan x={x} dy="18" className="fill-slate-800 dark:fill-slate-200" fontSize="13" fontWeight="500">{axis.detail}</tspan>
        </text>
      })}
    </svg>
  </div>
}

export function HardwareOverview() {
  const client = useAgentClient()
  const session = useAtomValue(useMemo(() => client.runtime.atom(DesktopSession), [client]))
  return Result.isSuccess(session) ? <ObservedHardware service={session.value} /> : <HardwareCard identity={null} />
}
function ObservedHardware({ service }: { service: DesktopSession }) {
  const identity = useAtomValue(service.machineIdentity)
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
  if (!Result.isSuccess(hardware)) return <div className="my-6 rounded-2xl border border-slate-200 p-6 text-sm text-slate-500 dark:border-slate-750">{Result.isFailure(hardware) ? "Hardware observation unavailable. Recommendations will return when it recovers." : "Getting to know your machine…"}</div>
  return <HardwareSummary identity={identity} value={hardware.value} />
}
export function HardwareSummary({ identity, value }: { identity: MachineIdentityObservation | null; value: LocalInferenceHardware }) {
  const presentation = hardwareDetails(identity, value)
  return <section aria-label="Your hardware" className={pageLayout.hardware}>
    <div className={`grid items-center gap-6 ${Option.isSome(presentation.photo) ? pageLayout.hardwareGrid : ""}`}>
      {Option.isSome(presentation.photo) && <div className="w-full max-w-[260px]"><HardwarePhotograph photo={presentation.photo.value} /></div>}
      <div className="min-w-0">
        <p className="text-xs font-medium uppercase tracking-widest text-slate-500 dark:text-slate-400">{presentation.category}{presentation.cpuInference ? " · CPU inference" : ""}</p>
        <h2 title={Option.isNone(presentation.deviceId) ? Option.getOrElse(presentation.name, () => "Your computer") : undefined} className={`mt-2 font-heading text-xl ${Option.isSome(presentation.deviceId) ? "break-words" : "truncate"}`}>{Option.getOrElse(presentation.name, () => "Your computer")}</h2>
        <div className="mt-4 grid grid-cols-[repeat(auto-fit,minmax(min(100%,220px),1fr))] gap-x-6 gap-y-4">
          {presentation.groups.map(group => {
            const Icon = group.label === "Memory" ? MemoryIcon : group.label.startsWith("GPU") ? CircuitryIcon : CpuIcon
            return <div key={group.label} className="min-w-0">
              <div className="flex items-center gap-2">
                <Icon aria-hidden="true" className="size-4 shrink-0 text-blue-600 dark:text-blue-400" />
                <p className="m-0 text-xs font-medium leading-tight text-slate-500 dark:text-slate-400">{group.label}</p>
              </div>
              <div className="min-w-0">
                <p title={group.truncateName ? group.name : undefined} className={`mb-0 mt-1 text-sm font-semibold leading-snug ${group.truncateName ? "truncate" : "break-words"}`}>{group.name}</p>
                <div className="mt-1 flex flex-col gap-0.5">{group.details.map(detail => <p key={detail} className="m-0 text-xs leading-snug text-slate-500 dark:text-slate-400">{detail.replace(/ \(spec\)/g, "")}</p>)}</div>
              </div>
            </div>
          })}
        </div>
      </div>
    </div>
  </section>
}
