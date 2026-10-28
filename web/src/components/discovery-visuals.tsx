import { HardwarePending } from "./page-skeletons"
import { ErrorNotice } from "./error-notice"
import { pageLayout } from "./page-layout"
import { Option } from "effect"
import { useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { InfoIcon } from "@phosphor-icons/react"
import { useAgentClient, localModelQuantizationAware, localModelRadarAxes, localModelSpeedTooltip, quantizationAwareTooltip, useLocalInferenceHardware } from "@magnitudedev/client-common"
import { ActionTooltip } from "./ui/tooltip"
import { type HardwarePhoto } from "./hardware-photos"
import { HardwareSummary } from "./machine-summary"
import { useSession } from "../session"
export { HardwareSummary } from "./machine-summary"
import type { LocalModel } from "@magnitudedev/sdk"

/** An estimated speed's small slate info icon, explaining what the estimate leaves out. */
export function SpeedInfo() {
  return <ActionTooltip label={localModelSpeedTooltip} trigger={<button type="button" aria-label={localModelSpeedTooltip} className="inline-flex rounded-sm text-slate-400 hover:text-slate-600 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-500 dark:hover:text-slate-300"><InfoIcon aria-hidden="true" className="size-3" /></button>} />
}

/** A quantization-aware model's small slate info icon, explaining what QAT means. A non-interactive trigger, so it
 * can sit inside a row that is itself a button. */
export function QuantizationAwareInfo({ model, className = "" }: { model: LocalModel; className?: string }) {
  if (!localModelQuantizationAware(model)) return null
  return <ActionTooltip label={quantizationAwareTooltip} trigger={<span role="img" aria-label={quantizationAwareTooltip} className={`inline-flex shrink-0 text-slate-400 hover:text-slate-600 dark:text-slate-500 dark:hover:text-slate-300 ${className}`}><InfoIcon aria-hidden="true" className="size-3" /></span>} />
}

/** The five-axis capability profile for any model; a custom model's intelligence is never rated, so that axis stays empty. */
export function ModelRadar({ model }: { model: LocalModel }) {
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
      {[20,40,60,80].map(radius => <polygon key={radius} points={polygon(radius)} fill="none" className="stroke-slate-200 dark:stroke-slate-700" strokeWidth="0.8" />)}
      {axes.value.map((axis,index) => <line key={axis.label} x1="180" y1="138" x2={point(index,80)[0]} y2={point(index,80)[1]} className="stroke-slate-200 dark:stroke-slate-700" strokeWidth="0.8" />)}
      <path d={profilePath} style={{ d: `path("${profilePath}")` }} className="motion-safe:transition-[d] motion-safe:duration-300 motion-safe:ease-out" fill="currentColor" fillOpacity="0.13" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      {axes.value.map((axis,index) => {
        const [x,y] = [[180,20],[290,83],[258,238],[102,238],[70,83]][index]!
        const label = axis.label.charAt(0)+axis.label.slice(1).toLowerCase()
        // Speed's label carries its estimate marker and info icon: an HTML row centered on the
        // axis, occupying the label line's box above the detail's baseline.
        return <g key={axis.label}>
          {axis.label === "SPEED"
            ? <foreignObject x={x - 70} y={y - 11} width="140" height="14">
                <div className="flex h-full items-center justify-center gap-1 text-[11px] leading-none text-slate-500 dark:text-slate-400">{label} (est.)<SpeedInfo /></div>
              </foreignObject>
            : <text x={x} y={y} textAnchor="middle" className="fill-slate-500 dark:fill-slate-400" fontSize="11">{label}</text>}
          <text x={x} y={y + 18} textAnchor="middle" className="fill-slate-800 dark:fill-slate-200" fontSize="13" fontWeight="500">{axis.detail}</text>
        </g>
      })}
    </svg>
  </div>
}

export function HardwareOverview() {
  const client = useAgentClient()
  const session = useSession()
  const state = useAtomValue(session.discover)
  const setExpanded = useAtomSet(useMemo(() => Atom.fn((expanded: boolean) => session.updateDiscover({ machineExpanded: expanded })), [session]))
  const machine = useAtomValue(client.Configuration.GetServerMachine({})).result
  const hardware = useLocalInferenceHardware()
  if (Result.isFailure(hardware)) return <ErrorNotice title="Couldn’t read your hardware" description="Recommendations will return when hardware information is available." className="mb-5" />
  if (!Result.isSuccess(hardware) || Result.isInitial(machine)) return <HardwarePending identifying={Result.isSuccess(hardware)} />
  return <HardwareSummary identity={Result.isSuccess(machine) ? machine.value.identity : null} value={hardware.value} expanded={state.machineExpanded} onExpandedChange={setExpanded} />
}
export function HardwarePhotograph({ photo }: { photo: HardwarePhoto }) {
  const frame = photo.framing
  return <figure className={`${pageLayout.hardwarePhoto} aspect-[4/3] p-[6%]`}>
    <svg role="img" aria-label={photo.subject} viewBox={`${frame.x} ${frame.y} ${frame.width} ${frame.height}`} className="h-full w-full" preserveAspectRatio="xMidYMid meet">
      <image href={photo.src} width={frame.sourceWidth} height={frame.sourceHeight} />
    </svg>
  </figure>
}
