import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { Cpu, MemoryStick, CircuitBoard } from "lucide-react"
import { localModelRadarAxes, useLocalInferenceHardware, formatMemorySize } from "@magnitudedev/client-common"
import type { CatalogLocalModel } from "@magnitudedev/sdk"

export function ModelRadar({ model }: { model: CatalogLocalModel }) {
  const axes = localModelRadarAxes(model)
  if (Option.isNone(axes)) return <p className="py-12 text-center text-sm text-slate-500">{model.servingState._tag === "Assessing" ? "Waiting for model assessment" : "No performance profile is available for this configuration."}</p>
  const point = (index: number, radius: number) => {
    const angle = -Math.PI / 2 + index * Math.PI * 2 / 5
    return [150 + Math.cos(angle) * radius, 118 + Math.sin(angle) * radius]
  }
  const polygon = (radius: number) => axes.value.map((_, index) => point(index, radius).join(",")).join(" ")
  return <div className="my-3">
    <svg viewBox="0 0 300 235" role="img" aria-label={`${model.presentation.displayName} capability profile`} className="mx-auto w-full max-w-60 text-blue-600 dark:text-blue-400">
      <title>{axes.value.map(axis => `${axis.label}: ${axis.detail}`).join("; ")}</title>
      {[20,40,60,80].map(radius => <polygon key={radius} points={polygon(radius)} fill="none" className="stroke-slate-200 dark:stroke-slate-700" strokeWidth="0.8" />)}
      {axes.value.map((axis,index) => <line key={axis.label} x1="150" y1="118" x2={point(index,80)[0]} y2={point(index,80)[1]} className="stroke-slate-200 dark:stroke-slate-700" strokeWidth="0.8" />)}
      <polygon points={axes.value.map((axis,index) => point(index,Option.getOrElse(axis.value,()=>0)*80).join(",")).join(" ")} fill="currentColor" fillOpacity="0.13" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      {axes.value.map((axis,index) => {const p=point(index,103);return <text key={axis.label} x={p[0]} y={p[1]} textAnchor="middle" dominantBaseline="middle" className="fill-slate-500 dark:fill-slate-400" fontSize="9" letterSpacing="0.5">{axis.label}</text>})}
    </svg>
    <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">{axes.value.map(axis => <div key={axis.label} className="flex flex-col gap-1"><dt className="text-slate-500">{axis.label.charAt(0)+axis.label.slice(1).toLowerCase()}</dt><dd className="font-medium">{axis.detail}</dd></div>)}</dl>
  </div>
}

export function HardwareOverview() {
  const hardware=useLocalInferenceHardware()
  if(!Result.isSuccess(hardware))return <div className="my-6 rounded-2xl border border-slate-200 p-6 text-sm text-slate-500 dark:border-slate-750">{Result.isFailure(hardware)?"Hardware observation unavailable. Recommendations will return when it recovers.":"Getting to know your machine…"}</div>
  const value=hardware.value
  return <section aria-label="Your hardware" className="relative my-6 overflow-hidden rounded-2xl border border-blue-200 bg-gradient-to-br from-blue-50 via-white to-slate-100 p-6 dark:border-slate-700 dark:from-slate-800 dark:via-slate-850 dark:to-slate-900">
    <div className="flex flex-wrap items-center gap-4"><div className="rounded-2xl border border-blue-200 bg-white/70 p-3 text-blue-700 dark:border-slate-650 dark:bg-slate-900 dark:text-blue-400"><Cpu className="size-7" /></div><div><p className="text-xs font-medium uppercase tracking-widest text-slate-500">Made for your machine</p><h2 className="mt-1 font-heading text-xl">{Option.getOrElse(value.processor,()=>value.platform)}</h2></div>
    <div className="ml-auto flex flex-wrap gap-x-8 gap-y-4"><div className="flex items-center gap-3"><MemoryStick className="size-5 text-blue-600 dark:text-blue-400" /><div><p className="text-lg font-semibold">{formatMemorySize(value.totalSystemMemoryBytes)}</p><p className="text-xs text-slate-500">System memory</p></div></div>{value.accelerators.map(accelerator=><div key={accelerator.name} className="flex items-center gap-3"><CircuitBoard className="size-5 text-blue-600 dark:text-blue-400" /><div><p className="text-sm font-medium">{accelerator.name}</p><p className="text-xs text-slate-500">Local acceleration</p></div></div>)}</div></div>
  </section>
}
