import { LOCAL_MODEL_RANKING_SCALE_LABELS } from "@magnitudedev/client-common"

export function ModelPreferenceSlider({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  return <div className="mt-5">
    <div className="relative flex h-6 items-center">
      <input type="range" min="0" max="4" step="1" aria-label="Model preference" aria-valuetext={LOCAL_MODEL_RANKING_SCALE_LABELS[value]} value={value} onChange={event => onChange(Number(event.target.value))} className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-200 accent-blue-600 dark:bg-slate-700 [&::-webkit-slider-thumb]:size-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:shadow-sm" />
      <div aria-hidden="true" className="pointer-events-none absolute inset-x-2 flex items-center justify-between">
        {LOCAL_MODEL_RANKING_SCALE_LABELS.map((label, index) => <span key={label} data-preference-tick={index} className={`h-3 w-0.5 rounded-full ${index === value ? "bg-transparent" : "bg-slate-500 dark:bg-slate-400"}`} />)}
      </div>
    </div>
    <div className="relative mt-2 h-8">
      {LOCAL_MODEL_RANKING_SCALE_LABELS.map((label, index) => <button key={label} onClick={() => onChange(index)} aria-pressed={index === value} className={`absolute rounded py-1 text-xs focus-visible:outline-2 focus-visible:outline-blue-500 ${index === value ? "font-semibold text-blue-700 dark:text-blue-400" : "text-slate-500"}`} style={{ left: `${index * 25}%`, transform: `translateX(${index === 0 ? 0 : index === 4 ? -100 : -50}%)` }}>{label}</button>)}
    </div>
  </div>
}
