import { LOCAL_MODEL_RANKING_SCALE_LABELS } from "@magnitudedev/client-common"

export function ModelPreferenceSlider({ value, onChange }: { value: number; onChange?: (value: number) => void }) {
  return <div>
    <div className="relative flex h-5 items-center">
      <input type="range" min="0" max="4" step="1" aria-label="Model preference" aria-valuetext={LOCAL_MODEL_RANKING_SCALE_LABELS[value]} value={value} disabled={!onChange} onChange={event => onChange?.(Number(event.target.value))} className="h-1 w-full cursor-pointer appearance-none rounded-full bg-slate-200 accent-blue-600 dark:bg-slate-700 [&::-webkit-slider-thumb]:size-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500" />
      <div aria-hidden="true" className="pointer-events-none absolute inset-x-1.5 flex justify-between">
        {LOCAL_MODEL_RANKING_SCALE_LABELS.map((label, index) => <span key={label} data-preference-tick={index} className={`h-2 w-px ${index === value ? "bg-transparent" : "bg-slate-400 dark:bg-slate-500"}`} />)}
      </div>
    </div>
    <div className="relative mt-0.5 h-5">
      {LOCAL_MODEL_RANKING_SCALE_LABELS.map((label, index) => <button key={label} type="button" disabled={!onChange} onClick={() => onChange?.(index)} aria-pressed={index === value} className={`absolute rounded py-0.5 text-xs focus-visible:outline-2 focus-visible:outline-blue-500 ${index === value ? "font-semibold text-blue-700 dark:text-blue-400" : "text-slate-500 dark:text-slate-400"}`} style={{ left: `${index * 25}%`, transform: `translateX(${index === 0 ? 0 : index === 4 ? -100 : -50}%)` }}>{label}</button>)}
    </div>
  </div>
}

export function RecommendationPreference({ value = 2, onChange }: { value?: number; onChange?: (value: number) => void }) {
  return <section aria-label="Recommendation preference" className="mb-5 mt-3"><p className="mb-2 text-[13px] font-medium text-slate-600 dark:text-slate-300">Ranked for your machine.<span className="ml-1 font-normal text-slate-500">Choose faster replies or smarter answers.</span></p><ModelPreferenceSlider value={value} onChange={onChange} /></section>
}
