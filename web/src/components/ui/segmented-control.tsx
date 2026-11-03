/** A small set of mutually exclusive values, one pressed. */
export function SegmentedControl<T extends string>({ label, options, value, onChange }: {
  label: string
  options: ReadonlyArray<{ readonly value: T; readonly label: string }>
  value: T
  onChange: (value: T) => void
}) {
  return <div className="flex rounded-lg bg-slate-100 p-1 dark:bg-slate-900" role="group" aria-label={label}>
    {options.map(option => <button key={option.value} type="button" aria-pressed={value === option.value} onClick={() => onChange(option.value)} className={`rounded-md px-3 py-1.5 text-sm ${value === option.value ? "bg-white text-blue-700 shadow-sm dark:bg-slate-750 dark:text-blue-300" : "text-slate-500"}`}>{option.label}</button>)}
  </div>
}
