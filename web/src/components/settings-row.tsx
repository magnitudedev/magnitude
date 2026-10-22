import type { ReactNode } from "react"
import { translateKnownMessage, useI18n } from "../i18n"

export function SettingsGroup({ label, children }: { label: string; children: ReactNode }) {
  const { language } = useI18n()
  const translatedLabel = translateKnownMessage(label, language)
  return <section aria-label={translatedLabel} className="mt-7">
    <h2 className="mb-2 px-1 text-xs font-medium uppercase tracking-wide text-slate-500">{translatedLabel}</h2>
    <div className="divide-y divide-slate-200 rounded-lg border border-slate-300 bg-white dark:divide-slate-800 dark:border-slate-750 dark:bg-slate-850">{children}</div>
  </section>
}
export function SettingsRow({ label, hint, alert, control, children, nested = false }: { label: ReactNode; hint?: ReactNode; alert?: ReactNode; control?: ReactNode; children?: ReactNode; nested?: boolean }) {
  const { language } = useI18n()
  const translatedLabel = typeof label === "string" ? translateKnownMessage(label, language) : label
  const translatedHint = typeof hint === "string" ? translateKnownMessage(hint, language) : hint
  return <div className={nested ? "bg-slate-50 py-2.5 pl-6 pr-4 dark:bg-slate-900/40 md:pl-10" : "px-4 py-3"}>
    <div className="flex flex-col items-start gap-2 md:flex-row md:items-center md:justify-between md:gap-6">
      <div className="min-w-0"><p className={nested ? "text-[13px] font-medium" : "text-sm font-medium"}>{translatedLabel}</p>
        {translatedHint && <div className="mt-0.5 text-xs text-slate-500">{translatedHint}</div>}

      </div>
      {control && <div className="flex shrink-0 flex-wrap items-center gap-2 md:flex-nowrap">{control}</div>}
    </div>
    {alert && <div className="mt-2">{alert}</div>}
    {children}
  </div>
}
