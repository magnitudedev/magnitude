import { FireIcon } from "@phosphor-icons/react"
import { useRef, useState, type KeyboardEvent } from "react"
import type { ServingUsageSnapshot } from "@magnitudedev/sdk"
import { ActionTooltip, TooltipProvider } from "../../web/src/components/ui/tooltip"
import { Skeleton } from "../../web/src/components/ui/skeleton"

type Activity = Extract<ServingUsageSnapshot, { _tag: "Available" }>["dailyActivity"]
const colors = [
  "bg-slate-100 dark:bg-slate-800",
  "bg-blue-100 dark:bg-blue-900",
  "bg-blue-300 dark:bg-blue-700",
  "bg-blue-500 dark:bg-blue-500",
  "bg-blue-700 dark:bg-blue-300",
] as const
const dateLabel = (date: string) => new Date(`${date}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", month: "long", day: "numeric", year: "numeric" })

export function activitySummary(days: Activity) {
  const totalTokens = days.reduce((total, day) => total + day.totalTokens, 0)
  // An unfinished today does not break yesterday's streak.
  let currentStreak = 0
  for (let index = days.length - (days.at(-1)?.totalTokens ? 1 : 2); index >= 0 && days[index]!.totalTokens > 0; index--) currentStreak++
  return { totalTokens, currentStreak }
}
export function activityLevel(tokens: number, maximum: number): 0 | 1 | 2 | 3 | 4 {
  return tokens === 0 ? 0 : Math.min(4, Math.max(1, Math.ceil(tokens / maximum * 4))) as 1 | 2 | 3 | 4
}

export function UsageActivity({ days }: { days: Activity | null }) {
  const table = useRef<HTMLTableElement>(null)
  const [focusedDate, setFocusedDate] = useState<string | null>(null)
  const summary = days ? activitySummary(days) : null
  const maximum = Math.max(1, ...(days?.map(day => day.totalTokens) ?? []))
  const weeks = Array.from({ length: 53 }, (_, index) => days?.slice(index * 7, index * 7 + 7) ?? [])
  const focusDate = days?.some(day => day.date === focusedDate) ? focusedDate : days?.at(-1)?.date
  const navigate = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (!days) return
    const row = index % 7
    const lastInRow = row + Math.floor((days.length - 1 - row) / 7) * 7
    const target = { ArrowLeft: index - 7, ArrowRight: index + 7, ArrowUp: index - 1, ArrowDown: index + 1,
      Home: row, End: lastInRow, PageUp: index - row, PageDown: Math.min(index - row + 6, days.length - 1) }[event.key]
    if (target === undefined) return
    event.preventDefault()
    const next = Math.max(0, Math.min(days.length - 1, target))
    setFocusedDate(days[next]!.date)
    table.current?.querySelector<HTMLButtonElement>(`[data-day-index="${next}"]`)?.focus()
  }
  return <section aria-label="Token activity" aria-busy={!days}>
    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
      <h2 className="text-sm font-medium">Token activity</h2>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
        {summary ? <span aria-label={`Current streak: ${summary.currentStreak} ${summary.currentStreak === 1 ? "day" : "days"}`} className="flex items-center gap-1.5 text-slate-500"><FireIcon aria-hidden="true" weight="fill" className="size-4 text-orange-500" /><span className="font-medium tabular-nums text-slate-900 dark:text-slate-100">{summary.currentStreak}</span> {summary.currentStreak === 1 ? "day" : "days"} streak</span> : <Skeleton className="h-5 w-28" />}
        {summary ? <p className="text-slate-500"><strong className="font-medium tabular-nums text-slate-900 dark:text-slate-100">{summary.totalTokens.toLocaleString()}</strong> tokens · Past year</p> : <Skeleton className="h-5 w-40" />}
      </div>
    </div>
    <div className="overflow-x-auto pb-2">
      <TooltipProvider>
        <table ref={table} aria-label="Daily token activity, Sunday through Saturday; use arrow keys to navigate" className="w-full min-w-[640px] table-fixed border-separate border-spacing-[3px]">
          <thead><tr><td className="w-8" />{weeks.map((week, column) => {
            const month = week.find(day => day.date.endsWith("-01"))
            return <th key={column} scope="col" className="h-6 overflow-visible whitespace-nowrap text-left text-xs font-normal text-slate-500">{month && new Date(`${month.date}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", month: "short" })}</th>
          })}</tr></thead>
          <tbody>{["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((weekday, row) => <tr key={weekday}>
            <th scope="row" className="text-left text-[10px] font-normal text-slate-500"><span className={row % 2 === 0 ? "sr-only" : ""}>{weekday}</span></th>
            {weeks.map((week, column) => {
              const day = week[row]
              const cell = "block aspect-square w-full rounded-[3px]"
              if (!days) return <td key={column}><Skeleton className={cell} /></td>
              if (!day) return <td key={column}><span aria-hidden="true" className={`${cell} invisible`} /></td>
              const label = `${day.totalTokens.toLocaleString()} tokens on ${dateLabel(day.date)}`
              return <td key={column}><ActionTooltip label={label} trigger={<button type="button" aria-label={label} data-day-index={column * 7 + row} data-activity-level={activityLevel(day.totalTokens, maximum)} tabIndex={day.date === focusDate ? 0 : -1}
                onFocus={() => setFocusedDate(day.date)} onKeyDown={event => navigate(event, column * 7 + row)}
                className={`${cell} ${colors[activityLevel(day.totalTokens, maximum)]} outline-offset-2 focus-visible:outline-2 focus-visible:outline-blue-500`} />} /></td>
            })}
          </tr>)}</tbody>
        </table>
      </TooltipProvider>
    </div>
    <div className="mt-2 flex justify-end text-xs text-slate-500">
      <div className="flex shrink-0 items-center gap-1.5" aria-label="Activity scale: no tokens, then four increasing levels relative to the busiest day"><span className="mr-1">Less</span>{colors.map(color => <span key={color} aria-hidden="true" className={`size-3 rounded-[3px] ${color}`} />)}<span className="ml-1">More</span></div>
    </div>
  </section>
}
