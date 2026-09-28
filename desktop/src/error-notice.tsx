import type { ReactNode } from "react"
import { InfoIcon, WarningCircleIcon } from "@phosphor-icons/react"
import { Button } from "../../web/src/components/ui/button"

export interface NoticeContent {
  readonly title: string
  readonly description?: string
  readonly severity?: "error" | "warning" | "info"
}

/** Presentation only: the feature owns the action and its admission/pending rules. */
export function NoticeAction({ children, onClick, disabled = false }: {
  children: ReactNode; onClick: () => void; disabled?: boolean
}) {
  return <Button variant="link" size="unstyled" disabled={disabled} onClick={onClick}
    className="min-h-6 py-0.5 text-xs font-medium text-inherit underline-offset-4 hover:underline">{children}</Button>
}

export function ErrorNotice({ title, description, severity = "error", actions, children, className = "" }: NoticeContent & {
  readonly actions?: ReactNode
  readonly children?: ReactNode
  readonly className?: string
}) {
  const Icon = severity === "info" ? InfoIcon : WarningCircleIcon
  const colors = severity === "error"
    ? "border-red-200 bg-red-200/10 text-red-700 dark:border-red-800 dark:bg-red-800/10 dark:text-red-300"
    : severity === "warning"
      ? "border-orange-200 bg-orange-200/10 text-orange-700 dark:border-orange-700 dark:bg-orange-700/10 dark:text-orange-300"
      : "border-slate-200 bg-slate-50 text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
  return <div role={severity === "error" ? "alert" : "status"} className={`flex min-w-0 items-start gap-2.5 rounded-lg border px-3 py-2.5 ${colors} ${className}`}>
    <Icon aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
    <div className="min-w-0 flex-1 break-words text-left">
      <p className="m-0 text-[13px] font-semibold leading-5 text-slate-900 dark:text-slate-100">{title}</p>
      {description && <p className="mt-0.5 text-xs leading-5 text-slate-600 dark:text-slate-400">{description}</p>}
      {actions && <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1">{actions}</div>}
      {children}
    </div>
  </div>
}
