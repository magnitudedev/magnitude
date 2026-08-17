import type { ReactNode } from "react"
import { ArrowLeft } from "lucide-react"
export interface ChatColumnPageProps {
  title: ReactNode
  backLabel?: string
  onBack: () => void
  children: ReactNode
  actions?: ReactNode
}
export function ChatColumnPage({
  title,
  backLabel = "Back to session",
  onBack,
  children,
  actions,
}: ChatColumnPageProps): ReactNode {
  return (
    <>
      <div className="mac:[-webkit-app-region:drag] h-11 shrink-0 flex items-center px-4 bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 select-none">
        <button
          type="button"
          onClick={onBack}
          aria-label={backLabel}
          title={backLabel}
          className="bg-transparent hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 [width:28px] [height:28px] flex items-center justify-center [background:transparent] border-0 rounded-[4px] cursor-pointer shrink-0 [margin-right:8px]"
        >
          <ArrowLeft size={16} />
        </button>
        <span className="min-w-0 max-w-[60%] overflow-hidden text-ellipsis whitespace-nowrap text-slate-900 dark:text-slate-200 font-sans text-[15px] font-medium">
          {title}
        </span>
        {actions ? (
          <div className="[margin-left:auto] flex items-center [gap:8px]">
            {actions}
          </div>
        ) : null}
      </div>
      <div className="flex min-h-0 flex-1 flex-col">{children}</div>
    </>
  )
}
