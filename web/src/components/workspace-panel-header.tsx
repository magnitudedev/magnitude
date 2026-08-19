import type { ReactNode } from "react"
import { Files, Globe2, PanelRight, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ActionTooltip } from "@/components/ui/tooltip"

export function WorkspacePanelHeader({
  surface,
  filesEnabled,
  browserEnabled,
  onSurfaceChange,
  onCollapse,
}: {
  readonly surface: "files" | "browser"
  readonly filesEnabled: boolean
  readonly browserEnabled: boolean
  readonly onSurfaceChange: (surface: "files" | "browser") => void
  readonly onCollapse: () => void
}): ReactNode {
  return (
    <header className="flex h-11 shrink-0 select-none items-center gap-2 border-b border-slate-200 px-2 dark:border-slate-800 [-webkit-app-region:drag]">
      <ActionTooltip
        label="Collapse sidebar"
        side="bottom"
        trigger={(
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onCollapse}
            className="[-webkit-app-region:no-drag]"
            aria-label="Collapse sidebar"
          >
            <PanelRight size={18} />
          </Button>
        )}
      />
      {browserEnabled ? <div
          className="flex items-center rounded-md bg-slate-100 p-0.5 dark:bg-slate-900 [-webkit-app-region:no-drag]"
          role="group"
          aria-label="Workspace panel"
        >
        <Button
          variant="unstyled"
          size="unstyled"
          aria-pressed={surface === "files"}
          disabled={!filesEnabled}
          onClick={() => onSurfaceChange("files")}
          className="flex h-7 items-center gap-1.5 rounded px-2 font-sans text-xs font-medium text-slate-500 hover:text-slate-800 aria-pressed:bg-white aria-pressed:text-slate-900 aria-pressed:shadow-xs disabled:opacity-40 dark:text-slate-400 dark:hover:text-slate-200 dark:aria-pressed:bg-slate-750 dark:aria-pressed:text-slate-100"
        >
          <Files size={14} />
          Files
        </Button>
        <Button
          variant="unstyled"
          size="unstyled"
          aria-pressed={surface === "browser"}
          onClick={() => onSurfaceChange("browser")}
          className="flex h-7 items-center gap-1.5 rounded px-2 font-sans text-xs font-medium text-slate-500 hover:text-slate-800 aria-pressed:bg-white aria-pressed:text-slate-900 aria-pressed:shadow-xs dark:text-slate-400 dark:hover:text-slate-200 dark:aria-pressed:bg-slate-750 dark:aria-pressed:text-slate-100"
        >
          <Globe2 size={14} />
          Browser
        </Button>
        </div> : (
          <span className="font-sans text-[15px] font-medium text-slate-900 dark:text-slate-200">
            Project Files
          </span>
        )}
      <div className="min-w-0 flex-1" />
      <Button
        variant="ghost"
        size="icon-sm"
        onClick={onCollapse}
        className="[-webkit-app-region:no-drag]"
        aria-label="Close sidebar"
      >
        <X size={16} />
      </Button>
    </header>
  )
}
