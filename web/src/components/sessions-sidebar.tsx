import { useCallback, useMemo, useRef, useState, type ReactNode } from "react"
import {
  ArrowLeft,
  HardDrive,
  Layers3,
  Monitor,
  Moon,
  Settings,
  SlidersHorizontal,
  Sun,
} from "lucide-react"
import {
  CaretDown,
  CaretRight,
  DotsThreeVertical,
  FolderOpen,
  FolderPlus,
  Gear,
  MagnifyingGlass,
  NotePencil,
  SidebarSimple,
  X,
} from "@phosphor-icons/react"
import { useAtomValue, useAtomSet } from "@effect-atom/atom-react"
import type { ProjectId, ProjectRecord, ProjectSummary } from "@magnitudedev/sdk"
import {
  formatRelativeTime,
  useSelectedSessionId,
} from "@magnitudedev/client-common"
import {
  collapsedProjectIdsAtom,
  sidebarCollapsedAtom,
  sidebarSearchAtom,
  sidebarWidthAtom,
  type SettingsTab,
} from "../state/web-atoms"
import { SidebarEmptyState, SidebarLoadingState } from "./sidebar-states"
import { ProjectFormDialog, RemoveProjectDialog } from "./project-dialogs"
import { setAppearancePreference, useAppearancePreference } from "../stores/appearance-store"

interface SessionItemData {
  readonly sessionId: string
  readonly projectId: ProjectId
  readonly title: string | null
  readonly updatedAt: number
  readonly workStatus: "idle" | "working"
  readonly cwd: string
  readonly sidebarOpen: boolean
}

export interface SessionsSidebarProps {
  readonly projects?: ReadonlyArray<ProjectSummary>
  readonly sessions?: ReadonlyArray<SessionItemData>
  readonly loading?: boolean
  readonly loadingMore?: boolean
  readonly hasMore?: boolean
  readonly onSelectSession?: (session: SessionItemData) => void
  readonly onCloseSession?: (sessionId: string) => void
  readonly onCompose?: () => void
  readonly onCreateProject?: (project: ProjectRecord) => void
  readonly onEditProject?: (project: ProjectRecord) => void
  readonly onRemoveProject?: (project: ProjectRecord) => void
  readonly revealKind?: "finder" | "folder" | "unsupported"
  readonly onRevealProject?: (projectId: ProjectId) => void
  readonly onLoadMore?: () => void
  readonly onOpenSettings?: () => void
  readonly settingsTab?: SettingsTab | null
  readonly onSettingsTabChange?: (tab: SettingsTab) => void
  readonly onCloseSettings?: () => void
  readonly overlay?: boolean
  readonly onCloseOverlay?: () => void
  readonly titlebarIntegrated?: boolean
}

const settingsSections = [
  { id: "models", label: "Models", detail: "Runtime & storage", icon: Layers3 },
  { id: "catalog", label: "Catalog", detail: "Compare & choose", icon: SlidersHorizontal },
  { id: "hardware", label: "Hardware", detail: "Capacity & compute", icon: HardDrive },
] as const

function SettingsNavigation({
  activeTab,
  onTabChange,
  onBack,
}: {
  readonly activeTab: SettingsTab
  readonly onTabChange?: (tab: SettingsTab) => void
  readonly onBack?: () => void
}): ReactNode {
  return (
    <>
      <button
        type="button"
        onClick={onBack}
        className="mac:[-webkit-app-region:no-drag] mx-2 mb-[5px] mt-1 flex h-9 w-[calc(100%-16px)] shrink-0 cursor-pointer items-center gap-2 rounded-[7px] border-0 bg-transparent px-2.5 text-left text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800 mac:[-webkit-app-region:drag]"
      >
        <ArrowLeft size={16} />
        <strong className="text-[14px] font-semibold text-slate-900 dark:text-slate-200">Settings</strong>
      </button>
      <nav className="flex min-h-0 flex-1 flex-col gap-[3px] px-2 py-2.5" aria-label="Settings sections">
        {settingsSections.map(({ id, label, detail, icon: Icon }) => (
          <button
            key={id}
            type="button"
            aria-current={activeTab === id ? "page" : undefined}
            onClick={() => onTabChange?.(id)}
            className="flex min-h-[52px] w-full cursor-pointer items-center gap-[11px] rounded-[7px] border-0 bg-transparent px-2.5 py-2 text-left text-slate-500 hover:bg-slate-150 aria-[current=page]:bg-slate-200 aria-[current=page]:text-blue-700 dark:hover:bg-slate-800 dark:aria-[current=page]:bg-slate-750 dark:aria-[current=page]:text-blue-400"
          >
            <Icon size={17} />
            <span className="flex min-w-0 flex-col gap-px">
              <strong className="text-[13px] font-semibold text-slate-700 dark:text-slate-300">{label}</strong>
              <small className="text-[11px] text-slate-500">{detail}</small>
            </span>
          </button>
        ))}
      </nav>
    </>
  )
}

function SidebarFooter({
  compact,
  floating = false,
  settingsActive,
  onOpenSettings,
}: {
  readonly compact: boolean
  readonly floating?: boolean
  readonly settingsActive: boolean
  readonly onOpenSettings?: () => void
}): ReactNode {
  const appearance = useAppearancePreference()
  const nextAppearance = appearance === "system" ? "light" : appearance === "light" ? "dark" : "system"
  const AppearanceIcon = appearance === "light" ? Sun : appearance === "dark" ? Moon : Monitor
  return (
    <div className={floating
      ? "fixed bottom-3 left-3 z-20 flex items-center gap-1 [-webkit-app-region:no-drag]"
      : `flex min-h-[49px] shrink-0 items-center border-t border-slate-200 p-2 dark:border-slate-800 mac:[-webkit-app-region:no-drag] ${compact ? "flex-col gap-1" : "gap-2"}`
    }>
      <button
        type="button"
        onClick={settingsActive ? undefined : onOpenSettings}
        className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white aria-[current=page]:bg-slate-200 aria-[current=page]:text-blue-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:aria-[current=page]:bg-slate-750 dark:aria-[current=page]:text-blue-400"
        aria-label="Settings"
        aria-current={settingsActive ? "page" : undefined}
        title="Settings"
      >
        {compact ? <Gear size={17} /> : <Settings size={16} />}
      </button>
      <button
        type="button"
        onClick={() => setAppearancePreference(nextAppearance)}
        className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800"
        aria-label={`Theme: ${appearance}`}
        title={`Theme: ${appearance}`}
      >
        <AppearanceIcon size={16} />
      </button>
    </div>
  )
}

type ProjectMenuState = {
  readonly project: ProjectRecord
  readonly x: number
  readonly y: number
}

function ProjectMenu({
  state,
  onDismiss,
  onEdit,
  onRemove,
  revealKind,
  onReveal,
}: {
  readonly state: ProjectMenuState
  readonly onDismiss: () => void
  readonly onEdit: () => void
  readonly onRemove: () => void
  readonly revealKind: "finder" | "folder" | "unsupported"
  readonly onReveal: () => void
}): ReactNode {
  const revealLabel = revealKind === "finder" ? "Reveal in Finder" : "Show in folder"
  return (
    <>
      <div className="fixed inset-0 z-[89]" onMouseDown={onDismiss} />
      <div
        role="menu"
        onKeyDown={(event) => {
          if (event.key === "Escape") onDismiss()
          if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return
          const items = [...event.currentTarget.querySelectorAll<HTMLElement>('[role="menuitem"]')]
          if (items.length === 0) return
          event.preventDefault()
          const current = items.indexOf(document.activeElement as HTMLElement)
          const next = event.key === "Home"
            ? 0
            : event.key === "End"
              ? items.length - 1
              : event.key === "ArrowDown"
                ? (current + 1) % items.length
                : (current <= 0 ? items.length : current) - 1
          items[next]?.focus()
        }}
        className="fixed z-[90] min-w-[170px] rounded-lg border border-slate-200 bg-white p-1 shadow-lg dark:border-slate-700 dark:bg-slate-800"
        style={{ left: state.x, top: state.y }}
      >
        <button autoFocus type="button" role="menuitem" onClick={onEdit} className="h-8 w-full rounded-md border-0 bg-transparent px-2.5 text-left text-[13px] text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-700">Edit project</button>
        {revealKind !== "unsupported" ? (
          <button
            type="button"
            role="menuitem"
            onClick={onReveal}
            className="h-8 w-full rounded-md border-0 bg-transparent px-2.5 text-left text-[13px] text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            {revealLabel}
          </button>
        ) : null}
        <div className="my-1 h-px bg-slate-200 dark:bg-slate-700" />
        <button type="button" role="menuitem" onClick={onRemove} className="h-8 w-full rounded-md border-0 bg-transparent px-2.5 text-left text-[13px] text-red-600 hover:bg-red-200/40 dark:text-red-400 dark:hover:bg-red-800/40">Remove project</button>
      </div>
    </>
  )
}

function projectGitLabel(project: ProjectSummary): string | null {
  if (project.gitState._tag !== "repository") return null
  return project.gitState.head._tag === "branch"
    ? project.gitState.head.name
    : project.gitState.head.revision
}

function SessionRow({
  session,
  selected,
  onSelect,
  onClose,
}: {
  readonly session: SessionItemData
  readonly selected: boolean
  readonly onSelect: () => void
  readonly onClose: () => void
}): ReactNode {
  const working = session.workStatus === "working"
  return (
    <div
      data-selected={selected || undefined}
      className="group/session ml-5 flex h-8 items-center rounded-md pr-1 text-slate-600 hover:bg-slate-150 data-[selected]:bg-blue-100 data-[selected]:text-blue-800 dark:text-slate-400 dark:hover:bg-slate-800 dark:data-[selected]:bg-blue-900/55 dark:data-[selected]:text-blue-300"
    >
      <button
        type="button"
        onClick={onSelect}
        title={session.title ?? "Untitled session"}
        className="min-w-0 flex-1 cursor-pointer overflow-hidden text-ellipsis whitespace-nowrap border-0 bg-transparent px-2 text-left font-sans text-[13px] font-medium text-inherit"
      >
        {session.title || "Untitled session"}
      </button>
      {working ? (
        <span className="mr-1 size-1.5 shrink-0 rounded-full bg-blue-500" title="Working" />
      ) : (
        <span className="mr-1 hidden shrink-0 text-[10px] text-slate-500 group-hover/session:inline group-focus-within/session:inline">{formatRelativeTime(session.updatedAt)}</span>
      )}
      {session.sidebarOpen ? (
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation()
            onClose()
          }}
          className="hidden size-6 shrink-0 cursor-pointer items-center justify-center rounded border-0 bg-transparent text-slate-500 hover:bg-slate-250 hover:text-slate-800 group-hover/session:flex group-focus-within/session:flex dark:hover:bg-slate-700 dark:hover:text-slate-200"
          aria-label={`Close ${session.title || "session"}`}
          title="Close session"
        >
          <X size={13} />
        </button>
      ) : null}
    </div>
  )
}

export function SessionsSidebar({
  projects = [],
  sessions = [],
  loading = false,
  loadingMore = false,
  hasMore = false,
  onSelectSession,
  onCloseSession,
  onCompose,
  onCreateProject,
  onEditProject,
  onRemoveProject,
  revealKind = "unsupported",
  onRevealProject,
  onLoadMore,
  onOpenSettings,
  settingsTab = null,
  onSettingsTabChange,
  onCloseSettings,
  overlay = false,
  onCloseOverlay,
  titlebarIntegrated = false,
}: SessionsSidebarProps): ReactNode {
  const selectedSessionId = useSelectedSessionId()
  const search = useAtomValue(sidebarSearchAtom)
  const setSearch = useAtomSet(sidebarSearchAtom)
  const sidebarWidth = useAtomValue(sidebarWidthAtom)
  const setSidebarWidth = useAtomSet(sidebarWidthAtom)
  const collapsed = useAtomValue(sidebarCollapsedAtom)
  const setCollapsed = useAtomSet(sidebarCollapsedAtom)
  const collapsedProjects = useAtomValue(collapsedProjectIdsAtom)
  const setCollapsedProjects = useAtomSet(collapsedProjectIdsAtom)
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null)
  const [projectMenu, setProjectMenu] = useState<ProjectMenuState | null>(null)
  const [formProject, setFormProject] = useState<ProjectRecord | "new" | null>(null)
  const [removeProject, setRemoveProject] = useState<ProjectRecord | null>(null)
  const compact = collapsed && !overlay && settingsTab === null
  const floatingFooter = compact && titlebarIntegrated

  const sessionsByProject = useMemo(() => {
    const grouped = new Map<ProjectId, SessionItemData[]>()
    for (const session of sessions) {
      const current = grouped.get(session.projectId) ?? []
      current.push(session)
      grouped.set(session.projectId, current)
    }
    return grouped
  }, [sessions])
  const orderedProjects = useMemo(() => [...projects].sort((left, right) =>
    left.project.createdAt - right.project.createdAt ||
    left.project.name.localeCompare(right.project.name) ||
    left.project.projectId.localeCompare(right.project.projectId),
  ), [projects])
  const normalizedSearch = search.trim().toLowerCase()
  const visibleProjects = normalizedSearch
    ? orderedProjects.filter((project) =>
        project.project.name.toLowerCase().includes(normalizedSearch) ||
        (sessionsByProject.get(project.project.projectId)?.length ?? 0) > 0,
      )
    : orderedProjects

  const toggleProject = (projectId: ProjectId) => {
    setCollapsedProjects((current) => {
      const next = new Set(current)
      if (next.has(projectId)) next.delete(projectId)
      else next.add(projectId)
      return next
    })
  }

  const handleResizeStart = useCallback((event: React.MouseEvent) => {
    event.preventDefault()
    const startX = event.clientX
    const startWidth = sidebarWidth
    dragRef.current = { startX, startWidth }
    const onMove = (move: MouseEvent) => setSidebarWidth(Math.min(400, Math.max(220, startWidth + move.clientX - startX)))
    const onUp = () => {
      dragRef.current = null
      document.removeEventListener("mousemove", onMove)
      document.removeEventListener("mouseup", onUp)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }
    document.addEventListener("mousemove", onMove)
    document.addEventListener("mouseup", onUp)
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
  }, [setSidebarWidth, sidebarWidth])

  const effectiveWidth = floatingFooter ? 0 : compact ? 48 : overlay ? 280 : sidebarWidth
  return (
    <>
      {overlay ? <div onClick={onCloseOverlay} className="fixed inset-0 z-[79] bg-black/70" /> : null}
      <aside
        data-overlay={overlay || undefined}
        style={{ width: effectiveWidth }}
        className={`${overlay ? "fixed inset-y-0 left-0 z-80 animate-[slide-in-left_200ms_ease-out]" : "relative shrink-0"} max-[640px]:[&:not([data-overlay])]:hidden flex flex-col overflow-hidden border-r border-slate-200 bg-slate-100 transition-[width] duration-150 dark:border-slate-800 dark:bg-slate-850`}
      >
        {settingsTab !== null ? (
          <SettingsNavigation activeTab={settingsTab} onTabChange={onSettingsTabChange} onBack={onCloseSettings} />
        ) : compact ? (
          <div className="flex flex-1 flex-col items-center gap-1 pt-2">
            {!titlebarIntegrated ? (
              <>
                <button type="button" onClick={() => setCollapsed(false)} className="flex size-8 items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800" aria-label="Expand sidebar" title="Expand sidebar"><SidebarSimple size={18} /></button>
                <button type="button" onClick={onCompose} className="flex size-8 items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800" aria-label="New chat" title="New chat"><NotePencil size={18} /></button>
              </>
            ) : null}
          </div>
        ) : (
          <>
            <div className="mac:[-webkit-app-region:no-drag] shrink-0 px-2.5 pb-2 pt-2">
              {!titlebarIntegrated ? (
                <div className="mb-2 flex h-8 items-center justify-end gap-1">
                  <button type="button" onClick={() => setCollapsed(true)} className="flex size-8 items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800" aria-label="Collapse sidebar" title="Collapse sidebar"><SidebarSimple size={18} /></button>
                  <button type="button" onClick={onCompose} className="flex size-8 items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800" aria-label="New chat" title="New chat"><NotePencil size={18} /></button>
                </div>
              ) : null}
              <div className="mb-2 flex h-8 items-center gap-2 rounded-md border border-slate-300 bg-white px-2 dark:border-slate-750 dark:bg-slate-900">
                <MagnifyingGlass size={15} className="shrink-0 text-slate-500" />
                <input id="sidebar-search-input" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search sessions" className="min-w-0 flex-1 border-0 bg-transparent font-sans text-[13px] text-slate-900 outline-none dark:text-slate-100" />
                {search ? <button type="button" onClick={() => setSearch("")} aria-label="Clear search" className="flex size-5 items-center justify-center border-0 bg-transparent text-slate-500"><X size={12} /></button> : null}
              </div>
              <button type="button" onClick={() => setFormProject("new")} className="flex h-8 w-full items-center gap-2 rounded-md border-0 bg-transparent px-2 text-left font-sans text-[13px] font-semibold text-slate-700 hover:bg-white dark:text-slate-300 dark:hover:bg-slate-800">
                <FolderPlus size={17} className="text-slate-500" />
                New project
              </button>
            </div>
            <div
              className="mac:[-webkit-app-region:no-drag] min-h-0 flex-1 overflow-y-auto px-2 pb-3"
              onScroll={(event) => {
                if (!hasMore || loading || loadingMore) return
                const element = event.currentTarget
                if (element.scrollHeight - element.scrollTop - element.clientHeight < 96) onLoadMore?.()
              }}
            >
              {loading ? <SidebarLoadingState /> : visibleProjects.length === 0 ? <SidebarEmptyState searchQuery={search} /> : visibleProjects.map((summary) => {
                const project = summary.project
                const projectSessions = sessionsByProject.get(project.projectId) ?? []
                const isCollapsed = collapsedProjects.has(project.projectId)
                const gitLabel = projectGitLabel(summary)
                const sourceWarning = summary.directoryState._tag === "missing"
                  ? "Missing"
                  : summary.directoryState._tag === "inaccessible"
                    ? "Unavailable"
                    : null
                return (
                  <section key={project.projectId} className="mt-2">
                    <div className="group/project flex h-8 items-center rounded-md text-slate-700 hover:bg-slate-150 dark:text-slate-300 dark:hover:bg-slate-800">
                      <button type="button" onClick={() => toggleProject(project.projectId)} className="flex min-w-0 flex-1 items-center gap-1.5 border-0 bg-transparent px-1.5 text-left text-inherit">
                        {isCollapsed ? <CaretRight size={13} className="shrink-0 text-slate-500" /> : <CaretDown size={13} className="shrink-0 text-slate-500" />}
                        <FolderOpen size={15} weight="regular" className="shrink-0 text-slate-500" />
                        <span className="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap font-sans text-[13px] font-semibold">{project.name}</span>
                        {sourceWarning ? (
                          <span className="shrink-0 font-sans text-[10px] font-semibold text-orange-700 dark:text-orange-400">{sourceWarning}</span>
                        ) : gitLabel ? (
                          <span className="max-w-[72px] overflow-hidden text-ellipsis whitespace-nowrap font-sans text-[10px] text-slate-500">{gitLabel}</span>
                        ) : null}
                      </button>
                      <button
                        type="button"
                        onClick={(event) => {
                          const rect = event.currentTarget.getBoundingClientRect()
                          const menuWidth = 170
                          const menuHeight = 112
                          setProjectMenu({
                            project,
                            x: Math.min(
                              window.innerWidth - menuWidth - 8,
                              Math.max(8, rect.right - menuWidth),
                            ),
                            y: rect.bottom + menuHeight <= window.innerHeight
                              ? rect.bottom + 3
                              : Math.max(8, rect.top - menuHeight - 3),
                          })
                        }}
                        aria-label={`Project options for ${project.name}`}
                        className="mr-1 flex size-6 shrink-0 items-center justify-center rounded border-0 bg-transparent text-slate-500 opacity-0 hover:bg-slate-250 focus:opacity-100 group-hover/project:opacity-100 group-focus-within/project:opacity-100 data-[open=true]:opacity-100 dark:hover:bg-slate-700"
                        data-open={projectMenu?.project.projectId === project.projectId || undefined}
                      >
                        <DotsThreeVertical size={16} weight="bold" />
                      </button>
                    </div>
                    {!isCollapsed ? (
                      <div className="mt-0.5 space-y-0.5">
                        {projectSessions.map((session) => (
                          <SessionRow key={session.sessionId} session={session} selected={selectedSessionId === session.sessionId} onSelect={() => { onSelectSession?.(session); if (overlay) onCloseOverlay?.() }} onClose={() => onCloseSession?.(session.sessionId)} />
                        ))}
                        {projectSessions.length === 0 ? <div className="ml-7 px-2 py-1 text-[11px] text-slate-500">No open sessions</div> : null}
                      </div>
                    ) : null}
                  </section>
                )
              })}
              {loadingMore ? <div className="py-3 text-center text-[11px] text-slate-500">Loading…</div> : null}
            </div>
          </>
        )}
        {!floatingFooter ? <SidebarFooter compact={compact} settingsActive={settingsTab !== null} onOpenSettings={() => { setCollapsed(false); onOpenSettings?.() }} /> : null}
      </aside>
      {floatingFooter ? <SidebarFooter compact floating settingsActive={false} onOpenSettings={() => { setCollapsed(false); onOpenSettings?.() }} /> : null}
      {!overlay && !compact ? <div className="absolute inset-y-0 -right-1 z-10 w-2 cursor-col-resize max-[640px]:hidden" onMouseDown={handleResizeStart} /> : null}
      {projectMenu ? (
        <ProjectMenu
          state={projectMenu}
          onDismiss={() => setProjectMenu(null)}
          onEdit={() => { setFormProject(projectMenu.project); setProjectMenu(null) }}
          onRemove={() => { setRemoveProject(projectMenu.project); setProjectMenu(null) }}
          revealKind={revealKind}
          onReveal={() => {
            onRevealProject?.(projectMenu.project.projectId)
            setProjectMenu(null)
          }}
        />
      ) : null}
      {formProject ? (
        <ProjectFormDialog
          project={formProject === "new" ? undefined : formProject}
          onDismiss={() => setFormProject(null)}
          onSaved={(project) => {
            const created = formProject === "new"
            setFormProject(null)
            if (created) {
              setCollapsedProjects((current) => {
                if (!current.has(project.projectId)) return current
                const next = new Set(current)
                next.delete(project.projectId)
                return next
              })
              onCreateProject?.(project)
            } else {
              onEditProject?.(project)
            }
          }}
        />
      ) : null}
      {removeProject ? <RemoveProjectDialog project={removeProject} onDismiss={() => setRemoveProject(null)} onRemoved={() => { onRemoveProject?.(removeProject); setRemoveProject(null) }} /> : null}
    </>
  )
}
