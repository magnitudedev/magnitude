import { useMemo, useState, type ReactNode } from "react"
import {
  ArrowLeft,
  HardDrive,
  Layers3,
  Monitor,
  Moon,
  SlidersHorizontal,
  Sun,
} from "lucide-react"
import {
  CaretDown,
  CaretRight,
  Archive,
  DotsThreeVertical,
  FolderOpen,
  FolderPlus,
  Gear,
  MagnifyingGlass,
  NotePencil,
  Plus,
  PushPin,
  PushPinSlash,
  SidebarSimple,
  X,
} from "@phosphor-icons/react"
import { useAtomValue, useAtomSet } from "@effect-atom/atom-react"
import type { ProjectId, ProjectRecord, ProjectSummary } from "@magnitudedev/sdk"
import { formatRelativeTime, useSelectedSessionId } from "@magnitudedev/client-common"
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
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { ResizableEdge } from "@/components/ui/resizable-edge"

interface SessionItemData {
  readonly sessionId: string
  readonly projectId: ProjectId
  readonly title: string | null
  readonly updatedAt: number
  readonly workStatus: "idle" | "working"
  readonly cwd: string
  readonly pinnedAt: number | null
}

export interface SessionsSidebarProps {
  readonly projects?: ReadonlyArray<ProjectSummary>
  readonly sessions?: ReadonlyArray<SessionItemData>
  readonly loading?: boolean
  readonly loadingMore?: boolean
  readonly hasMore?: boolean
  readonly onSelectSession?: (session: SessionItemData) => void
  readonly onArchiveSession?: (sessionId: string) => void
  readonly onSetSessionPinned?: (sessionId: string, pinned: boolean) => void
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
  { id: "models", label: "Models", icon: Layers3 },
  {
    id: "catalog",
    label: "Catalog",
    icon: SlidersHorizontal,
  },
  {
    id: "hardware",
    label: "Hardware",
    icon: HardDrive,
  },
  {
    id: "archived",
    label: "Archived Chats",
    icon: Archive,
  },
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
  const appearance = useAppearancePreference()
  const nextAppearance =
    appearance === "system" ? "light" : appearance === "light" ? "dark" : "system"
  const AppearanceIcon = appearance === "light" ? Sun : appearance === "dark" ? Moon : Monitor
  return (
    <>
      <Button
        variant="unstyled"
        size="unstyled"
        type="button"
        onClick={onBack}
        className="mx-2 mb-[5px] mt-1 flex h-9 w-[calc(100%-16px)] shrink-0 cursor-pointer items-center gap-2 rounded-[7px] border-0 bg-transparent px-2.5 text-left text-slate-600 [-webkit-app-region:no-drag] hover:bg-slate-150 dark:text-slate-400 dark:hover:bg-slate-750"
      >
        <ArrowLeft size={16} />
        <strong className="text-[14px] font-semibold text-slate-900 dark:text-slate-200">
          Settings
        </strong>
      </Button>
      <nav
        className="flex min-h-0 flex-1 flex-col gap-[3px] px-2 py-2.5"
        aria-label="Settings sections"
      >
        {settingsSections.map(({ id, label, icon: Icon }) => (
          <Button
            key={id}
            variant="unstyled"
            size="unstyled"
            type="button"
            aria-current={activeTab === id ? "page" : undefined}
            onClick={() => onTabChange?.(id)}
            className="flex h-9 w-full cursor-pointer items-center gap-[11px] rounded-[7px] border-0 bg-transparent px-2.5 text-left text-slate-500 hover:bg-slate-150 aria-[current=page]:bg-slate-200 aria-[current=page]:text-blue-700 dark:hover:bg-slate-800 dark:aria-[current=page]:bg-slate-750 dark:aria-[current=page]:text-blue-400"
          >
            <Icon size={17} />
            <strong className="min-w-0 truncate text-[13px] font-semibold text-slate-700 dark:text-slate-300">
              {label}
            </strong>
          </Button>
        ))}
      </nav>
      <div className="shrink-0 px-2 pb-2 [-webkit-app-region:no-drag]">
        <Button
          variant="unstyled"
          size="unstyled"
          type="button"
          onClick={() => setAppearancePreference(nextAppearance)}
          className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-slate-150 dark:text-slate-400 dark:hover:bg-slate-800"
          aria-label={`Theme: ${appearance}`}
          title={`Theme: ${appearance}`}
        >
          <AppearanceIcon size={16} />
        </Button>
      </div>
    </>
  )
}

function SidebarHeaderActions({
  vertical = false,
  collapsed,
  settingsActive,
  onToggleSettings,
  onToggleSidebar,
  onCompose,
}: {
  readonly vertical?: boolean
  readonly collapsed: boolean
  readonly settingsActive: boolean
  readonly onToggleSettings?: () => void
  readonly onToggleSidebar: () => void
  readonly onCompose?: () => void
}): ReactNode {
  return (
    <div className={`flex gap-1 [-webkit-app-region:no-drag] ${vertical ? "flex-col" : "items-center"}`}>
      <Button
        variant="unstyled"
        size="unstyled"
        type="button"
        onClick={onToggleSettings}
        className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white aria-[current=page]:bg-slate-200 aria-[current=page]:text-blue-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:aria-[current=page]:bg-slate-750 dark:aria-[current=page]:text-blue-400"
        aria-label="Settings"
        aria-current={settingsActive ? "page" : undefined}
        title="Settings"
      >
        <Gear size={17} />
      </Button>
      <Button
        variant="unstyled"
        size="unstyled"
        type="button"
        onClick={onToggleSidebar}
        className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800"
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        <SidebarSimple size={18} />
      </Button>
      <Button
        variant="unstyled"
        size="unstyled"
        type="button"
        onClick={onCompose}
        className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800"
        aria-label="New chat"
        title="New chat"
      >
        <NotePencil size={18} />
      </Button>
    </div>
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
  projectName,
  nested = true,
  selected,
  onSelect,
  onArchive,
  onSetPinned,
}: {
  readonly session: SessionItemData
  readonly projectName: string
  readonly nested?: boolean
  readonly selected: boolean
  readonly onSelect: () => void
  readonly onArchive: () => void
  readonly onSetPinned: (pinned: boolean) => void
}): ReactNode {
  const working = session.workStatus === "working"
  const pinned = session.pinnedAt !== null
  const title = session.title || "Untitled session"
  return (
    <div
      data-selected={selected || undefined}
      className={`group/session relative flex h-8 items-center rounded-md pr-1 text-slate-600 hover:bg-slate-150 data-[selected]:bg-blue-100 data-[selected]:text-blue-800 dark:text-slate-400 dark:hover:bg-slate-800 dark:data-[selected]:bg-blue-900/55 dark:data-[selected]:text-blue-300 ${nested ? "ml-5" : ""}`}
    >
      <Button
        variant="unstyled"
        size="unstyled"
        type="button"
        onClick={onSelect}
        title={`${title} — ${projectName}`}
        className="min-w-0 flex-1 cursor-pointer overflow-hidden text-ellipsis whitespace-nowrap border-0 bg-transparent px-2 text-left font-sans text-[13px] font-medium text-inherit group-hover/session:pr-14 group-focus-within/session:pr-14"
      >
        {title}
      </Button>
      {working ? (
        <span className="mr-1 size-1.5 shrink-0 rounded-full bg-blue-500 group-hover/session:hidden group-focus-within/session:hidden" title="Working" />
      ) : (
        <span className="mr-1 shrink-0 text-[10px] text-slate-500 group-hover/session:hidden group-focus-within/session:hidden">
          {formatRelativeTime(session.updatedAt)}
        </span>
      )}
      <div className="pointer-events-none absolute right-1 flex items-center gap-0.5 opacity-0 group-hover/session:pointer-events-auto group-hover/session:opacity-100 group-focus-within/session:pointer-events-auto group-focus-within/session:opacity-100">
        <Button
          variant="unstyled"
          size="unstyled"
          type="button"
          onClick={(event) => {
            event.stopPropagation()
            onSetPinned(!pinned)
          }}
          className="flex size-6 shrink-0 cursor-pointer items-center justify-center rounded border-0 bg-transparent text-slate-500 hover:bg-slate-250 hover:text-slate-800 dark:hover:bg-slate-700 dark:hover:text-slate-200"
          aria-label={`${pinned ? "Unpin" : "Pin"} ${title}`}
          title={pinned ? "Unpin chat" : "Pin chat"}
        >
          {pinned ? <PushPinSlash size={14} /> : <PushPin size={14} />}
        </Button>
        <Button
          variant="unstyled"
          size="unstyled"
          type="button"
          onClick={(event) => {
            event.stopPropagation()
            onArchive()
          }}
          className="flex size-6 shrink-0 cursor-pointer items-center justify-center rounded border-0 bg-transparent text-slate-500 hover:bg-slate-250 hover:text-slate-800 dark:hover:bg-slate-700 dark:hover:text-slate-200"
          aria-label={`Archive ${title}`}
          title="Archive chat"
        >
          <Archive size={14} />
        </Button>
      </div>
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
  onArchiveSession,
  onSetSessionPinned,
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
  const [formProject, setFormProject] = useState<ProjectRecord | "new" | null>(null)
  const [removeProject, setRemoveProject] = useState<ProjectRecord | null>(null)
  const compact = collapsed && !overlay

  const pinnedSessions = useMemo(
    () => sessions.filter((session) => session.pinnedAt !== null),
    [sessions],
  )
  const pinnedProjectIds = useMemo(
    () => new Set(pinnedSessions.map((session) => session.projectId)),
    [pinnedSessions],
  )
  const projectNames = useMemo(
    () => new Map(projects.map((summary) => [summary.project.projectId, summary.project.name])),
    [projects],
  )
  const sessionsByProject = useMemo(() => {
    const grouped = new Map<ProjectId, SessionItemData[]>()
    for (const session of sessions) {
      if (session.pinnedAt !== null) continue
      const current = grouped.get(session.projectId) ?? []
      current.push(session)
      grouped.set(session.projectId, current)
    }
    return grouped
  }, [sessions])
  const orderedProjects = useMemo(
    () =>
      [...projects].sort(
        (left, right) =>
          left.project.createdAt - right.project.createdAt ||
          left.project.name.localeCompare(right.project.name) ||
          left.project.projectId.localeCompare(right.project.projectId)
      ),
    [projects]
  )
  const normalizedSearch = search.trim().toLowerCase()
  const visibleProjects = normalizedSearch
    ? orderedProjects.filter(
        (project) =>
          project.project.name.toLowerCase().includes(normalizedSearch) ||
          (sessionsByProject.get(project.project.projectId)?.length ?? 0) > 0
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

  const toggleSettings = () => {
    if (settingsTab !== null) {
      onCloseSettings?.()
      return
    }
    setCollapsed(false)
    onOpenSettings?.()
  }

  const effectiveWidth = compact && titlebarIntegrated ? 0 : compact ? 48 : overlay ? 280 : sidebarWidth
  return (
    <>
      {overlay ? (
        <div onClick={onCloseOverlay} className="fixed inset-0 z-[79] bg-black/70" />
      ) : null}
      <aside
        data-overlay={overlay || undefined}
        style={{ width: effectiveWidth }}
        className={`${
          overlay
            ? "fixed inset-y-0 left-0 z-80 animate-[slide-in-left_200ms_ease-out]"
            : "relative shrink-0"
        } max-[640px]:[&:not([data-overlay])]:hidden flex flex-col overflow-hidden border-r border-slate-200 bg-slate-100 transition-[width] duration-150 dark:border-slate-800 dark:bg-slate-850`}
      >
        {!compact && !titlebarIntegrated ? (
          <div className="flex h-11 shrink-0 items-center justify-end px-2.5">
            <SidebarHeaderActions
              collapsed={false}
              settingsActive={settingsTab !== null}
              onToggleSettings={toggleSettings}
              onToggleSidebar={() => setCollapsed(true)}
              onCompose={onCompose}
            />
          </div>
        ) : null}
        {compact ? (
          <div className="flex flex-1 flex-col items-center gap-1 pt-2">
            {!titlebarIntegrated ? (
              <SidebarHeaderActions
                vertical
                collapsed
                settingsActive={settingsTab !== null}
                onToggleSettings={toggleSettings}
                onToggleSidebar={() => setCollapsed(false)}
                onCompose={onCompose}
              />
            ) : null}
          </div>
        ) : settingsTab !== null ? (
          <SettingsNavigation
            activeTab={settingsTab}
            onTabChange={onSettingsTabChange}
            onBack={onCloseSettings}
          />
        ) : (
          <>
            <div className="mac:[-webkit-app-region:no-drag] shrink-0 px-2.5 pb-2 pt-2">
              <div className="mb-2 flex h-8 items-center gap-2 rounded-md border border-slate-300 bg-white px-2 dark:border-slate-750 dark:bg-slate-900">
                <MagnifyingGlass size={15} className="shrink-0 text-slate-500" />
                <Input
                  id="sidebar-search-input"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search sessions"
                  className="h-full min-w-0 flex-1 border-0 bg-transparent p-0 text-[13px] text-slate-900 shadow-none focus-visible:ring-0 dark:bg-transparent dark:text-slate-100"
                />
                {search ? (
                  <Button
                    variant="unstyled"
                    size="unstyled"
                    type="button"
                    onClick={() => setSearch("")}
                    aria-label="Clear search"
                    className="flex size-5 items-center justify-center border-0 bg-transparent text-slate-500"
                  >
                    <X size={12} />
                  </Button>
                ) : null}
              </div>
              <Button
                variant="unstyled"
                size="unstyled"
                type="button"
                onClick={() => setFormProject("new")}
                className="flex h-8 w-full items-center gap-2 rounded-md border-0 bg-transparent px-2 text-left font-sans text-[13px] font-semibold text-slate-700 hover:bg-white dark:text-slate-300 dark:hover:bg-slate-800"
              >
                <FolderPlus size={17} className="text-slate-500" />
                <span className="min-w-0 flex-1">New project</span>
                <Plus size={15} aria-hidden="true" className="shrink-0 text-slate-500" />
              </Button>
            </div>
            <div
              className="mac:[-webkit-app-region:no-drag] min-h-0 flex-1 overflow-y-auto px-2 pb-3"
              onScroll={(event) => {
                if (!hasMore || loading || loadingMore) return
                const element = event.currentTarget
                if (element.scrollHeight - element.scrollTop - element.clientHeight < 96)
                  onLoadMore?.()
              }}
            >
              {loading ? (
                <SidebarLoadingState />
              ) : visibleProjects.length === 0 && pinnedSessions.length === 0 ? (
                <SidebarEmptyState searchQuery={search} />
              ) : (
                <>
                  {pinnedSessions.length > 0 ? (
                    <section className="mb-1 mt-1" aria-label="Pinned sessions">
                      <div className="flex h-7 items-center px-2 font-sans text-[11px] font-semibold text-slate-500">
                        Pinned
                      </div>
                      <div className="space-y-0.5">
                        {pinnedSessions.map((session) => (
                          <SessionRow
                            key={session.sessionId}
                            session={session}
                            projectName={projectNames.get(session.projectId) ?? "Project"}
                            nested={false}
                            selected={selectedSessionId === session.sessionId}
                            onSelect={() => {
                              onSelectSession?.(session)
                              if (overlay) onCloseOverlay?.()
                            }}
                            onArchive={() => onArchiveSession?.(session.sessionId)}
                            onSetPinned={(pinned) => onSetSessionPinned?.(session.sessionId, pinned)}
                          />
                        ))}
                      </div>
                    </section>
                  ) : null}
                  {visibleProjects.map((summary) => {
                    const project = summary.project
                    const projectSessions = sessionsByProject.get(project.projectId) ?? []
                    const isCollapsed = collapsedProjects.has(project.projectId)
                    const gitLabel = projectGitLabel(summary)
                    const sourceWarning =
                      summary.directoryState._tag === "missing"
                        ? "Missing"
                        : summary.directoryState._tag === "inaccessible"
                        ? "Unavailable"
                        : null
                    const revealLabel =
                      revealKind === "finder" ? "Reveal in Finder" : "Show in folder"
                    return (
                      <Collapsible
                        key={project.projectId}
                        open={!isCollapsed}
                        onOpenChange={() => toggleProject(project.projectId)}
                        className="mt-2"
                      >
                      <div className="group/project flex h-8 items-center rounded-md text-slate-700 hover:bg-slate-150 dark:text-slate-300 dark:hover:bg-slate-800">
                        <CollapsibleTrigger className="flex min-w-0 flex-1 items-center gap-1.5 border-0 bg-transparent px-1.5 text-left text-inherit">
                          {isCollapsed ? (
                            <CaretRight size={13} className="shrink-0 text-slate-500" />
                          ) : (
                            <CaretDown size={13} className="shrink-0 text-slate-500" />
                          )}
                          <FolderOpen
                            size={15}
                            weight="regular"
                            className="shrink-0 text-slate-500"
                          />
                          <span className="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap font-sans text-[13px] font-semibold">
                            {project.name}
                          </span>
                          {sourceWarning ? (
                            <span className="shrink-0 font-sans text-[10px] font-semibold text-orange-700 dark:text-orange-400">
                              {sourceWarning}
                            </span>
                          ) : gitLabel ? (
                            <span className="max-w-[72px] overflow-hidden text-ellipsis whitespace-nowrap font-sans text-[10px] text-slate-500">
                              {gitLabel}
                            </span>
                          ) : null}
                        </CollapsibleTrigger>
                        <DropdownMenu>
                          <DropdownMenuTrigger
                            render={
                              <Button
                                variant="ghost"
                                size="icon-xs"
                                aria-label={`Project options for ${project.name}`}
                                className="mr-1 opacity-0 group-hover/project:opacity-100 group-focus-within/project:opacity-100 data-popup-open:opacity-100"
                              />
                            }
                          >
                            <DotsThreeVertical size={16} weight="bold" />
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="min-w-[170px]">
                            <DropdownMenuItem onClick={() => setFormProject(project)}>
                              Edit project
                            </DropdownMenuItem>
                            {revealKind !== "unsupported" ? (
                              <DropdownMenuItem
                                onClick={() => onRevealProject?.(project.projectId)}
                              >
                                {revealLabel}
                              </DropdownMenuItem>
                            ) : null}
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              variant="destructive"
                              onClick={() => setRemoveProject(project)}
                            >
                              Remove project
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                      <CollapsibleContent>
                        <div className="mt-0.5 space-y-0.5">
                          {projectSessions.map((session) => (
                            <SessionRow
                              key={session.sessionId}
                              session={session}
                              projectName={project.name}
                              selected={selectedSessionId === session.sessionId}
                              onSelect={() => {
                                onSelectSession?.(session)
                                if (overlay) onCloseOverlay?.()
                              }}
                              onArchive={() => onArchiveSession?.(session.sessionId)}
                              onSetPinned={(pinned) => onSetSessionPinned?.(session.sessionId, pinned)}
                            />
                          ))}
                          {projectSessions.length === 0 ? (
                            <div className="ml-7 px-2 py-1 text-[11px] text-slate-500">
                              {pinnedProjectIds.has(project.projectId) ? "No other sessions" : "No sessions"}
                            </div>
                          ) : null}
                        </div>
                      </CollapsibleContent>
                      </Collapsible>
                    )
                  })}
                </>
              )}
              {loadingMore ? (
                <div className="py-3 text-center text-[11px] text-slate-500">Loading…</div>
              ) : null}
            </div>
          </>
        )}
        {!overlay && !compact ? (
          <ResizableEdge
            side="right"
            value={sidebarWidth}
            minimum={220}
            maximum={400}
            onValueChange={setSidebarWidth}
            label="Resize sessions sidebar"
            className="max-[640px]:hidden"
          />
        ) : null}
      </aside>
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
      {removeProject ? (
        <RemoveProjectDialog
          project={removeProject}
          onDismiss={() => setRemoveProject(null)}
          onRemoved={() => {
            onRemoveProject?.(removeProject)
            setRemoveProject(null)
          }}
        />
      ) : null}
    </>
  )
}
