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
import type { DirectoryPath, Project, ProjectId } from "@magnitudedev/sdk"
import {
  formatCwdForDisplay,
  formatRelativeTime,
  useProjectPages,
  useSelectedSessionId,
  useSessionPages,
  type RecentChat,
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
import { ActionTooltip } from "@/components/ui/tooltip"

export interface SessionItemData {
  readonly sessionId: string
  readonly title: string | null
  readonly updatedAt: number
  readonly workStatus: "idle" | "working"
  readonly cwd: DirectoryPath
  readonly pinnedAt: number | null
}

export interface SessionLiveStatus {
  readonly workStatus: "idle" | "working"
  readonly lastMessageAt: number
}

export interface SessionsSidebarProps {
  readonly liveStatuses?: Readonly<Record<string, SessionLiveStatus>>
  readonly onSelectSession?: (session: SessionItemData, project: Project | null) => void
  readonly onArchiveSession?: (session: SessionItemData) => void
  readonly onSetSessionPinned?: (sessionId: string, pinned: boolean) => void
  readonly onCompose?: () => void
  readonly onCreateProject?: (project: Project) => void
  readonly onEditProject?: (project: Project) => void
  readonly onRemoveProject?: (project: Project, next: Project | null) => void
  readonly onRevealProject?: (projectId: ProjectId) => void
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
        <ActionTooltip
          label={`Theme: ${appearance}`}
          side="right"
          trigger={
            <Button
              variant="unstyled"
              size="unstyled"
              type="button"
              onClick={() => setAppearancePreference(nextAppearance)}
              className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-slate-150 dark:text-slate-400 dark:hover:bg-slate-800"
              aria-label={`Theme: ${appearance}`}
            >
              <AppearanceIcon size={16} />
            </Button>
          }
        />
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
      <ActionTooltip
        label="Settings"
        side={vertical ? "right" : "bottom"}
        trigger={
          <Button
            variant="unstyled"
            size="unstyled"
            type="button"
            onClick={onToggleSettings}
            className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white aria-[current=page]:bg-slate-200 aria-[current=page]:text-blue-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:aria-[current=page]:bg-slate-750 dark:aria-[current=page]:text-blue-400"
            aria-label="Settings"
            aria-current={settingsActive ? "page" : undefined}
          >
            <Gear size={17} />
          </Button>
        }
      />
      <ActionTooltip
        label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        side={vertical ? "right" : "bottom"}
        trigger={
          <Button
            variant="unstyled"
            size="unstyled"
            type="button"
            onClick={onToggleSidebar}
            className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <SidebarSimple size={18} />
          </Button>
        }
      />
      <ActionTooltip
        label="New chat"
        side={vertical ? "right" : "bottom"}
        trigger={
          <Button
            variant="unstyled"
            size="unstyled"
            type="button"
            onClick={onCompose}
            className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800"
            aria-label="New chat"
          >
            <NotePencil size={18} />
          </Button>
        }
      />
    </div>
  )
}

function SessionRow({
  session,
  contextLabel,
  nested = true,
  selected,
  onSelect,
  onArchive,
  onSetPinned,
}: {
  readonly session: SessionItemData
  readonly contextLabel: string
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
        title={`${title} — ${contextLabel}`}
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
        <ActionTooltip
          label={pinned ? "Unpin" : "Pin"}
          trigger={
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
            >
              {pinned ? <PushPinSlash size={14} /> : <PushPin size={14} />}
            </Button>
          }
        />
        <ActionTooltip
          label="Archive"
          trigger={
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
            >
              <Archive size={14} />
            </Button>
          }
        />
      </div>
    </div>
  )
}

function ShowMoreRow({
  label,
  nested = false,
  loading,
  onActivate,
}: {
  readonly label: string
  readonly nested?: boolean
  readonly loading: boolean
  readonly onActivate: () => void
}): ReactNode {
  return (
    <Button
      variant="unstyled"
      size="unstyled"
      type="button"
      disabled={loading}
      onClick={onActivate}
      className={`flex h-7 w-full cursor-pointer items-center rounded-md px-2 text-left font-sans text-[12px] font-medium text-slate-500 hover:bg-slate-150 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-300 ${nested ? "ml-5 w-[calc(100%-20px)]" : ""}`}
    >
      {loading ? "Loading…" : label}
    </Button>
  )
}

const toSessionItem = (
  chat: RecentChat,
  liveStatuses: Readonly<Record<string, SessionLiveStatus>>,
): SessionItemData => {
  const live = liveStatuses[chat.id]
  return {
    sessionId: chat.id,
    title: chat.title,
    updatedAt: live?.lastMessageAt ?? chat.timestamp,
    workStatus: live?.workStatus ?? "idle",
    cwd: chat.workingDirectory,
    pinnedAt: chat.pinnedAt,
  }
}

interface SessionRowActions {
  readonly selectedSessionId: string | null
  readonly liveStatuses: Readonly<Record<string, SessionLiveStatus>>
  readonly onSelect: (session: SessionItemData, project: Project | null) => void
  readonly onArchive: (session: SessionItemData) => void
  readonly onSetPinned: (sessionId: string, pinned: boolean) => void
}

/** Nested unpinned sessions of one expanded Project: five first, ten more per activation. */
function ProjectSessions({
  project,
  actions,
}: {
  readonly project: Project
  readonly actions: SessionRowActions
}): ReactNode {
  const page = useSessionPages({
    cwd: project.cwd,
    pin: "unpinned",
    archive: "active",
    pageSize: 5,
  })
  return (
    <div className="mt-0.5 space-y-0.5">
      {page.sessions.map((chat) => {
        const session = toSessionItem(chat, actions.liveStatuses)
        return (
          <SessionRow
            key={session.sessionId}
            session={session}
            contextLabel={project.name}
            selected={actions.selectedSessionId === session.sessionId}
            onSelect={() => actions.onSelect(session, project)}
            onArchive={() => actions.onArchive(session)}
            onSetPinned={(pinned) => actions.onSetPinned(session.sessionId, pinned)}
          />
        )
      })}
      {page.hasMore ? (
        <ShowMoreRow
          nested
          label="Show more"
          loading={page.loadingMore}
          onActivate={() => page.loadMore(10)}
        />
      ) : null}
      {!page.loading && page.sessions.length === 0 ? (
        <div className="ml-7 px-2 py-1 text-[11px] text-slate-500">No sessions</div>
      ) : null}
    </div>
  )
}

/** Server-bounded search results grouped under loaded Projects by cwd. */
function SidebarSearchResults({
  query,
  projects,
  actions,
}: {
  readonly query: string
  readonly projects: ReadonlyArray<Project>
  readonly actions: SessionRowActions
}): ReactNode {
  const page = useSessionPages({ archive: "active", query, pageSize: 50 })
  const normalized = query.toLowerCase()
  const byCwd = useMemo(() => {
    const grouped = new Map<string, SessionItemData[]>()
    for (const chat of page.sessions) {
      const session = toSessionItem(chat, actions.liveStatuses)
      const current = grouped.get(session.cwd) ?? []
      current.push(session)
      grouped.set(session.cwd, current)
    }
    return grouped
  }, [page.sessions, actions.liveStatuses])

  const projectCwds = new Set<string>(projects.map((project) => project.cwd))
  const visibleProjects = projects.filter((project) =>
    project.name.toLowerCase().includes(normalized) || byCwd.has(project.cwd))
  const ungrouped = [...byCwd.entries()].filter(([cwd]) => !projectCwds.has(cwd))

  if (page.loading) return <SidebarLoadingState />
  if (visibleProjects.length === 0 && ungrouped.length === 0) {
    return <SidebarEmptyState searchQuery={query} />
  }

  const renderGroup = (
    label: string,
    key: string,
    project: Project | null,
    sessions: ReadonlyArray<SessionItemData>,
  ) => (
    <section key={key} className="mt-2" aria-label={label}>
      <div className="flex h-7 items-center gap-1.5 px-1.5 font-sans text-[13px] font-semibold text-slate-700 dark:text-slate-300">
        <FolderOpen size={15} weight="regular" className="shrink-0 text-slate-500" />
        <span className="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
          {label}
        </span>
      </div>
      <div className="mt-0.5 space-y-0.5">
        {sessions.map((session) => (
          <SessionRow
            key={session.sessionId}
            session={session}
            contextLabel={label}
            selected={actions.selectedSessionId === session.sessionId}
            onSelect={() => actions.onSelect(session, project)}
            onArchive={() => actions.onArchive(session)}
            onSetPinned={(pinned) => actions.onSetPinned(session.sessionId, pinned)}
          />
        ))}
        {sessions.length === 0 ? (
          <div className="ml-7 px-2 py-1 text-[11px] text-slate-500">No matching sessions</div>
        ) : null}
      </div>
    </section>
  )

  return (
    <>
      {visibleProjects.map((project) =>
        renderGroup(project.name, project.projectId, project, byCwd.get(project.cwd) ?? []))}
      {ungrouped.map(([cwd, sessions]) =>
        renderGroup(formatCwdForDisplay(cwd), `cwd:${cwd}`, null, sessions))}
      {page.hasMore ? (
        <ShowMoreRow
          label="Show more results"
          loading={page.loadingMore}
          onActivate={() => page.loadMore(50)}
        />
      ) : null}
    </>
  )
}

export function SessionsSidebar({
  liveStatuses = {},
  onSelectSession,
  onArchiveSession,
  onSetSessionPinned,
  onCompose,
  onCreateProject,
  onEditProject,
  onRemoveProject,
  onRevealProject,
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
  const [formProject, setFormProject] = useState<Project | "new" | null>(null)
  const [removeProject, setRemoveProject] = useState<Project | null>(null)
  const compact = collapsed && !overlay
  const trimmedSearch = search.trim()

  const projectPage = useProjectPages({ pageSize: 20 })
  const pinnedPage = useSessionPages({ pin: "pinned", archive: "active" })
  const projectNamesByCwd = useMemo(
    () => new Map(projectPage.projects.map((project) => [project.cwd as string, project.name])),
    [projectPage.projects],
  )

  const actions: SessionRowActions = {
    selectedSessionId,
    liveStatuses,
    onSelect: (session, project) => {
      onSelectSession?.(session, project)
      if (overlay) onCloseOverlay?.()
    },
    onArchive: (session) => onArchiveSession?.(session),
    onSetPinned: (sessionId, pinned) => onSetSessionPinned?.(sessionId, pinned),
  }

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

  const pinnedSessions = pinnedPage.sessions.map((chat) => toSessionItem(chat, liveStatuses))

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
            <div className="mac:[-webkit-app-region:no-drag] min-h-0 flex-1 overflow-y-auto px-2 pb-3">
              {trimmedSearch ? (
                <SidebarSearchResults
                  query={trimmedSearch}
                  projects={projectPage.projects}
                  actions={actions}
                />
              ) : projectPage.loading ? (
                <SidebarLoadingState />
              ) : projectPage.projects.length === 0 && pinnedSessions.length === 0 ? (
                <SidebarEmptyState searchQuery="" />
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
                            contextLabel={
                              projectNamesByCwd.get(session.cwd) ?? formatCwdForDisplay(session.cwd)
                            }
                            nested={false}
                            selected={selectedSessionId === session.sessionId}
                            onSelect={() => actions.onSelect(
                              session,
                              projectPage.projects.find((project) => project.cwd === session.cwd) ?? null,
                            )}
                            onArchive={() => actions.onArchive(session)}
                            onSetPinned={(pinned) => actions.onSetPinned(session.sessionId, pinned)}
                          />
                        ))}
                      </div>
                    </section>
                  ) : null}
                  {projectPage.projects.map((project) => {
                    const isCollapsed = collapsedProjects.has(project.projectId)
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
                            <DropdownMenuItem
                              onClick={() => onRevealProject?.(project.projectId)}
                            >
                              Reveal folder
                            </DropdownMenuItem>
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
                        {!isCollapsed ? (
                          <ProjectSessions project={project} actions={actions} />
                        ) : null}
                      </CollapsibleContent>
                      </Collapsible>
                    )
                  })}
                  {projectPage.hasMore ? (
                    <div className="mt-2">
                      <ShowMoreRow
                        label="Show more projects"
                        loading={projectPage.loadingMore}
                        onActivate={() => projectPage.loadMore(20)}
                      />
                    </div>
                  ) : null}
                </>
              )}
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
            const next = projectPage.projects.find(
              (candidate) => candidate.projectId !== removeProject.projectId,
            ) ?? null
            onRemoveProject?.(removeProject, next)
            setRemoveProject(null)
          }}
        />
      ) : null}
    </>
  )
}
