/**
 * SessionsSidebar — spec §8.2
 *
 * Persistent left sidebar with new-session button, search input,
 * and session list. Session list items show status dot, title,
 * cwd, message count, and relative time.
 *
 * Features:
 * - Resizable via drag handle (§8.1): 200px min, 400px max, 260px default
 * - Right-click context menu (§8.2): Rename (placeholder), Delete
 * - Responsive overlay mode (§12): ≤640px sidebar becomes overlay
 */
import { useCallback, useMemo, useRef, useState, type ReactNode } from "react"
import {
  ArrowLeft,
  ChevronDown,
  GripVertical,
  HardDrive,
  Layers3,
  Loader2,
  Monitor,
  Moon,
  Pencil,
  Plus,
  Search,
  Settings,
  SlidersHorizontal,
  Sun,
  Trash2,
  X,
} from "lucide-react"
import { useAtomValue, useAtomSet } from "@effect-atom/atom-react"
import {
  formatCwdForDisplay,
  formatRelativeTime,
} from "@magnitudedev/client-common"
import {
  useAgentClient,
  useSelectedSessionId,
  useSessionActions,
} from "@magnitudedev/client-common"
import {
  sidebarSearchAtom,
  sidebarWidthAtom,
  type SettingsTab,
} from "../state/web-atoms"
import { SidebarEmptyState, SidebarLoadingState } from "./sidebar-states"
import {
  setAppearancePreference,
  useAppearancePreference,
} from "../stores/appearance-store"

// ── Types ──

interface SessionItemData {
  sessionId: string
  title: string | null
  cwd: string
  messageCount: number
  updatedAt: number
  workStatus: "idle" | "working"
  activeWorkerCount: number
}
export interface SessionsSidebarProps {
  sessions?: SessionItemData[]
  loading?: boolean
  loadingMore?: boolean
  hasMore?: boolean
  cwdFilter?: string | null
  cwdOptions?: string[]
  onCwdFilterChange?: (cwd: string | null) => void
  onSelectSession?: (sessionId: string) => void
  onNewSession?: () => void
  onLoadMore?: () => void
  onOpenSettings?: () => void
  settingsTab?: SettingsTab | null
  onSettingsTabChange?: (tab: SettingsTab) => void
  onCloseSettings?: () => void
  /** Overlay mode — sidebar is an overlay (responsive ≤640px) */
  overlay?: boolean
  /** Close overlay sidebar (used in overlay mode) */
  onCloseOverlay?: () => void
}

// ── Context Menu State ──

interface ContextMenuState {
  x: number
  y: number
  sessionId: string
}
const settingsSections = [
  {
    id: "models",
    label: "Models",
    detail: "Runtime & storage",
    icon: Layers3,
  },
  {
    id: "catalog",
    label: "Catalog",
    detail: "Compare & choose",
    icon: SlidersHorizontal,
  },
  {
    id: "hardware",
    label: "Hardware",
    detail: "Capacity & compute",
    icon: HardDrive,
  },
] as const
function SettingsNavigation({
  activeTab,
  onTabChange,
  onBack,
}: {
  activeTab: SettingsTab
  onTabChange?: (tab: SettingsTab) => void
  onBack?: () => void
}): ReactNode {
  return (
    <>
      <button
        type="button"
        onClick={onBack}
        className="mac:[-webkit-app-region:no-drag] w-[calc(100%-16px)] h-9 mx-2 mt-1 mb-[5px] px-2.5 border-0 rounded-[7px] bg-transparent text-slate-600 dark:text-slate-400 flex items-center gap-2 shrink-0 text-left cursor-pointer hover:bg-white dark:hover:bg-slate-875 [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_strong]:text-[14px] [&_strong]:leading-[normal] [&_strong]:font-semibold mac:[-webkit-app-region:drag]"
        aria-label="Back to sessions"
        title="Back to sessions"
      >
        <ArrowLeft size={16} aria-hidden="true" />
        <strong>Settings</strong>
      </button>
      <nav
        className="flex min-h-0 flex-1 flex-col gap-[3px] px-2 py-2.5"
        aria-label="Settings sections"
      >
        {settingsSections.map(({ id, label, detail, icon: Icon }) => (
          <button
            key={id}
            type="button"
            className="w-full min-h-[52px] px-2.5 py-2 border-0 rounded-[7px] bg-transparent text-slate-500 flex items-center gap-[11px] text-left cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-600 dark:hover:text-slate-400 aria-[current=page]:bg-slate-100 dark:aria-[current=page]:bg-slate-800 aria-[current=page]:text-blue-700 dark:aria-[current=page]:text-blue-500 [&>span]:min-w-0 [&>span]:flex [&>span]:flex-col [&>span]:gap-px [&_strong]:text-slate-600 dark:[&_strong]:text-slate-400 [&_strong]:text-[13px] [&_strong]:font-semibold hover:[&_strong]:text-slate-900 dark:hover:[&_strong]:text-slate-200 aria-[current=page]:[&_strong]:text-slate-900 dark:aria-[current=page]:[&_strong]:text-slate-200 [&_small]:text-slate-500 [&_small]:text-[11px]"
            aria-current={activeTab === id ? "page" : undefined}
            onClick={() => onTabChange?.(id)}
          >
            <Icon size={17} aria-hidden="true" />
            <span>
              <strong>{label}</strong>
              <small>{detail}</small>
            </span>
          </button>
        ))}
      </nav>
    </>
  )
}
function SidebarFooter({
  settingsActive,
  onOpenSettings,
}: {
  settingsActive: boolean
  onOpenSettings?: () => void
}): ReactNode {
  const appearance = useAppearancePreference()
  const nextAppearance =
    appearance === "system"
      ? "light"
      : appearance === "light"
      ? "dark"
      : "system"
  const AppearanceIcon =
    appearance === "light" ? Sun : appearance === "dark" ? Moon : Monitor
  const appearanceLabel =
    appearance === "light" ? "Light" : appearance === "dark" ? "Dark" : "System"
  return (
    <div className="shrink-0 min-h-[49px] border-t border-slate-200 dark:border-slate-800 mac:border-slate-300/[.72] dark:mac:border-slate-800/[.68] p-2 flex items-center gap-2 mac:[-webkit-app-region:no-drag]">
      <button
        type="button"
        onClick={settingsActive ? undefined : onOpenSettings}
        className="size-8 border-0 rounded-md bg-transparent text-slate-600 dark:text-slate-400 flex items-center justify-center cursor-pointer shrink-0 hover:bg-white dark:hover:bg-slate-875 aria-[current=page]:bg-slate-100 dark:aria-[current=page]:bg-slate-800 aria-[current=page]:text-blue-700 dark:aria-[current=page]:text-blue-500"
        aria-label="Settings"
        aria-current={settingsActive ? "page" : undefined}
        title="Settings"
      >
        <Settings size={16} />
      </button>
      <button
        type="button"
        onClick={() => setAppearancePreference(nextAppearance)}
        className="size-8 border-0 rounded-md bg-transparent text-slate-600 dark:text-slate-400 flex items-center justify-center cursor-pointer shrink-0 hover:bg-white dark:hover:bg-slate-875 aria-[current=page]:bg-slate-100 dark:aria-[current=page]:bg-slate-800 aria-[current=page]:text-blue-700 dark:aria-[current=page]:text-blue-500"
        aria-label={`Theme: ${appearanceLabel}. Change to ${
          nextAppearance === "system"
            ? "System"
            : nextAppearance === "light"
            ? "Light"
            : "Dark"
        }`}
        title={`Theme: ${appearanceLabel}`}
      >
        <AppearanceIcon size={16} />
      </button>
    </div>
  )
}

// ── Context Menu ──

function SessionContextMenu({
  menu,
  onClose,
}: {
  menu: ContextMenuState
  onClose: () => void
}): ReactNode {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const { startNewSession } = useSessionActions()
  const deleteMutation = useAtomSet(client.rpc.mutation("DeleteSession"), {
    mode: "promise",
  })

  // Close on any click outside — attached in onContextMenu, but we also
  // handle it with a backdrop click here
  const handleDelete = useCallback(async () => {
    onClose()
    try {
      await deleteMutation({
        payload: {
          sessionId: menu.sessionId,
        },
        reactivityKeys: ["sessions"],
      })
      if (selectedSessionId === menu.sessionId) {
        startNewSession()
      }
    } catch (err) {
      console.error("[DeleteSession] Failed:", err)
    }
  }, [
    menu.sessionId,
    deleteMutation,
    selectedSessionId,
    startNewSession,
    onClose,
  ])
  return (
    <>
      {/* Invisible backdrop to catch outside clicks */}
      <div
        onClick={onClose}
        onContextMenu={(e) => {
          e.preventDefault()
          onClose()
        }}
        className="fixed [top:0px] [left:0px] [right:0px] [bottom:0px] z-[90]"
      />
      <div
        className="session-context-menu absolute z-30 rounded-md border border-slate-300 dark:border-slate-750 bg-slate-100 dark:bg-slate-800 shadow-[0_4px_24px_rgba(0,0,0,.4)] fixed rounded-[4px] z-[91] [min-width:140px] [padding:4px_0] [animation:fade-in_100ms_ease-out]"
        style={{
          left: menu.x,
          top: menu.y,
        }}
      >
        {/* Rename — placeholder, grayed out */}
        <div className="flex items-center [gap:8px] [padding:6px_12px] font-sans text-[13px] text-slate-500 cursor-default opacity-[0.5]">
          <Pencil size={14} className="text-slate-500" />
          <span>Rename</span>
        </div>

        {/* Delete */}
        <div
          className="bg-transparent hover:bg-red-200 dark:hover:bg-red-800 flex items-center [gap:8px] [padding:6px_12px] font-sans text-[13px] text-red-600 dark:text-red-500 cursor-pointer [transition:background_100ms]"
          onClick={handleDelete}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault()
              handleDelete()
            }
          }}
          tabIndex={0}
          role="menuitem"
        >
          <Trash2 size={14} className="text-red-600 dark:text-red-500" />
          <span>Delete</span>
        </div>
      </div>
    </>
  )
}

// ── Session Item ──

function SessionItem({
  session,
  isSelected,
  onSelect,
  onContextMenu,
}: {
  session: SessionItemData
  isSelected: boolean
  onSelect: () => void
  onContextMenu: (e: React.MouseEvent) => void
}): ReactNode {
  const title = session.title || "Untitled session"
  const isWorking = session.workStatus === "working"
  const statusLabel = isWorking
    ? session.activeWorkerCount && session.activeWorkerCount > 0
      ? `${session.activeWorkerCount} worker${
          session.activeWorkerCount === 1 ? "" : "s"
        }`
      : "Working"
    : "Idle"
  return (
    <div
      className="mb-1.5 flex cursor-pointer rounded px-2.5 py-2 bg-transparent transition-colors duration-100 hover:bg-slate-150 data-[selected=true]:bg-slate-200 dark:hover:bg-slate-800 dark:data-[selected=true]:bg-slate-750"
      data-selected={isSelected}
      data-active={isWorking}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onSelect()
        }
      }}
      onContextMenu={onContextMenu}
      tabIndex={0}
      role="button"
      title={session.title ? `${session.title} — ${session.cwd}` : session.cwd}
    >
      <div className="session-item-body [flex:1] min-w-0 flex flex-col [gap:3px]">
        <div className="flex items-center [gap:8px] min-w-0">
          <span
            className={`${
              session.title
                ? "text-slate-900 dark:text-slate-200"
                : "text-slate-500"
            } ${
              session.title ? "not-italic" : "italic"
            }  session-item-title font-sans text-[14px] font-medium overflow-hidden text-ellipsis whitespace-nowrap [flex:1] min-w-0`}
          >
            {title}
          </span>
          <span className="text-slate-500 font-sans text-[12px] shrink-0">
            {formatRelativeTime(session.updatedAt)}
          </span>
        </div>
        <div className="session-item-meta flex items-center [gap:8px] min-w-0 font-sans text-[12px] text-slate-600 dark:text-slate-400">
          <span className="font-mono overflow-hidden text-ellipsis whitespace-nowrap [flex:1] min-w-0">
            {formatCwdForDisplay(session.cwd, {
              maxLen: 28,
              abbreviateHome: true,
            })}
          </span>
          <span
            className={`${
              isWorking ? "text-blue-700 dark:text-blue-500" : "text-slate-500"
            } ${isWorking ? "font-semibold" : "font-normal"}  shrink-0`}
          >
            {statusLabel}
          </span>
        </div>
      </div>
    </div>
  )
}

// ── Main Sidebar ──

export function SessionsSidebar({
  sessions = [],
  loading = false,
  loadingMore = false,
  hasMore = false,
  cwdFilter = null,
  cwdOptions = [],
  onCwdFilterChange,
  onSelectSession,
  onNewSession,
  onLoadMore,
  onOpenSettings,
  settingsTab = null,
  onSettingsTabChange,
  onCloseSettings,
  overlay = false,
  onCloseOverlay,
}: SessionsSidebarProps): ReactNode {
  const selectedSessionId = useSelectedSessionId()
  const searchQuery = useAtomValue(sidebarSearchAtom)
  const setSearchQuery = useAtomSet(sidebarSearchAtom)
  const sidebarWidth = useAtomValue(sidebarWidthAtom)
  const setSidebarWidth = useAtomSet(sidebarWidthAtom)
  // Context menu state
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null)

  // Drag state ref — tracks active resize drag without useEffect
  const dragRef = useRef<{
    startX: number
    startWidth: number
  } | null>(null)
  const visibleCwdOptions = useMemo(() => {
    if (!cwdFilter || cwdOptions.includes(cwdFilter)) return cwdOptions
    return [cwdFilter, ...cwdOptions]
  }, [cwdFilter, cwdOptions])
  const handleSessionListScroll = useCallback(
    (event: React.UIEvent<HTMLDivElement>) => {
      if (!hasMore || loading || loadingMore) return
      const element = event.currentTarget
      const distanceFromBottom =
        element.scrollHeight - element.scrollTop - element.clientHeight
      if (distanceFromBottom < 96) {
        onLoadMore?.()
      }
    },
    [hasMore, loading, loadingMore, onLoadMore]
  )
  const handleSelect = useCallback(
    (sessionId: string) => {
      onSelectSession?.(sessionId)
      // Close overlay sidebar on selection
      if (overlay && onCloseOverlay) onCloseOverlay()
    },
    [onSelectSession, overlay, onCloseOverlay]
  )

  // ── Resize handle: onMouseDown starts drag, attaches mousemove/mouseup on document ──
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      const startX = e.clientX
      const startWidth = sidebarWidth
      dragRef.current = {
        startX,
        startWidth,
      }
      const onMouseMove = (ev: MouseEvent) => {
        const delta = ev.clientX - startX
        const newWidth = Math.min(400, Math.max(200, startWidth + delta))
        setSidebarWidth(newWidth)
      }
      const onMouseUp = () => {
        dragRef.current = null
        document.removeEventListener("mousemove", onMouseMove)
        document.removeEventListener("mouseup", onMouseUp)
        document.body.style.cursor = ""
        document.body.style.userSelect = ""
      }
      document.addEventListener("mousemove", onMouseMove)
      document.addEventListener("mouseup", onMouseUp)
      document.body.style.cursor = "col-resize"
      document.body.style.userSelect = "none"
    },
    [sidebarWidth, setSidebarWidth]
  )

  // ── Context menu handler ──
  const handleContextMenu = useCallback(
    (e: React.MouseEvent, sessionId: string) => {
      e.preventDefault()
      e.stopPropagation()
      setContextMenu({
        x: e.clientX,
        y: e.clientY,
        sessionId,
      })
    },
    []
  )

  // Width: fixed 280px in overlay mode, otherwise the atom value
  const effectiveWidth = overlay ? 280 : sidebarWidth
  const sidebarStyle: React.CSSProperties = { width: effectiveWidth }
  return (
    <>
      {/* Overlay backdrop — click to close */}
      {overlay && (
        <div
          onClick={onCloseOverlay}
          className="fixed inset-0 z-[79] bg-black/70"
        />
      )}

      <div
        className={`${
          overlay
            ? "fixed inset-y-0 left-0 z-80 animate-[slide-in-left_200ms_ease-out]"
            : "relative shrink-0"
        } max-[640px]:[&:not([data-overlay])]:hidden flex flex-col overflow-hidden border-r border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-875 mac:border-slate-300/[.72] mac:bg-slate-50/[.88] dark:mac:border-slate-800/[.68] dark:mac:bg-slate-875/[.86]`}
        data-overlay={overlay || undefined}
        style={sidebarStyle}
      >
        <div
          className="hidden mac:block mac:h-[38px] mac:shrink-0 mac:[-webkit-app-region:drag]"
          aria-hidden="true"
        />

        {settingsTab !== null ? (
          <SettingsNavigation
            activeTab={settingsTab}
            onTabChange={onSettingsTabChange}
            onBack={onCloseSettings}
          />
        ) : (
          <>
            {/* Header */}
            <div className="mac:[-webkit-app-region:drag] flex shrink-0 flex-col gap-2 border-b border-slate-200 px-3 pt-2 pb-3 dark:border-slate-800">
              <button
                type="button"
                onClick={onNewSession}
                className="flex h-7 w-full cursor-pointer items-center gap-[7px] rounded-[5px] border border-slate-300 bg-white px-2 text-left font-sans text-[13px] font-medium text-slate-600 hover:bg-slate-50 dark:border-slate-750 dark:bg-slate-875 dark:text-slate-400 dark:hover:bg-slate-800"
              >
                <Plus size={15} className="[color:inherit] shrink-0" />
                <span>New session</span>
              </button>

              {/* Search input */}
              <div className="mac:[-webkit-app-region:no-drag] flex h-7 items-center gap-[7px] rounded-[5px] border border-slate-300 bg-white px-2 transition-colors duration-100 dark:border-slate-750 dark:bg-slate-875">
                <Search size={14} className="text-slate-500 shrink-0" />
                <input
                  type="text"
                  id="sidebar-search-input"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search sessions..."
                  className="[flex:1] [background:transparent] border-0 [outline:none] font-sans text-[13px] text-slate-900 dark:text-slate-200"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery("")}
                    aria-label="Clear search"
                    className="[background:transparent] border-0 cursor-pointer [padding:0px] flex items-center text-slate-500"
                  >
                    <X size={13} />
                  </button>
                )}
              </div>

              <div
                className={`${
                  cwdFilter
                    ? "text-slate-900 dark:text-slate-200"
                    : "text-slate-600 dark:text-slate-400"
                } relative h-7 rounded-[5px] border border-slate-300 bg-white dark:border-slate-750 dark:bg-slate-875`}
              >
                <select
                  value={cwdFilter ?? ""}
                  onChange={(e) =>
                    onCwdFilterChange?.(e.target.value ? e.target.value : null)
                  }
                  aria-label="Filter sessions by working directory"
                  className={`${
                    cwdFilter ? "font-mono" : "font-sans"
                  }  w-full h-full [background:transparent] border-0 [color:inherit] text-[13px] [padding:0_28px_0_8px] [outline:none] appearance-none cursor-pointer`}
                >
                  <option value="">All working directories</option>
                  {visibleCwdOptions.map((cwd) => (
                    <option key={cwd} value={cwd}>
                      {formatCwdForDisplay(cwd, {
                        maxLen: 34,
                        abbreviateHome: true,
                      })}
                    </option>
                  ))}
                </select>
                <ChevronDown
                  size={15}
                  aria-hidden="true"
                  className="absolute [right:8px] [top:50%] [transform:translateY(-50%)] text-slate-500 pointer-events-none"
                />
              </div>
            </div>

            {/* Session list */}
            <div
              className="mac:[-webkit-app-region:no-drag] [flex:1] overflow-y-auto [padding:8px_6px]"
              onScroll={handleSessionListScroll}
            >
              {loading ? (
                <SidebarLoadingState />
              ) : sessions.length === 0 ? (
                <SidebarEmptyState searchQuery={searchQuery} />
              ) : (
                <>
                  {sessions.map((session) => (
                    <SessionItem
                      key={session.sessionId}
                      session={session}
                      isSelected={selectedSessionId === session.sessionId}
                      onSelect={() => handleSelect(session.sessionId)}
                      onContextMenu={(e) =>
                        handleContextMenu(e, session.sessionId)
                      }
                    />
                  ))}
                  {loadingMore && (
                    <div className="[height:32px] flex items-center justify-center text-slate-500">
                      <Loader2
                        size={14}
                        className="[animation:spin_1s_linear_infinite]"
                      />
                    </div>
                  )}
                </>
              )}
            </div>
          </>
        )}

        <SidebarFooter
          settingsActive={settingsTab !== null}
          onOpenSettings={onOpenSettings}
        />
      </div>

      {/* Resize handle — only in docked mode, on the right edge of the sidebar */}
      {!overlay && (
        <div
          className="absolute inset-y-0 -right-1 z-10 flex w-2 cursor-col-resize items-center justify-center mac:[-webkit-app-region:no-drag] max-[640px]:hidden"
          onMouseDown={handleResizeStart}
        >
          <GripVertical size={8} className="text-slate-500 opacity-[0.3]" />
        </div>
      )}

      {/* Context menu */}
      {contextMenu && (
        <SessionContextMenu
          menu={contextMenu}
          onClose={() => setContextMenu(null)}
        />
      )}
    </>
  )
}
