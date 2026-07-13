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
import { Loader2, Plus, Search, X, GripVertical, Pencil, Trash2, Settings, BarChart3, ChevronDown } from "lucide-react"
import { useAtomValue, useAtomSet } from "@effect-atom/atom-react"
import { formatCwdForDisplay, formatRelativeTime } from "@magnitudedev/client-common"
import {
  useAgentClient,
  useSelectedSessionId,
  useSessionActions,
} from "@magnitudedev/client-common"
import {
  sidebarSearchAtom,
  sidebarWidthAtom,
  sidebarVisibleAtom,
} from "../state/web-atoms"
import { SidebarEmptyState, SidebarLoadingState } from "./sidebar-states"

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
  accountLabel?: string | null
  accountSubLabel?: string | null
  onCwdFilterChange?: (cwd: string | null) => void
  onSelectSession?: (sessionId: string) => void
  onNewSession?: () => void
  onLoadMore?: () => void
  onOpenSettings?: () => void
  onOpenUsage?: () => void
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
  const deleteMutation = useAtomSet(client.mutation("DeleteSession"), { mode: "promise" })

  // Close on any click outside — attached in onContextMenu, but we also
  // handle it with a backdrop click here
  const handleDelete = useCallback(async () => {
    onClose()
    try {
      await deleteMutation({
        payload: { sessionId: menu.sessionId },
        reactivityKeys: ["sessions"],
      })
      if (selectedSessionId === menu.sessionId) {
        startNewSession()
      }
    } catch (err) {
      console.error("[DeleteSession] Failed:", err)
    }
  }, [menu.sessionId, deleteMutation, selectedSessionId, startNewSession, onClose])

  return (
    <>
      {/* Invisible backdrop to catch outside clicks */}
      <div
        onClick={onClose}
        onContextMenu={(e) => {
          e.preventDefault()
          onClose()
        }}
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 90,
        }}
      />
      <div
        className="session-context-menu popover"
        style={{
          position: "fixed",
          left: menu.x,
          top: menu.y,
          borderRadius: 4,
          zIndex: 91,
          minWidth: 140,
          padding: "4px 0",
          animation: "fade-in 100ms ease-out",
        }}
      >
        {/* Rename — placeholder, grayed out */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 12px",
            fontFamily: "var(--font-sans)",
            fontSize: 13,
            color: "var(--fg-tertiary)",
            cursor: "default",
            opacity: 0.5,
          }}
        >
          <Pencil size={14} style={{ color: "var(--fg-tertiary)" }} />
          <span>Rename</span>
        </div>

        {/* Delete */}
        <div
          className="hover-danger"
          onClick={handleDelete}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleDelete() } }}
          tabIndex={0}
          role="menuitem"
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 12px",
            fontFamily: "var(--font-sans)",
            fontSize: 13,
            color: "var(--accent-error)",
            cursor: "pointer",
            transition: "background 100ms",
          }}
        >
          <Trash2 size={14} style={{ color: "var(--accent-error)" }} />
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
      ? `${session.activeWorkerCount} worker${session.activeWorkerCount === 1 ? "" : "s"}`
      : "Working"
    : "Idle"

  return (
    <div
      className="session-item"
      data-selected={isSelected}
      data-active={isWorking}
      onClick={onSelect}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect() } }}
      onContextMenu={onContextMenu}
      tabIndex={0}
      role="button"
      title={session.title ? `${session.title} — ${session.cwd}` : session.cwd}
    >
      <div className="session-item-body" style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 3 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            minWidth: 0,
          }}
        >
          <span
            className="session-item-title"
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: 14,
              fontWeight: 500,
              color: session.title ? "var(--fg-primary)" : "var(--fg-tertiary)",
              fontStyle: session.title ? "normal" : "italic",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              flex: 1,
              minWidth: 0,
            }}
          >
            {title}
          </span>
          <span
            style={{
              color: "var(--fg-tertiary)",
              fontFamily: "var(--font-sans)",
              fontSize: 12,
              flexShrink: 0,
            }}
          >
            {formatRelativeTime(session.updatedAt)}
          </span>
        </div>
        <div
          className="session-item-meta"
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            minWidth: 0,
            fontFamily: "var(--font-sans)",
            fontSize: 12,
            color: "var(--fg-secondary)",
          }}
        >
          <span style={{ fontFamily: "var(--font-mono)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1, minWidth: 0 }}>
            {formatCwdForDisplay(session.cwd, { maxLen: 28, abbreviateHome: true })}
          </span>
          <span
            style={{
              flexShrink: 0,
              color: isWorking ? "var(--accent-primary)" : "var(--fg-tertiary)",
              fontWeight: isWorking ? 600 : 400,
            }}
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
  accountLabel = "Account",
  accountSubLabel = null,
  onCwdFilterChange,
  onSelectSession,
  onNewSession,
  onLoadMore,
  onOpenSettings,
  onOpenUsage,
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
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null)

  const visibleCwdOptions = useMemo(() => {
    if (!cwdFilter || cwdOptions.includes(cwdFilter)) return cwdOptions
    return [cwdFilter, ...cwdOptions]
  }, [cwdFilter, cwdOptions])

  const handleSessionListScroll = useCallback(
    (event: React.UIEvent<HTMLDivElement>) => {
      if (!hasMore || loading || loadingMore) return
      const element = event.currentTarget
      const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight
      if (distanceFromBottom < 96) {
        onLoadMore?.()
      }
    },
    [hasMore, loading, loadingMore, onLoadMore],
  )

  const handleSelect = useCallback(
    (sessionId: string) => {
      onSelectSession?.(sessionId)
      // Close overlay sidebar on selection
      if (overlay && onCloseOverlay) onCloseOverlay()
    },
    [onSelectSession, overlay, onCloseOverlay],
  )

  // ── Resize handle: onMouseDown starts drag, attaches mousemove/mouseup on document ──
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      const startX = e.clientX
      const startWidth = sidebarWidth
      dragRef.current = { startX, startWidth }

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
    [sidebarWidth, setSidebarWidth],
  )

  // ── Context menu handler ──
  const handleContextMenu = useCallback(
    (e: React.MouseEvent, sessionId: string) => {
      e.preventDefault()
      e.stopPropagation()
      setContextMenu({ x: e.clientX, y: e.clientY, sessionId })
    },
    [],
  )

  // Width: fixed 280px in overlay mode, otherwise the atom value
  const effectiveWidth = overlay ? 280 : sidebarWidth

  const sidebarStyle: React.CSSProperties = overlay
    ? {
        position: "fixed",
        top: 0,
        left: 0,
        bottom: 0,
        width: effectiveWidth,
        background: "var(--bg-sidebar-window, var(--bg-sidebar))",
        borderRight: "1px solid var(--border-sidebar, var(--border-subtle))",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        zIndex: 80,
        animation: "slide-in-left 200ms ease-out",
      }
    : {
        width: effectiveWidth,
        flexShrink: 0,
        background: "var(--bg-sidebar-window, var(--bg-sidebar))",
        borderRight: "1px solid var(--border-sidebar, var(--border-subtle))",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        position: "relative",
      }

  return (
    <>
      {/* Overlay backdrop — click to close */}
      {overlay && (
        <div
          onClick={onCloseOverlay}
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "var(--bg-overlay)",
            zIndex: 79,
          }}
        />
      )}

      <div
        className="sessions-sidebar"
        data-overlay={overlay || undefined}
        style={sidebarStyle}
      >
        <div className="sidebar-window-drag-region" aria-hidden="true" />

        {/* Header */}
        <div
          className="sidebar-header"
          style={{
            padding: "8px 12px 12px",
            borderBottom: "1px solid var(--border-sidebar, var(--border-subtle))",
            flexShrink: 0,
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          <button
            type="button"
            onClick={onNewSession}
            className="hover-surface-flat"
            style={{
              width: "100%",
              height: 28,
              background: "var(--bg-surface)",
              border: "1px solid var(--border-sidebar, var(--border-subtle))",
              borderRadius: 5,
              padding: "0 8px",
              display: "flex",
              alignItems: "center",
              gap: 7,
              color: "var(--fg-secondary)",
              fontFamily: "var(--font-sans)",
              fontSize: 13,
              fontWeight: 500,
              cursor: "pointer",
              textAlign: "left",
            }}
          >
            <Plus size={15} style={{ color: "inherit", flexShrink: 0 }} />
            <span>New session</span>
          </button>

          {/* Search input */}
          <div
            className="search-input-container"
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--border-sidebar, var(--border-subtle))",
              borderRadius: 5,
              height: 28,
              display: "flex",
              alignItems: "center",
              padding: "0 8px",
              gap: 7,
              transition: "border-color 100ms, background 100ms",
            }}
          >
            <Search size={14} style={{ color: "var(--fg-tertiary)", flexShrink: 0 }} />
            <input
              type="text"
              id="sidebar-search-input"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search sessions..."
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                outline: "none",
                fontFamily: "var(--font-sans)",
                fontSize: 13,
                color: "var(--fg-primary)",
              }}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                aria-label="Clear search"
                style={{
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  padding: 0,
                  display: "flex",
                  alignItems: "center",
                  color: "var(--fg-tertiary)",
                }}
              >
                <X size={13} />
              </button>
            )}
          </div>

          <div
            style={{
              position: "relative",
              height: 28,
              background: "var(--bg-surface)",
              border: "1px solid var(--border-sidebar, var(--border-subtle))",
              borderRadius: 5,
              color: cwdFilter ? "var(--fg-primary)" : "var(--fg-secondary)",
            }}
          >
            <select
              value={cwdFilter ?? ""}
              onChange={(e) => onCwdFilterChange?.(e.target.value ? e.target.value : null)}
              aria-label="Filter sessions by working directory"
              style={{
                width: "100%",
                height: "100%",
                background: "transparent",
                border: "none",
                color: "inherit",
                fontFamily: cwdFilter ? "var(--font-mono)" : "var(--font-sans)",
                fontSize: 13,
                padding: "0 28px 0 8px",
                outline: "none",
                appearance: "none",
                cursor: "pointer",
              }}
            >
              <option value="">All working directories</option>
              {visibleCwdOptions.map((cwd) => (
                <option key={cwd} value={cwd}>
                  {formatCwdForDisplay(cwd, { maxLen: 34, abbreviateHome: true })}
                </option>
              ))}
            </select>
            <ChevronDown
              size={15}
              aria-hidden="true"
              style={{
                position: "absolute",
                right: 8,
                top: "50%",
                transform: "translateY(-50%)",
                color: "var(--fg-tertiary)",
                pointerEvents: "none",
              }}
            />
          </div>
        </div>

        {/* Session list */}
        <div
          className="session-list"
          onScroll={handleSessionListScroll}
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "8px 6px",
          }}
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
                  onContextMenu={(e) => handleContextMenu(e, session.sessionId)}
                />
              ))}
              {loadingMore && (
                <div
                  style={{
                    height: 32,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "var(--fg-tertiary)",
                  }}
                >
                  <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
                </div>
              )}
            </>
          )}
        </div>

        <div
          className="sidebar-account"
          style={{
            flexShrink: 0,
            borderTop: "1px solid var(--border-sidebar, var(--border-subtle))",
            padding: "8px",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <button
            type="button"
            onClick={onOpenSettings}
            className="hover-surface-flat"
            style={{
              minWidth: 0,
              flex: 1,
              height: 38,
              border: "none",
              borderRadius: 6,
              background: "transparent",
              color: "var(--fg-primary)",
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "0 6px",
              cursor: "pointer",
              fontFamily: "var(--font-sans)",
              textAlign: "left",
            }}
            aria-label="Account"
            title="Account"
          >
            <span
              aria-hidden="true"
              style={{
                width: 24,
                height: 24,
                borderRadius: "50%",
                border: "1px solid var(--border-default)",
                background: "var(--bg-surface-elevated)",
                color: "var(--fg-secondary)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 11,
                fontWeight: 650,
                flexShrink: 0,
              }}
            >
              {(accountLabel?.trim()?.[0] ?? "A").toUpperCase()}
            </span>
            <span style={{ minWidth: 0, flex: 1 }}>
              <span
                style={{
                  display: "block",
                  color: "var(--fg-primary)",
                  fontSize: 13,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {accountLabel}
              </span>
              {accountSubLabel && (
                <span
                  style={{
                    display: "block",
                    marginTop: 1,
                    color: "var(--fg-tertiary)",
                    fontSize: 11,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {accountSubLabel}
                </span>
              )}
            </span>
            <ChevronDown size={14} style={{ color: "var(--fg-tertiary)", flexShrink: 0 }} />
          </button>

          <button
            type="button"
            onClick={onOpenUsage}
            className="hover-surface-flat"
            aria-label="Usage"
            title="Usage"
            style={{
              width: 32,
              height: 32,
              border: "none",
              borderRadius: 6,
              background: "transparent",
              color: "var(--fg-secondary)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              flexShrink: 0,
            }}
          >
            <BarChart3 size={16} />
          </button>

          <button
            type="button"
            onClick={onOpenSettings}
            className="hover-surface-flat"
            aria-label="Settings"
            title="Settings"
            style={{
              width: 32,
              height: 32,
              border: "none",
              borderRadius: 6,
              background: "transparent",
              color: "var(--fg-secondary)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              flexShrink: 0,
            }}
          >
            <Settings size={16} />
          </button>
        </div>
      </div>

      {/* Resize handle — only in docked mode, on the right edge of the sidebar */}
      {!overlay && (
        <div
          className="sidebar-resize-handle"
          onMouseDown={handleResizeStart}
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            right: -4,
            width: 8,
            cursor: "col-resize",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10,
          }}
        >
          <GripVertical size={8} style={{ color: "var(--fg-tertiary)", opacity: 0.3 }} />
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
