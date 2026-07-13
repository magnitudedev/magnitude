/**
 * TaskList — spec §9.5
 *
 * Rendered above composer when DisplayState has tasks.
 * Task rows with assignee status, expand/collapse, worker timers.
 */
import { useState, useSyncExternalStore } from "react"
import { ChevronDown, ChevronUp, Circle, Plus, X } from "lucide-react"
import { formatElapsedMs } from "@magnitudedev/client-common"
import type { TaskDisplayRow, TaskAssignee, DisplayTasks } from "@magnitudedev/sdk"
import { subscribeTick, getTickSnapshot, subscribeNoop } from "@magnitudedev/client-common"

export interface TaskListProps {
  tasks: DisplayTasks | null
  /** Callback when a worker row is clicked (open worker detail) */
  onWorkerClick?: (forkId: string) => void
}

export function TaskList({ tasks, onWorkerClick }: TaskListProps): React.ReactNode {
  const [expanded, setExpanded] = useState(false)

  // Subscribe to tick store so working worker timers update every second.
  // The hook checks if any worker is active and only runs the interval then.
  useTaskTimers(tasks)

  if (!tasks || tasks.order.length === 0) return null

  const rows: TaskDisplayRow[] = tasks.order
    .map((id) => tasks.byId[id])
    .filter((r): r is TaskDisplayRow => r != null)

  if (rows.length === 0) return null

  const completed = rows.filter((r) => r.status === "completed").length
  const active = rows.filter((r) => {
    const a = r.assignee
    return a.kind === "worker" || (a.kind === "actor" && a.taskState === "assigned")
  }).length

  const COLLAPSED_LIMIT = 6
  const EXPANDED_LIMIT = 25
  const visibleRows = expanded ? rows.slice(0, EXPANDED_LIMIT) : rows.slice(0, COLLAPSED_LIMIT)
  const hiddenCount = rows.length - (expanded ? Math.min(rows.length, EXPANDED_LIMIT) : Math.min(rows.length, COLLAPSED_LIMIT))

  return (
    <div
      className="task-list"
      style={{
        margin: "0 12px 8px",
        border: "1px solid var(--line-task)",
        borderRadius: 4,
        padding: "6px 10px",
        fontFamily: "var(--font-mono)",
        fontSize: 13,
      }}
    >
      {/* Header */}
      <div
        className="task-list-header"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 4,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ fontWeight: 600, color: "var(--fg-primary)" }}>Task</span>
          <span style={{ color: "var(--fg-secondary)" }}>
            ({completed} completed, {active} active)
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontWeight: 600, color: "var(--fg-primary)" }}>Assigned To</span>
          <button
            onClick={() => setExpanded((v) => !v)}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--fg-secondary)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 2,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              padding: "2px 4px",
              borderRadius: 3,
            }}
            title={expanded ? "Collapse" : "Expand"}
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            <span>{expanded ? "Collapse" : "Expand"}</span>
          </button>
        </div>
      </div>

      {/* Task rows */}
      <div className="task-list-rows" style={{ display: "flex", flexDirection: "column" }}>
        {visibleRows.map((row) => (
          <TaskRow key={row.rowId} row={row} onWorkerClick={onWorkerClick} />
        ))}
      </div>

      {/* Hidden count */}
      {hiddenCount > 0 && (
        <div
          tabIndex={0}
          role="button"
          style={{
            padding: "2px 0",
            color: "var(--fg-tertiary)",
            fontSize: 12,
            cursor: "pointer",
          }}
          onClick={() => setExpanded(true)}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setExpanded(true) } }}
        >
          +{hiddenCount} more {expanded ? "" : "(show all)"}
        </div>
      )}
    </div>
  )
}

function TaskRow({
  row,
  onWorkerClick,
}: {
  row: TaskDisplayRow
  onWorkerClick?: (forkId: string) => void
}): React.ReactNode {
  const indent = "  ".repeat(row.depth)
  const isCompleted = row.status === "completed"
  const assignee = row.assignee

  const interactiveForkId = getInteractiveForkId(assignee)
  const isInteractive = interactiveForkId !== null

  return (
    <div
      className={`task-row${isInteractive ? " hover-surface" : ""}`}
      style={{
        height: 22,
        display: "flex",
        alignItems: "center",
        borderRadius: 3,
        padding: "0 4px",
        cursor: isInteractive ? "pointer" : "default",
        transition: "background 100ms",
      }}
      onClick={() => {
        if (isInteractive && onWorkerClick) onWorkerClick(interactiveForkId)
      }}
      onKeyDown={isInteractive ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (onWorkerClick) onWorkerClick(interactiveForkId) } } : undefined}
      tabIndex={isInteractive ? 0 : undefined}
      role={isInteractive ? "button" : undefined}
    >
      {/* Name column ~45% */}
      <div
        style={{
          flex: "0 0 45%",
          display: "flex",
          alignItems: "center",
          gap: 4,
          overflow: "hidden",
          whiteSpace: "nowrap",
          textOverflow: "ellipsis",
        }}
      >
        <span style={{ color: "var(--fg-tertiary)", flexShrink: 0 }}>{indent}</span>
        <span style={{ flexShrink: 0, color: isCompleted ? "var(--accent-success)" : "var(--fg-tertiary)" }}>
          {isCompleted ? "\u2713" : "\u25CB"}
        </span>
        <span
          style={{
            color: isCompleted ? "var(--fg-secondary)" : "var(--fg-primary)",
            textDecoration: isCompleted ? "line-through" : "none",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {row.title}
        </span>
      </div>

      {/* Assignee column */}
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          gap: 6,
          justifyContent: "flex-end",
          overflow: "hidden",
        }}
      >
        <AssigneeCell assignee={assignee} />
      </div>
    </div>
  )
}

function AssigneeCell({ assignee }: { assignee: TaskAssignee }): React.ReactNode {
  if (assignee.kind === "none") return null

  if (assignee.kind === "user") {
    return <span style={{ color: "var(--accent-warning)", fontSize: 12 }}>user</span>
  }

  if (assignee.kind === "worker") {
    return (
      <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <Plus
          size={12}
          className="animate-pulse-dot"
          style={{ color: "var(--accent-primary)" }}
        />
        <span style={{ color: "var(--accent-primary)" }}>{assignee.label}</span>
      </span>
    )
  }

  if (assignee.taskState === "killing") {
    return (
      <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <X size={12} style={{ color: "var(--accent-error)" }} />
        <span style={{ color: "var(--accent-error)" }}>{assignee.actorKey}</span>
        {assignee.timer._tag === "Some" && (
          <span style={{ color: "var(--fg-secondary)" }}>
            {formatElapsedMs(assignee.timer.value)}
          </span>
        )}
      </span>
    )
  }

  return (
    <span style={{ display: "flex", alignItems: "center", gap: 4, overflow: "hidden" }}>
      <Circle
        size={8}
        fill="currentColor"
        style={{ color: "var(--fg-tertiary)", flexShrink: 0 }}
      />
      <span style={{ color: "var(--fg-primary)", overflow: "hidden", textOverflow: "ellipsis" }}>
        {assignee.actorKey}
      </span>
    </span>
  )
}

/**
 * Extract interactive forkId from an assignee, if any.
 */
function getInteractiveForkId(assignee: TaskAssignee): string | null {
  if (assignee.kind === "actor") return assignee.actorKey
  if (assignee.kind !== "worker") return null
  if (assignee.variant === "spawning") {
    return assignee.interactiveForkId._tag === "Some" ? assignee.interactiveForkId.value : null
  }
  return null
}

/**
 * Hook to get a live-updating timer for working workers.
 * Re-renders every second when there are active workers.
 */
export function useTaskTimers(tasks: DisplayTasks | null): number {
  const hasActiveWorker =
    tasks?.order.some((id) => {
      const row = tasks.byId[id]
      if (!row) return false
      const a = row.assignee
      return a.kind === "worker"
    }) ?? false

  // Subscribe to tick store only while workers are active
  const tick = useSyncExternalStore(
    hasActiveWorker ? subscribeTick : subscribeNoop,
    getTickSnapshot,
  )
  return tick
}
