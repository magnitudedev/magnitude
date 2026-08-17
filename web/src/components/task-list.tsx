/**
 * TaskList — spec §9.5
 *
 * Rendered above composer when DisplayState has tasks.
 * Task rows with assignee status, expand/collapse, worker timers.
 */
import { useState, useSyncExternalStore } from "react"
import { ChevronDown, ChevronUp, Circle, Plus, X } from "lucide-react"
import { formatElapsedMs } from "@magnitudedev/client-common"
import type {
  TaskDisplayRow,
  TaskAssignee,
  DisplayTasks,
} from "@magnitudedev/sdk"
import {
  subscribeTick,
  getTickSnapshot,
  subscribeNoop,
} from "@magnitudedev/client-common"
export interface TaskListProps {
  tasks: DisplayTasks | null
  /** Callback when a worker row is clicked (open worker detail) */
  onWorkerClick?: (forkId: string) => void
}
export function TaskList({
  tasks,
  onWorkerClick,
}: TaskListProps): React.ReactNode {
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
    return (
      a.kind === "worker" || (a.kind === "actor" && a.taskState === "assigned")
    )
  }).length
  const COLLAPSED_LIMIT = 6
  const EXPANDED_LIMIT = 25
  const visibleRows = expanded
    ? rows.slice(0, EXPANDED_LIMIT)
    : rows.slice(0, COLLAPSED_LIMIT)
  const hiddenCount =
    rows.length -
    (expanded
      ? Math.min(rows.length, EXPANDED_LIMIT)
      : Math.min(rows.length, COLLAPSED_LIMIT))
  return (
    <div className="task-list [margin:0_12px_8px] border border-slate-600 dark:border-slate-400 rounded-[4px] [padding:6px_10px] font-mono text-[13px]">
      {/* Header */}
      <div className="task-list-header flex items-center justify-between [margin-bottom:4px]">
        <div className="flex items-center [gap:4px]">
          <span className="font-semibold text-slate-900 dark:text-slate-200">
            Task
          </span>
          <span className="text-slate-600 dark:text-slate-400">
            ({completed} completed, {active} active)
          </span>
        </div>
        <div className="flex items-center [gap:6px]">
          <span className="font-semibold text-slate-900 dark:text-slate-200">
            Assigned To
          </span>
          <button
            onClick={() => setExpanded((v) => !v)}
            title={expanded ? "Collapse" : "Expand"}
            className="[background:transparent] border-0 text-slate-600 dark:text-slate-400 cursor-pointer flex items-center [gap:2px] font-mono text-[12px] [padding:2px_4px] rounded-[3px]"
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            <span>{expanded ? "Collapse" : "Expand"}</span>
          </button>
        </div>
      </div>

      {/* Task rows */}
      <div className="task-list-rows flex flex-col">
        {visibleRows.map((row) => (
          <TaskRow key={row.rowId} row={row} onWorkerClick={onWorkerClick} />
        ))}
      </div>

      {/* Hidden count */}
      {hiddenCount > 0 && (
        <div
          tabIndex={0}
          role="button"
          onClick={() => setExpanded(true)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault()
              setExpanded(true)
            }
          }}
          className="[padding:2px_0] text-slate-500 text-[12px] cursor-pointer"
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
      className={`${`task-row${
        isInteractive
          ? " bg-transparent hover:bg-slate-100 dark:hover:bg-slate-800"
          : ""
      }`} ${
        isInteractive ? "cursor-pointer" : "cursor-default"
      }  [height:22px] flex items-center rounded-[3px] [padding:0_4px] [transition:background_100ms]`}
      onClick={() => {
        if (isInteractive && onWorkerClick) onWorkerClick(interactiveForkId)
      }}
      onKeyDown={
        isInteractive
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault()
                if (onWorkerClick) onWorkerClick(interactiveForkId)
              }
            }
          : undefined
      }
      tabIndex={isInteractive ? 0 : undefined}
      role={isInteractive ? "button" : undefined}
    >
      {/* Name column ~45% */}
      <div className="[flex:0_0_45%] flex items-center [gap:4px] overflow-hidden whitespace-nowrap text-ellipsis">
        <span className="text-slate-500 shrink-0">{indent}</span>
        <span
          className={`${
            isCompleted
              ? "text-green-700 dark:text-green-500"
              : "text-slate-500"
          }  shrink-0`}
        >
          {isCompleted ? "\u2713" : "\u25CB"}
        </span>
        <span
          className={`${
            isCompleted
              ? "text-slate-600 dark:text-slate-400"
              : "text-slate-900 dark:text-slate-200"
          } ${
            isCompleted
              ? "[text-decoration:line-through]"
              : "[text-decoration:none]"
          }  overflow-hidden text-ellipsis`}
        >
          {row.title}
        </span>
      </div>

      {/* Assignee column */}
      <div className="[flex:1] flex items-center [gap:6px] justify-end overflow-hidden">
        <AssigneeCell assignee={assignee} />
      </div>
    </div>
  )
}
function AssigneeCell({
  assignee,
}: {
  assignee: TaskAssignee
}): React.ReactNode {
  if (assignee.kind === "none") return null
  if (assignee.kind === "user") {
    return (
      <span className="text-orange-700 dark:text-orange-500 text-[12px]">
        user
      </span>
    )
  }
  if (assignee.kind === "worker") {
    return (
      <span className="flex items-center [gap:4px]">
        <Plus
          size={12}
          className="animate-pulse-dot text-blue-700 dark:text-blue-500"
        />
        <span className="text-blue-700 dark:text-blue-500">
          {assignee.label}
        </span>
      </span>
    )
  }
  if (assignee.taskState === "killing") {
    return (
      <span className="flex items-center [gap:4px]">
        <X size={12} className="text-red-600 dark:text-red-500" />
        <span className="text-red-600 dark:text-red-500">
          {assignee.actorKey}
        </span>
        {assignee.timer._tag === "Some" && (
          <span className="text-slate-600 dark:text-slate-400">
            {formatElapsedMs(assignee.timer.value)}
          </span>
        )}
      </span>
    )
  }
  return (
    <span className="flex items-center [gap:4px] overflow-hidden">
      <Circle
        size={8}
        fill="currentColor"
        className="text-slate-500 shrink-0"
      />
      <span className="text-slate-900 dark:text-slate-200 overflow-hidden text-ellipsis">
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
    return assignee.interactiveForkId._tag === "Some"
      ? assignee.interactiveForkId.value
      : null
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
    getTickSnapshot
  )
  return tick
}
