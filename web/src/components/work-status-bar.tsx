import { useState, useSyncExternalStore, type ReactNode } from "react"
import { Option } from "effect"
import { ChevronDown, ChevronUp, Circle, Loader2 } from "lucide-react"
import {
  formatElapsedMs,
  getTickSnapshot,
  subscribeNoop,
  subscribeTick,
  type SlotProfile,
  type SlotProfiles,
} from "@magnitudedev/client-common"
import type { DisplayActor, DisplayTasks, TaskAssignee, TaskDisplayRow } from "@magnitudedev/sdk"
import { isRoleId, ROLE_TO_SLOT } from "@magnitudedev/sdk"
import { ContextUsageIndicator } from "./context-usage-indicator"

export interface WorkStatusBarProps {
  rootActor: DisplayActor | null
  actors: Record<string, DisplayActor>
  tasks: DisplayTasks | null
  slotProfiles?: SlotProfiles | null
  onWorkerClick?: (forkId: string) => void
}

function findSlotProfileForRole(
  profiles: SlotProfiles | null | undefined,
  role: string | null | undefined,
): SlotProfile | null {
  if (!profiles || !role || !isRoleId(role)) return null
  return ROLE_TO_SLOT[role] === "primary" ? profiles.primary ?? null : profiles.secondary ?? null
}

function rowsFromTasks(tasks: DisplayTasks | null): TaskDisplayRow[] {
  if (!tasks) return []
  return tasks.order
    .map((id) => tasks.byId[id])
    .filter((row): row is TaskDisplayRow => row != null)
}

function getInteractiveForkId(assignee: TaskAssignee, actors: Record<string, DisplayActor>): string | null {
  if (assignee.kind === "actor") {
    const actor = actors[assignee.actorKey]
    return actor?.kind === "worker" ? assignee.actorKey : null
  }
  if (assignee.kind !== "worker") return null
  if (assignee.variant === "spawning") {
    return assignee.interactiveForkId._tag === "Some" ? assignee.interactiveForkId.value : null
  }
  return null
}

function actorTimer(actor: DisplayActor | undefined): number | null {
  if (!actor) return null
  const { work } = actor
  if (work.phase === "working") {
    return Math.max(0, Date.now() - (work.activeSince ?? Date.now()))
  }
  if (work.lastWorkMs > 0) return work.lastWorkMs
  return null
}

function formatRoleLabel(role: string): string {
  return role.length === 0 ? role : role.charAt(0).toUpperCase() + role.slice(1)
}

function assigneeLabel(assignee: TaskAssignee, actors: Record<string, DisplayActor>): string {
  if (assignee.kind === "none") return "Unassigned"
  if (assignee.kind === "user") return "User"
  if (assignee.kind === "actor") {
    const actor = actors[assignee.actorKey]
    return actor?.role ? formatRoleLabel(actor.role) : (actor?.name ?? "Worker")
  }
  return assignee.label
}

function assigneeStatus(
  assignee: TaskAssignee,
  actors: Record<string, DisplayActor>,
): "working" | "idle" | "spawning" | "killing" | "user" | "none" {
  if (assignee.kind === "none") return "none"
  if (assignee.kind === "user") return "user"
  if (assignee.kind === "actor") {
    if (assignee.taskState === "killing") return "killing"
    return actors[assignee.actorKey]?.work.phase === "working" ? "working" : "idle"
  }
  return assignee.variant
}

function StatusSummary({
  active,
  elapsed,
  completedDuration,
  hasRecentWork,
  activity,
}: {
  active: boolean
  elapsed: number
  completedDuration: number | null
  hasRecentWork: boolean
  activity: DisplayActor["work"]["activity"] | null
}): ReactNode {
  return (
    <>
      <Circle
        size={9}
        fill="currentColor"
        className={active ? "animate-pulse-dot" : undefined}
        style={{ color: active ? "var(--accent-primary)" : "var(--fg-tertiary)", flexShrink: 0 }}
      />
      <span style={{ flexShrink: 0, color: active ? "var(--fg-primary)" : "var(--fg-secondary)" }}>
        {active ? `Working... ${formatElapsedMs(elapsed)}` : hasRecentWork ? `Worked ${formatElapsedMs(completedDuration ?? 0)}` : "Idle"}
      </span>
      {activity && (
        <span style={{ display: "flex", alignItems: "center", gap: 5, minWidth: 0, overflow: "hidden" }}>
          {activity.kind === "tool" && Option.getOrNull(activity.decorator) === "spinner" && (
            <Loader2 size={14} className="animate-spin" style={{ color: "var(--fg-secondary)", flexShrink: 0 }} />
          )}
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {activity.message}
          </span>
        </span>
      )}
    </>
  )
}

function TaskStatusRow({
  row,
  actors,
  slotProfiles,
  onWorkerClick,
}: {
  row: TaskDisplayRow
  actors: Record<string, DisplayActor>
  slotProfiles?: SlotProfiles | null
  onWorkerClick?: (forkId: string) => void
}): ReactNode {
  const assignee = row.assignee
  const status = assigneeStatus(assignee, actors)
  const forkId = getInteractiveForkId(assignee, actors)
  const isInteractive = forkId !== null
  const actor = assignee.kind === "actor" ? actors[assignee.actorKey] : undefined
  const actorProfile = findSlotProfileForRole(slotProfiles, actor?.role)
  const tokenCap = actorProfile?.contextWindow ?? null
  const detailLabel = actor ? (actorProfile?.modelDisplayName ?? status) : status
  const timerMs = actorTimer(actor)
  const isCompleted = row.status === "completed"
  const statusColor = status === "working" || status === "spawning"
    ? "var(--accent-primary)"
    : status === "killing"
      ? "var(--accent-error)"
      : isCompleted
        ? "var(--accent-success)"
        : "var(--fg-tertiary)"

  return (
    <button
      type="button"
      disabled={!isInteractive}
      onClick={() => {
        if (forkId) onWorkerClick?.(forkId)
      }}
      className={isInteractive ? "hover-surface" : undefined}
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr) minmax(80px, 120px) minmax(72px, 110px) minmax(64px, 88px) minmax(54px, 72px)",
        alignItems: "center",
        gap: 8,
        width: "100%",
        minHeight: 30,
        border: "none",
        borderRadius: 4,
        background: "transparent",
        color: "var(--fg-primary)",
        cursor: isInteractive ? "pointer" : "default",
        fontFamily: "var(--font-sans)",
        fontSize: 12,
        textAlign: "left",
        padding: "0 6px",
        opacity: isCompleted ? 0.72 : 1,
      }}
    >
      <span style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <Circle
          size={8}
          fill="currentColor"
          className={status === "working" || status === "spawning" ? "animate-pulse-dot" : undefined}
          style={{ color: statusColor, flexShrink: 0 }}
        />
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {row.title}
        </span>
      </span>
      <span style={{ color: "var(--fg-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {assigneeLabel(assignee, actors)}
      </span>
      <span style={{ color: "var(--fg-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {detailLabel}
      </span>
      <span style={{ display: "flex", justifyContent: "flex-end", minWidth: 0 }}>
        {actor ? (
          <ContextUsageIndicator
            context={actor.context}
            tokenCap={tokenCap}
            size={16}
            strokeWidth={1.7}
            showTokenLabel
            tooltip="native"
          />
        ) : null}
      </span>
      <span style={{ color: "var(--fg-tertiary)", textAlign: "right" }}>
        {timerMs !== null && timerMs > 0 ? formatElapsedMs(timerMs) : ""}
      </span>
    </button>
  )
}

export function WorkStatusBar({
  rootActor,
  actors,
  tasks,
  slotProfiles,
  onWorkerClick,
}: WorkStatusBarProps): ReactNode {
  const [expanded, setExpanded] = useState(false)
  const rows = rowsFromTasks(tasks)

  const rootWork = rootActor?.work ?? null
  const active = rootWork?.phase === "working"
  const activity = rootWork?.activity ?? null
  const anyActorWorking = Object.values(actors).some((actor) => actor.work.phase === "working")

  const tick = useSyncExternalStore(active || anyActorWorking ? subscribeTick : subscribeNoop, getTickSnapshot)
  void tick

  const elapsed = active && rootWork
    ? Math.max(0, Date.now() - (rootWork.activeSince ?? Date.now()))
    : 0
  const completedDuration = !active && rootWork && rootWork.lastWorkMs > 0 ? rootWork.lastWorkMs : null
  const hasRecentWork = completedDuration !== null && completedDuration > 0
  const hasTasks = rows.length > 0

  if (!active && !hasRecentWork && !hasTasks && !activity) return null

  const incompleteCount = tasks?.summary.incompleteCount ?? rows.filter((row) => row.status !== "completed").length
  const taskCountLabel = incompleteCount === 1 ? "1 task" : `${incompleteCount} tasks`
  const visibleRows = rows.slice(0, 10)

  return (
    <div
      style={{
        margin: 0,
        border: "1px solid var(--border-default)",
        borderRadius: 6,
        background: "var(--bg-surface)",
        overflow: "hidden",
        fontFamily: "var(--font-sans)",
        flexShrink: 0,
      }}
    >
      {hasTasks ? (
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          aria-expanded={expanded}
          aria-label={expanded ? "Collapse tasks" : "Expand tasks"}
          title={expanded ? "Collapse tasks" : "Expand tasks"}
          className="hover-surface"
          style={{
            minHeight: 34,
            width: "100%",
            padding: "0 10px",
            display: "flex",
            alignItems: "center",
            gap: 8,
            color: "var(--fg-secondary)",
            fontSize: 13,
            fontFamily: "var(--font-sans)",
            textAlign: "left",
            background: "transparent",
            border: "none",
            borderRadius: 0,
            cursor: "pointer",
          }}
        >
          <StatusSummary
            active={active}
            elapsed={elapsed}
            completedDuration={completedDuration}
            hasRecentWork={hasRecentWork}
            activity={activity}
          />
          <span
            style={{
              marginLeft: "auto",
              display: "flex",
              alignItems: "center",
              gap: 4,
              color: "var(--fg-secondary)",
              fontSize: 12,
              flexShrink: 0,
            }}
          >
            <span>{taskCountLabel}</span>
            {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </span>
        </button>
      ) : (
        <div
          style={{
            minHeight: 34,
            padding: "0 10px",
            display: "flex",
            alignItems: "center",
            gap: 8,
            color: "var(--fg-secondary)",
            fontSize: 13,
          }}
        >
          <StatusSummary
            active={active}
            elapsed={elapsed}
            completedDuration={completedDuration}
            hasRecentWork={hasRecentWork}
            activity={activity}
          />
        </div>
      )}

      {expanded && hasTasks && (
        <div
          style={{
            borderTop: "1px solid var(--border-subtle)",
            padding: "6px",
            display: "flex",
            flexDirection: "column",
            gap: 2,
          }}
        >
          {visibleRows.map((row) => (
            <TaskStatusRow
              key={row.rowId}
              row={row}
              actors={actors}
              slotProfiles={slotProfiles}
              onWorkerClick={onWorkerClick}
            />
          ))}
          {rows.length > visibleRows.length && (
            <div style={{ padding: "4px 6px", color: "var(--fg-tertiary)", fontSize: 12 }}>
              +{rows.length - visibleRows.length} more
            </div>
          )}
        </div>
      )}
    </div>
  )
}
