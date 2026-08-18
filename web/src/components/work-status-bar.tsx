import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { useState, useSyncExternalStore, type ReactNode } from "react"
import { Option } from "effect"
import { ChevronDown, ChevronUp, Circle } from "lucide-react"
import {
  formatElapsedMs,
  getTickSnapshot,
  displayRootStatusElapsedMs,
  displayWorkerStatusElapsedMs,
  isDisplayWorkerStatusClockRunning,
  rootDetailSegments,
  subscribeNoop,
  subscribeTick,
  useStabilizedRootDetail,
  type SlotProfile,
  type SlotProfiles,
  type LocalModelLoadActivity,
} from "@magnitudedev/client-common"
import type {
  DisplayActor,
  DisplayRootStatus,
  DisplayTasks,
  TaskAssignee,
  TaskDisplayRow,
} from "@magnitudedev/sdk"
import { isRoleId, ROLE_TO_SLOT } from "@magnitudedev/sdk"
import { ContextUsageIndicator } from "./context-usage-indicator"
export interface WorkStatusBarProps {
  rootActor: DisplayActor | null
  actors: Record<string, DisplayActor>
  tasks: DisplayTasks | null
  slotProfiles?: SlotProfiles | null
  modelLoadActivity?: LocalModelLoadActivity | null
  onWorkerClick?: (forkId: string) => void
}
export function isWorkStatusBarVisible(
  rootStatus: DisplayRootStatus | null,
  hasTasks: boolean
): boolean {
  return rootStatus !== null && (rootStatus._tag === "Working" || hasTasks)
}
function findSlotProfileForRole(
  profiles: SlotProfiles | null | undefined,
  role: string | null | undefined
): SlotProfile | null {
  if (!profiles || !role || !isRoleId(role)) return null
  return ROLE_TO_SLOT[role] === "primary"
    ? profiles.primary ?? null
    : profiles.secondary ?? null
}
function rowsFromTasks(tasks: DisplayTasks | null): TaskDisplayRow[] {
  if (!tasks) return []
  return tasks.order
    .map((id) => tasks.byId[id])
    .filter((row): row is TaskDisplayRow => row != null)
}
function getInteractiveForkId(
  assignee: TaskAssignee,
  actors: Record<string, DisplayActor>
): string | null {
  if (assignee.kind === "actor") {
    const actor = actors[assignee.actorKey]
    return actor?.kind === "worker" ? assignee.actorKey : null
  }
  if (assignee.kind !== "worker") return null
  if (assignee.variant === "spawning") {
    return assignee.interactiveForkId._tag === "Some"
      ? assignee.interactiveForkId.value
      : null
  }
  return null
}
function actorTimer(actor: DisplayActor | undefined): number | null {
  if (actor?.kind !== "worker") return null
  const { status } = actor
  if (isDisplayWorkerStatusClockRunning(status)) {
    return displayWorkerStatusElapsedMs(status, Date.now())
  }
  if (status.lastWorkMs > 0) return status.lastWorkMs
  return null
}
function formatRoleLabel(role: string): string {
  return role.length === 0 ? role : role.charAt(0).toUpperCase() + role.slice(1)
}
function assigneeLabel(
  assignee: TaskAssignee,
  actors: Record<string, DisplayActor>
): string {
  if (assignee.kind === "none") return "Unassigned"
  if (assignee.kind === "user") return "User"
  if (assignee.kind === "actor") {
    const actor = actors[assignee.actorKey]
    return actor?.role ? formatRoleLabel(actor.role) : actor?.name ?? "Worker"
  }
  return assignee.label
}
function assigneeStatus(
  assignee: TaskAssignee,
  actors: Record<string, DisplayActor>
): "working" | "idle" | "spawning" | "killing" | "user" | "none" {
  if (assignee.kind === "none") return "none"
  if (assignee.kind === "user") return "user"
  if (assignee.kind === "actor") {
    if (assignee.taskState === "killing") return "killing"
    const actor = actors[assignee.actorKey]
    return actor?.kind === "worker" && actor.status.phase === "working"
      ? "working"
      : "idle"
  }
  return assignee.variant
}
function StatusSummary({
  status,
  modelLoadActivity,
}: {
  status: DisplayRootStatus
  modelLoadActivity: LocalModelLoadActivity | null
}): ReactNode {
  const residency = modelLoadActivity?.residency
  const stabilizedDetail = useStabilizedRootDetail(status)
  if (residency?._tag === "Requested" || residency?._tag === "Loading") {
    const percentage = residency._tag === "Requested"
      ? 0
      : Math.min(
          100,
          Math.max(0, Math.round(Option.getOrElse(residency.progress, () => 0) * 100))
        )
    return (
      <>
        <Spinner className="size-[14px] shrink-0 text-blue-700 motion-reduce:animate-none dark:text-blue-500" />
        <span className="shrink-0 text-slate-900 dark:text-slate-200">
          Loading model
        </span>
        <span className="shrink-0 text-slate-600 dark:text-slate-400">
          {` · ${percentage}%`}
        </span>
      </>
    )
  }
  const active = status._tag === "Working"
  const detail =
    stabilizedDetail === null ? null : rootDetailSegments(stabilizedDetail)
  const terminalLabel =
    status._tag === "Worked"
      ? `Worked ${formatElapsedMs(status.lastProductiveMs)}`
      : status._tag === "Interrupted"
      ? `Worked ${formatElapsedMs(status.lastProductiveMs)}`
      : "Idle"
  return (
    <>
      <Circle
        size={9}
        fill="currentColor"
        className={`${active ? "animate-pulse-dot" : undefined} ${
          active ? "text-blue-700 dark:text-blue-500" : "text-slate-500"
        }  shrink-0`}
      />
      <span
        className={`${
          active
            ? "text-slate-900 dark:text-slate-200"
            : "text-slate-600 dark:text-slate-400"
        }  shrink-0`}
      >
        {status._tag === "Working" ? "Working" : terminalLabel}
      </span>
      {status._tag === "Working" && (
        <span className="text-slate-600 dark:text-slate-400 shrink-0">
          {` · ${formatElapsedMs(
            displayRootStatusElapsedMs(status, Date.now())
          )}`}
        </span>
      )}
      {detail?.keyword != null && (
        <>
          <span className="text-slate-600 dark:text-slate-400"> · </span>
          <span className="animate-pulse-dot text-slate-600 dark:text-slate-400">
            {detail.keyword}
          </span>
        </>
      )}
      {detail?.detail != null && (
        <span className="text-slate-600 dark:text-slate-400">{` · ${detail.detail}`}</span>
      )}
      {detail?.trailing != null && (
        <span className="text-slate-600 dark:text-slate-400">{` · ${detail.trailing}`}</span>
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
  const actor =
    assignee.kind === "actor" ? actors[assignee.actorKey] : undefined
  const actorProfile = findSlotProfileForRole(slotProfiles, actor?.role)
  const tokenCap = actorProfile?.contextWindow ?? null
  const detailLabel = actor ? actorProfile?.modelDisplayName ?? status : status
  const timerMs = actorTimer(actor)
  const isCompleted = row.status === "completed"
  const statusColorClass =
    status === "working" || status === "spawning"
      ? "text-blue-700 dark:text-blue-500"
      : status === "killing"
      ? "text-red-600 dark:text-red-500"
      : isCompleted
      ? "text-green-700 dark:text-green-500"
      : "text-slate-500"
  return (
    <Button variant="unstyled" size="unstyled"
      type="button"
      disabled={!isInteractive}
      onClick={() => {
        if (forkId) onWorkerClick?.(forkId)
      }}
      className={`${
        isInteractive
          ? "bg-transparent hover:bg-slate-100 dark:hover:bg-slate-800"
          : undefined
      } ${isInteractive ? "cursor-pointer" : "cursor-default"} ${
        isCompleted ? "opacity-[0.72]" : "opacity-[1]"
      }  grid [grid-template-columns:minmax(0,_1fr)_minmax(80px,_120px)_minmax(72px,_110px)_minmax(64px,_88px)_minmax(54px,_72px)] items-center [gap:8px] w-full [min-height:30px] border-0 rounded-[4px] [background:transparent] text-slate-900 dark:text-slate-200 font-sans text-[12px] text-left [padding:0_6px]`}
    >
      <span className="flex items-center [gap:7px] min-w-0">
        <Circle
          size={8}
          fill="currentColor"
          className={`${statusColorClass} ${
            status === "working" || status === "spawning"
              ? "animate-pulse-dot"
              : undefined
          }  shrink-0`}
        />
        <span className="overflow-hidden text-ellipsis whitespace-nowrap">
          {row.title}
        </span>
      </span>
      <span className="text-slate-600 dark:text-slate-400 overflow-hidden text-ellipsis whitespace-nowrap">
        {assigneeLabel(assignee, actors)}
      </span>
      <span className="text-slate-600 dark:text-slate-400 overflow-hidden text-ellipsis whitespace-nowrap">
        {detailLabel}
      </span>
      <span className="flex justify-end min-w-0">
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
      <span className="text-slate-500 text-right">
        {timerMs !== null && timerMs > 0 ? formatElapsedMs(timerMs) : ""}
      </span>
    </Button>
  )
}
export function WorkStatusBar({
  rootActor,
  actors,
  tasks,
  slotProfiles,
  modelLoadActivity = null,
  onWorkerClick,
}: WorkStatusBarProps): ReactNode {
  const [expanded, setExpanded] = useState(false)
  const rows = rowsFromTasks(tasks)
  const rootStatus = rootActor?.kind === "root" ? rootActor.status : null
  const chainActive = rootStatus?._tag === "Working"
  const anyActorClockRunning = Object.values(actors).some(
    (actor) =>
      actor.kind === "worker" && isDisplayWorkerStatusClockRunning(actor.status)
  )
  const tick = useSyncExternalStore(
    chainActive || anyActorClockRunning ? subscribeTick : subscribeNoop,
    getTickSnapshot,
    getTickSnapshot
  )
  void tick
  const hasTasks = rows.length > 0
  if (rootStatus === null || !isWorkStatusBarVisible(rootStatus, hasTasks))
    return null
  const incompleteCount =
    tasks?.summary.incompleteCount ??
    rows.filter((row) => row.status !== "completed").length
  const taskCountLabel =
    incompleteCount === 1 ? "1 task" : `${incompleteCount} tasks`
  const visibleRows = rows.slice(0, 10)
  return (
    <div className="[margin:0px] border border-slate-300 dark:border-slate-750 rounded-[6px] bg-white dark:bg-slate-850 overflow-hidden font-sans shrink-0">
      {hasTasks ? (
        <Button variant="unstyled" size="unstyled"
          type="button"
          onClick={() => setExpanded((value) => !value)}
          aria-expanded={expanded}
          aria-label={expanded ? "Collapse tasks" : "Expand tasks"}
          title={expanded ? "Collapse tasks" : "Expand tasks"}
          className="bg-transparent hover:bg-slate-100 dark:hover:bg-slate-800 [min-height:34px] w-full [padding:0_10px] flex items-center [gap:8px] text-slate-600 dark:text-slate-400 text-[13px] font-sans text-left [background:transparent] border-0 rounded-[0px] cursor-pointer"
        >
          <StatusSummary
            status={rootStatus}
            modelLoadActivity={modelLoadActivity}
          />
          <span className="[margin-left:auto] flex items-center [gap:4px] text-slate-600 dark:text-slate-400 text-[12px] shrink-0">
            <span>{taskCountLabel}</span>
            {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </span>
        </Button>
      ) : (
        <div className="[min-height:34px] [padding:0_10px] flex items-center [gap:8px] text-slate-600 dark:text-slate-400 text-[13px]">
          <StatusSummary
            status={rootStatus}
            modelLoadActivity={modelLoadActivity}
          />
        </div>
      )}

      {expanded && hasTasks && (
        <div className="border-t border-t-slate-200 dark:border-t-slate-800 [padding:6px] flex flex-col [gap:2px]">
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
            <div className="[padding:4px_6px] text-slate-500 text-[12px]">
              +{rows.length - visibleRows.length} more
            </div>
          )}
        </div>
      )}
    </div>
  )
}
