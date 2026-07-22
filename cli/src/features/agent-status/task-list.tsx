import { TextAttributes } from '@opentui/core'
import { blue, red, slate, subscribeAnimationTick, getAnimationTickSnapshot, findSlotProfile, type SlotProfiles } from '@magnitudedev/client-common'
import { Atom, useAtomMount } from '@effect-atom/atom-react'
import { Effect, Option } from 'effect'
import { useCallback, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { useTheme } from '../../hooks/use-theme'
import { useLocalWidth } from '../../hooks/use-local-width'
import { Button } from '../../components/button'
import { computeWorkerElapsedMs, formatWorkerTimer, isWorkerResumed } from '../../utils/task-list-worker-timer'
import { BOX_CHARS } from '../../utils/ui-constants'
import { formatTokensCompact } from '@magnitudedev/client-common'
import {
  computeInheritedVisualStatusMap,
  type VisualStatus,
} from '@magnitudedev/client-common'
import {
  buildRootSummaries,
  findOwningRootIndex,
} from '@magnitudedev/client-common'
import type { TaskAssignee, TaskDisplayRow } from '@magnitudedev/client-common'
import type { DisplayActor, DisplayTasks } from '@magnitudedev/sdk'
import { isRoleId, PRIMARY_SLOT_ID, ROLE_TO_SLOT, SECONDARY_SLOT_ID } from '@magnitudedev/sdk'

const COLLAPSED_ROWS = 6
const EXPANDED_ROWS = 25

export function getVisibleTasks(tasks: readonly TaskDisplayRow[], expanded: boolean): readonly TaskDisplayRow[] {
  return expanded ? tasks : tasks.slice(-COLLAPSED_ROWS)
}

const PULSE_BLUE_SHADES = [
  blue[50], blue[100], blue[200], blue[300], blue[400], blue[500], blue[600], blue[700], blue[800], blue[900],
  blue[800], blue[700], blue[600], blue[500], blue[400], blue[300], blue[200], blue[100], blue[50],
] as const

type Props = {
  tasks: readonly TaskDisplayRow[]
  actors?: Record<string, DisplayActor>
  taskSummary?: DisplayTasks['summary']
  pushForkOverlay: (forkId: string) => void
  slotProfiles: SlotProfiles | null
  scrollRefOverride?: { current: { scrollTo: (offset: number) => void } | null }
}

type TaskRowProps = {
  task: TaskDisplayRow
  effectiveStatus: VisualStatus
  pushForkOverlay: (forkId: string) => void
  hovered: boolean
  onHover: (taskId: string) => void
  onHoverEnd: () => void
  now: number
  actors: Record<string, DisplayActor>
  taskNameWidth: number
  columnGap: number
  agentIdWidth: number
  slotProfiles: SlotProfiles | null
}

type WorkerPresentation = {
  icon: string
  iconColor: string
  labelColor: string
  timerColor: string
  showTimer: boolean
  showResumed: boolean
  interactiveForkId: string | null
}

function truncate(s: string, maxWidth: number) {
  return s.length > maxWidth ? s.slice(0, maxWidth - 1) + '…' : s
}

function splitWorkerLabel(label: string): { badgeText: string; nameText: string } {
  const match = label.match(/^(.+?)\s·\s(.+)$/)
  if (match) return { badgeText: match[1], nameText: match[2] }
  return { badgeText: '', nameText: label }
}

const NAME_MIN_WIDTH = 4

function pickAssigneeLayout(args: {
  agentIdWidth: number
  iconWidth: number
  badgeText: string
  nameText: string
  modelText: string
  timerWidth: number
  tokensWidth: number
}) {
  const { agentIdWidth, iconWidth, badgeText, nameText, modelText, timerWidth, tokensWidth } = args
  const badgeWidth = badgeText ? badgeText.length + 1 : 0 // includes trailing space
  const modelWidth = modelText ? ` · ${modelText}`.length : 0

  const fits = (showBadge: boolean, showModel: boolean, showTokens: boolean, nameWidth: number) => {
    const used = iconWidth
      + (showBadge ? badgeWidth : 0)
      + nameWidth
      + (showModel ? modelWidth : 0)
      + timerWidth
      + (showTokens ? tokensWidth : 0)
    return used <= agentIdWidth
  }

  // Try, in order: full → drop tokens → drop badge → drop model → truncate name
  if (fits(!!badgeText, !!modelText, tokensWidth > 0, nameText.length)) {
    return { showBadge: !!badgeText, showModel: !!modelText, showTokens: tokensWidth > 0, nameMaxWidth: nameText.length }
  }
  if (fits(!!badgeText, !!modelText, false, nameText.length)) {
    return { showBadge: !!badgeText, showModel: !!modelText, showTokens: false, nameMaxWidth: nameText.length }
  }
  if (fits(false, !!modelText, false, nameText.length)) {
    return { showBadge: false, showModel: !!modelText, showTokens: false, nameMaxWidth: nameText.length }
  }
  if (fits(false, false, false, nameText.length)) {
    return { showBadge: false, showModel: false, showTokens: false, nameMaxWidth: nameText.length }
  }
  // Truncate name to whatever fits with just icon + timer
  const nameMaxWidth = Math.max(NAME_MIN_WIDTH, agentIdWidth - iconWidth - timerWidth)
  return { showBadge: false, showModel: false, showTokens: false, nameMaxWidth }
}

function getStatusGlyph(status: VisualStatus): '✓' | '○' {
  return status === 'completed' ? '✓' : '○'
}

function getStatusColor(status: VisualStatus, theme: ReturnType<typeof useTheme>): string {
  return status === 'completed' ? theme.success : theme.muted
}

function buildTaskTitleText(task: TaskDisplayRow) {
  return task.title
}

function getTaskIndent(depth: number): string {
  return depth > 0 ? '  '.repeat(depth) : ''
}

function formatRoleLabel(role: string): string {
  return role.length === 0 ? role : role.charAt(0).toUpperCase() + role.slice(1)
}

function getAssigneeLabel(
  assignee: TaskAssignee,
  actors: Record<string, DisplayActor>,
): string {
  if (assignee.kind === 'none') return ''
  if (assignee.kind === 'actor') {
    const actor = actors[assignee.actorKey]
    return actor?.role ? formatRoleLabel(actor.role) : (actor?.name ?? assignee.actorKey)
  }
  if (assignee.kind === 'user') return 'user'
  return assignee.label
}

function getWorkerPresentation(
  assignee: TaskAssignee,
  actors: Record<string, DisplayActor>,
  now: number,
  theme: ReturnType<typeof useTheme>,
  hovered: boolean,
): WorkerPresentation | null {
  switch (assignee.kind) {
    case 'none':
      return null
    case 'user':
      return {
        icon: '',
        iconColor: theme.warning ?? theme.foreground,
        labelColor: theme.warning ?? theme.foreground,
        timerColor: theme.warning ?? theme.foreground,
        showTimer: false,
        showResumed: false,
        interactiveForkId: null,
      }
    case 'worker': {
      const labelBaseColor = theme.foreground
      const labelColor = hovered ? theme.primary : labelBaseColor
      return {
        icon: assignee.icon,
        iconColor: PULSE_BLUE_SHADES[Math.floor(now / 200) % PULSE_BLUE_SHADES.length],
        labelColor,
        timerColor: theme.muted,
        showTimer: false,
        showResumed: false,
        interactiveForkId: assignee.interactiveForkId._tag === 'Some' ? assignee.interactiveForkId.value : null,
      }
    }
    case 'actor': {
      const actor = actors[assignee.actorKey]
      const isKilling = assignee.taskState === 'killing'
      const isWorking = actor?.work.phase === 'working'
      const labelBaseColor = isKilling ? red[500] : theme.foreground
      const labelColor = hovered && actor?.kind === 'worker' ? theme.primary : labelBaseColor

      return {
        icon: isKilling ? '✕' : '●',
        iconColor: isKilling
          ? red[500]
          : isWorking
            ? PULSE_BLUE_SHADES[Math.floor(now / 200) % PULSE_BLUE_SHADES.length]
            : slate[600],
        labelColor,
        timerColor: isKilling ? labelColor : theme.muted,
        showTimer: !isKilling,
        showResumed: !isKilling,
        interactiveForkId: actor?.kind === 'worker' ? assignee.actorKey : null,
      }
    }
  }
}

function TaskNameContent({
  task,
  effectiveStatus,
  taskNameWidth,
  theme,
}: {
  task: TaskDisplayRow
  effectiveStatus: VisualStatus
  taskNameWidth: number
  theme: ReturnType<typeof useTheme>
}) {
  const isCompleted = task.status === 'completed'
  const indent = getTaskIndent(task.depth)
  const glyphText = `${getStatusGlyph(effectiveStatus)} `
  const prefixWidth = indent.length + glyphText.length
  const titleText = buildTaskTitleText(task)
  const taskNameStr = truncate(titleText, Math.max(1, taskNameWidth - prefixWidth))

  return (
    <>
      {indent.length > 0 && <text style={{ fg: theme.muted }}>{indent}</text>}
      <text style={{ fg: getStatusColor(effectiveStatus, theme) }}>{glyphText}</text>
      {isCompleted
        ? <text attributes={TextAttributes.STRIKETHROUGH} style={{ fg: theme.muted }}>{taskNameStr}</text>
        : <text style={{ fg: theme.foreground }}>{taskNameStr}</text>}
    </>
  )
}

function TaskRow({
  task,
  effectiveStatus,
  pushForkOverlay,
  hovered,
  onHover,
  onHoverEnd,
  now,
  actors,
  taskNameWidth,
  columnGap,
  agentIdWidth,
  slotProfiles,
}: TaskRowProps) {
  const theme = useTheme()
  const workerPresentation = getWorkerPresentation(task.assignee, actors, now, theme, hovered)
  const workerLabel = getAssigneeLabel(task.assignee, actors)
  const actor = task.assignee.kind === 'actor' ? actors[task.assignee.actorKey] : null
  const workerTimerState = (() => {
    if (!actor || task.assignee.kind !== 'actor' || task.assignee.taskState === 'killing') return null
    const work = actor.work
    return {
      state: work.phase === 'working' ? 'working' as const : 'idle' as const,
      activeSince: work.phase === 'working' ? work.activeSince : null,
      accumulatedActiveMs: work.accumulatedMs,
      resumeCount: work.resumeCount,
    }
  })()
  const workerTimer = workerTimerState && workerPresentation?.showTimer
    ? formatWorkerTimer(computeWorkerElapsedMs(workerTimerState, now))
    : null
  const workerResumed = workerTimerState && workerPresentation?.showResumed
    ? isWorkerResumed(workerTimerState)
    : false
  const canOpenWorkerFork = Boolean(workerPresentation?.interactiveForkId)

  const workerTokens = actor && actor.context.tokenEstimate > 0 ? actor.context.tokenEstimate : null
  const workerRole = actor?.role ?? null
  const workerSlot = workerRole && isRoleId(workerRole) ? ROLE_TO_SLOT[workerRole] : null
  const workerSlotId = workerSlot === null
    ? null
    : workerSlot === 'primary' ? PRIMARY_SLOT_ID : SECONDARY_SLOT_ID
  const modelDisplayName = workerSlotId && slotProfiles
    ? Option.getOrNull(Option.map(findSlotProfile(slotProfiles, workerSlotId), ({ modelDisplayName }) => modelDisplayName))
    : null
  const tokensLabel = workerTokens != null ? formatTokensCompact(workerTokens) : null

  const { badgeText, nameText } = splitWorkerLabel(workerLabel)
  const iconText = workerPresentation?.icon ? `${workerPresentation.icon} ` : ''
  const timerText = workerTimer ? ` · ${workerResumed ? '↺ ' : ''}${workerTimer}` : ''
  const layout = pickAssigneeLayout({
    agentIdWidth,
    iconWidth: iconText.length,
    badgeText,
    nameText,
    modelText: modelDisplayName ?? '',
    timerWidth: timerText.length,
    tokensWidth: tokensLabel ? ` · ${tokensLabel}`.length : 0,
  })
  const displayedName = layout.nameMaxWidth < nameText.length
    ? truncate(nameText, layout.nameMaxWidth)
    : nameText

  return (
    <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
      <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0, flexDirection: 'row' }}>
        <TaskNameContent task={task} effectiveStatus={effectiveStatus} taskNameWidth={taskNameWidth} theme={theme} />
      </box>
      <box style={{ width: columnGap, minWidth: columnGap, maxWidth: columnGap, flexShrink: 0 }} />
      {workerPresentation && workerLabel ? (
        <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row' }}>
          {canOpenWorkerFork ? (
            <Button
              onClick={() => pushForkOverlay(workerPresentation.interactiveForkId!)}
              onMouseOver={() => onHover(task.taskId)}
              onMouseOut={() => onHoverEnd()}
            >
              <text style={{ fg: workerPresentation.labelColor }}>
                <span fg={workerPresentation.iconColor}>{iconText}</span>
                <span fg={workerPresentation.labelColor}>{workerLabel}</span>
              </text>
            </Button>
          ) : (
            <text style={{ fg: workerPresentation.labelColor }}>
              <span fg={workerPresentation.iconColor}>{iconText}</span>
              <span fg={workerPresentation.labelColor}>{workerLabel}</span>
            </text>
          )}
          {layout.showModel && modelDisplayName ? (
            <text style={{ fg: theme.muted }}>{` · ${modelDisplayName}`}</text>
          ) : null}
          {workerTimer ? (
            <text style={{ fg: workerPresentation.timerColor }}>
              <span fg={workerPresentation.timerColor}>{' · '}</span>
              {workerResumed ? <span fg={workerPresentation.timerColor}>↺ </span> : null}
              <span fg={workerPresentation.timerColor}>{workerTimer}</span>
            </text>
          ) : null}
          {layout.showTokens && tokensLabel ? (
            <text style={{ fg: theme.muted }}>{` · ${tokensLabel}`}</text>
          ) : null}
        </box>
      ) : null}
    </box>
  )
}

export function TaskList({
  tasks,
  actors = {},
  taskSummary,
  pushForkOverlay,
  slotProfiles,
  scrollRefOverride,
}: Props) {
  const theme = useTheme()
  const [expanded, setExpanded] = useState(false)
  const [expandHovered, setExpandHovered] = useState(false)
  const [hoveredTaskId, setHoveredTaskId] = useState<string | null>(null)
  // now is derived from animation tick below
  const taskScrollRef = useRef<any>(null)

  const box = useLocalWidth()
  const usableWidth = Math.max(1, (box.width ?? 60) - 4)
  const columnGap = 2
  const contentWidth = Math.max(14, usableWidth - columnGap)
  const agentIdWidth = Math.max(12, Math.floor(contentWidth * 0.55))
  const taskNameWidth = Math.max(1, contentWidth - agentIdWidth)

  const visibleAllTasks = tasks
  const realTasksOnly = useMemo(
    () => visibleAllTasks,
    [visibleAllTasks]
  )

  const effectiveVisualStates = useMemo(() => computeInheritedVisualStatusMap(realTasksOnly), [realTasksOnly])
  const rootSummaries = useMemo(() => buildRootSummaries(realTasksOnly), [realTasksOnly])

  const needsFastTick = useMemo(
    () => visibleAllTasks.some(task => (
      task.assignee.kind === 'worker'
        || (task.assignee.kind === 'actor'
          && actors[task.assignee.actorKey]?.work.phase === 'working')
    )),
    [actors, visibleAllTasks]
  )

  // Timer tick — use animation tick store for re-renders (no useEffect)
  const animTick = useSyncExternalStore(subscribeAnimationTick, getAnimationTickSnapshot, getAnimationTickSnapshot)
  const nowRef = useRef(Date.now())
  const tickDivisor = needsFastTick ? 3 : 13
  if (animTick % tickDivisor === 0) nowRef.current = Date.now()
  const now = nowRef.current

  const handleHoverEnd = useCallback(() => setHoveredTaskId(null), [])
  const snapExpandedToBottom = useCallback(() => {
    const scrollTarget = scrollRefOverride?.current ?? taskScrollRef.current
    scrollTarget?.scrollTo(Number.MAX_SAFE_INTEGER)
  }, [scrollRefOverride])
  const visibleTasks = getVisibleTasks(visibleAllTasks, expanded)
  const completedCount = taskSummary?.completedCount ?? realTasksOnly.filter(task => task.status === 'completed').length
  const activeCount = taskSummary?.incompleteCount ?? realTasksOnly.filter(task => task.status !== 'completed').length

  const stickyRootSummary = useMemo(() => {
    if (expanded) return null

    const collapsedTasks = visibleTasks
    if (collapsedTasks.length === 0) return null
    const firstRealCollapsedTask = collapsedTasks[0]
    if (!firstRealCollapsedTask) return null
    const firstIdx = realTasksOnly.findIndex(t => t.taskId === firstRealCollapsedTask.taskId)
    if (firstIdx < 0) return null
    const rootIdx = findOwningRootIndex(realTasksOnly, firstIdx)
    if (rootIdx === null) return null
    const rootTask = realTasksOnly[rootIdx]
    if (!rootTask || collapsedTasks.some(t => t.taskId === rootTask.taskId)) return null
    return rootSummaries.find((root) => root.task.taskId === rootTask.taskId) ?? null
  }, [expanded, rootSummaries, realTasksOnly, visibleTasks])

  // Initial snap on expand — useAtomMount lifecycle (post-commit, ref is populated)
  const snapAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (!expanded) return
          snapExpandedToBottom()
          yield* Effect.sleep('50 millis')
          snapExpandedToBottom()
        }),
      ),
    [expanded, tasks.length, snapExpandedToBottom],
  )
  useAtomMount(snapAtom)

  if (visibleAllTasks.length === 0) return null

  return (
    <box
      ref={box.ref}
      onSizeChange={box.onSizeChange}
      style={{ flexDirection: 'column', flexShrink: 0, borderStyle: 'single', border: ['left', 'right', 'top', 'bottom'], borderColor: slate[500], customBorderChars: BOX_CHARS, backgroundColor: 'transparent', paddingLeft: 1, paddingRight: 1 }}
    >
      {stickyRootSummary && stickyRootSummary.task.kind === 'task' ? (
        <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
          <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0, flexDirection: 'row' }}>
            {(() => {
              const stickyTask = stickyRootSummary.task
              const countsStr = ` (${stickyRootSummary.completed} completed, ${stickyRootSummary.active} active)`
              return <>
                <TaskNameContent task={stickyTask} effectiveStatus={effectiveVisualStates.get(stickyTask.taskId) ?? 'pending'} taskNameWidth={taskNameWidth - countsStr.length} theme={theme} />
                <text style={{ fg: theme.muted }}>{countsStr}</text>
              </>
            })()}
          </box>
          <box style={{ width: columnGap, minWidth: columnGap, maxWidth: columnGap, flexShrink: 0 }} />
          <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row', justifyContent: 'space-between' }}>
            {(() => {
              const stickyAssignee = stickyRootSummary.task.assignee
              const stickyActor = stickyAssignee.kind === 'actor' ? actors[stickyAssignee.actorKey] : null
              const stickyRole = stickyActor?.role ?? null
              const stickySlot = stickyRole && isRoleId(stickyRole) ? ROLE_TO_SLOT[stickyRole] : null
              const stickySlotId = stickySlot === null
                ? null
                : stickySlot === 'primary' ? PRIMARY_SLOT_ID : SECONDARY_SLOT_ID
              const stickyModel = stickySlotId && slotProfiles
                ? Option.getOrNull(Option.map(findSlotProfile(slotProfiles, stickySlotId), ({ modelDisplayName }) => modelDisplayName))
                : null
              const labelText = getAssigneeLabel(stickyAssignee, actors)
              const expandWidth = (expanded ? 'Collapse all ▼  ' : 'Expand all ▲  ').length
              const availableWidth = Math.max(0, agentIdWidth - expandWidth)
              const modelSuffix = stickyModel ? ` · ${stickyModel}` : ''
              const showModel = stickyModel != null && labelText.length + modelSuffix.length <= availableWidth
              const labelDisplay = truncate(labelText, Math.max(1, showModel ? availableWidth - modelSuffix.length : availableWidth))
              return (
                <text style={{ fg: theme.muted }}>
                  <span fg={theme.muted}>{labelDisplay}</span>
                  {showModel ? <span fg={theme.muted}>{modelSuffix}</span> : null}
                </text>
              )
            })()}
            <Button onClick={() => setExpanded(prev => !prev)} onMouseOver={() => setExpandHovered(true)} onMouseOut={() => setExpandHovered(false)}>
              <text style={{ fg: expandHovered ? theme.foreground : theme.muted }}>{expanded ? 'Collapse all ▼  ' : 'Expand all ▲  '}</text>
            </Button>
          </box>
        </box>
      ) : (
        <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
          <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0, flexDirection: 'row' }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Task</text>
            <text style={{ fg: theme.muted }}>{` (${completedCount} completed, ${activeCount} active)`}</text>
          </box>
          <box style={{ width: columnGap, minWidth: columnGap, maxWidth: columnGap, flexShrink: 0 }} />
          <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row', justifyContent: 'space-between' }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Assigned To</text>
            <Button onClick={() => setExpanded(prev => !prev)} onMouseOver={() => setExpandHovered(true)} onMouseOut={() => setExpandHovered(false)}>
              <text style={{ fg: expandHovered ? theme.foreground : theme.muted }}>{expanded ? 'Collapse all ▼  ' : 'Expand all ▲  '}</text>
            </Button>
          </box>
        </box>
      )}

      {expanded ? (
        <scrollbox
          ref={taskScrollRef}
          stickyScroll
          stickyStart="bottom"
          scrollX={false}
          scrollbarOptions={{ visible: false }}
          verticalScrollbarOptions={{ visible: true, trackOptions: { width: 1 } }}
          style={{
            flexShrink: 0,
            rootOptions: {
              height: EXPANDED_ROWS,
              flexShrink: 0,
              backgroundColor: 'transparent',
            },
            wrapperOptions: {
              border: false,
              backgroundColor: 'transparent',
            },
            contentOptions: {
              flexDirection: 'column',
            },
          }}
        >
          {visibleTasks.map(task => (
            <TaskRow
              key={task.rowId}
              task={task}
              effectiveStatus={effectiveVisualStates.get(task.taskId) ?? 'pending'}
              pushForkOverlay={pushForkOverlay}
              hovered={hoveredTaskId === task.taskId}
              onHover={setHoveredTaskId}
              onHoverEnd={handleHoverEnd}
              now={now}
              actors={actors}
              taskNameWidth={taskNameWidth}
              columnGap={columnGap}
              agentIdWidth={agentIdWidth}
              slotProfiles={slotProfiles}
            />
          ))}
        </scrollbox>
      ) : (
        <box style={{ flexDirection: 'column' }}>
          {visibleTasks.map(task => (
            <TaskRow
              key={task.rowId}
              task={task}
              effectiveStatus={effectiveVisualStates.get(task.taskId) ?? 'pending'}
              pushForkOverlay={pushForkOverlay}
              hovered={hoveredTaskId === task.taskId}
              onHover={setHoveredTaskId}
              onHoverEnd={handleHoverEnd}
              now={now}
              actors={actors}
              taskNameWidth={taskNameWidth}
              columnGap={columnGap}
              agentIdWidth={agentIdWidth}
              slotProfiles={slotProfiles}
            />
          ))}
        </box>
      )}
    </box>
  )
}
