import { TextAttributes } from '@opentui/core'
import { blue, orange, red, slate } from '../../utils/palette'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTheme } from '../../hooks/use-theme'
import { useTerminalWidth } from '../../hooks/use-terminal-width'
import { useBoxWidth } from '../../hooks/use-chat-width'
import { Button } from '../button'
import { computeWorkerElapsedMs, formatWorkerTimer, isWorkerResumed } from '../../utils/task-list-worker-timer'
import { BOX_CHARS } from '../../utils/ui-constants'
import {
  computeInheritedVisualStatusMap,
  type VisualStatus,
} from '../../utils/task-visual-status'
import {
  buildRootSummaries,
  findOwningRootIndex,
} from '../../utils/task-tree'
import type { TaskListItem } from './types'

const COLLAPSED_ROWS = 6
const EXPANDED_ROWS = 15

const PULSE_BLUE_SHADES = [
  blue[50], blue[100], blue[200], blue[300], blue[400], blue[500], blue[600], blue[700], blue[800], blue[900],
  blue[800], blue[700], blue[600], blue[500], blue[400], blue[300], blue[200], blue[100], blue[50],
] as const

type Props = {
  tasks: readonly TaskListItem[]
  pushForkOverlay: (forkId: string) => void
  fileViewerOpen?: boolean
}

type TaskRowProps = {
  task: TaskListItem
  effectiveStatus: VisualStatus
  pushForkOverlay: (forkId: string) => void
  hovered: boolean
  onHover: (taskId: string) => void
  onHoverEnd: () => void
  now: number
  taskNameWidth: number
  agentIdWidth: number
  showAssigneeColumn: boolean

}

function truncate(s: string, maxWidth: number) {
  return s.length > maxWidth ? s.slice(0, maxWidth - 1) + '…' : s
}

function getStatusGlyph(status: VisualStatus): '✓' | '○' {
  return status === 'completed' ? '✓' : '○'
}

function getStatusColor(status: VisualStatus, theme: ReturnType<typeof useTheme>): string {
  return status === 'completed' ? theme.success : theme.muted
}

function buildTaskTitleText(task: TaskListItem) {
  const SHOWN_TYPES = new Set(['feature', 'bug', 'refactor'])
  const typeLabel = SHOWN_TYPES.has(task.type) ? `[${task.type}] ` : ''
  return `${typeLabel}${task.title}`
}

function getTaskIndent(depth: number): string {
  return depth > 0 ? '  '.repeat(depth) : ''
}

function HeaderCountsText({ completed, active, theme }: { completed: number; active: number; theme: ReturnType<typeof useTheme> }) {
  return <text style={{ fg: theme.muted }}>{` (${completed} completed, ${active} active)`}</text>
}

function getWorkerStatusIcon(task: TaskListItem, now: number): { text: string; color: string } | null {
  if (task.assignee.kind !== 'worker') return null
  const state = task.workerExecution?.state
  if (state === 'working') return { text: '◉', color: PULSE_BLUE_SHADES[Math.floor(now / 200) % PULSE_BLUE_SHADES.length] }
  if (state === 'idle') return { text: '◌', color: orange[500] }
  if (state === 'killed') return { text: '✕', color: red[500] }
  return null
}

function isCompositeTask(task: TaskListItem): boolean {
  return task.type === 'bug' || task.type === 'feature' || task.type === 'refactor' || task.type === 'group'
}

function getAssigneeLabel(task: TaskListItem): string {
  if (isCompositeTask(task)) return '---'
  if (task.assignee.kind === 'lead') return 'lead'
  if (task.assignee.kind === 'none') return ''
  if (task.assignee.kind === 'user') return 'user'
  return task.assignee.workerType ? `[${task.assignee.workerType}] ${task.assignee.agentId}` : task.assignee.agentId
}

function TaskNameContent({
  task,
  effectiveStatus,
  taskNameWidth,
}: {
  task: TaskListItem
  effectiveStatus: VisualStatus
  taskNameWidth: number
}) {
  const theme = useTheme()
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
  taskNameWidth,
  agentIdWidth,
  showAssigneeColumn,
}: TaskRowProps) {
  const theme = useTheme()
  const assigneeLabel = getAssigneeLabel(task)
  const workerStatusIcon = getWorkerStatusIcon(task, now)
  const workerResumed = task.assignee.kind === 'worker' && task.workerExecution ? isWorkerResumed(task.workerExecution) : false
  const workerTimer = task.assignee.kind === 'worker' && task.workerExecution
    ? formatWorkerTimer(computeWorkerElapsedMs(task.workerExecution, now))
    : null
  return (
    <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
      <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0, flexDirection: 'row' }}>
        <TaskNameContent task={task} effectiveStatus={effectiveStatus} taskNameWidth={taskNameWidth} />
      </box>

      {showAssigneeColumn && (
        <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row' }}>
          {task.assignee.kind === 'worker' && task.workerForkId ? (
            <Button onClick={() => pushForkOverlay(task.workerForkId!)} onMouseOver={() => onHover(task.taskId)} onMouseOut={() => onHoverEnd()}>
              <text style={{ fg: hovered ? slate[300] : theme.muted }}>
                <span fg={workerStatusIcon?.color ?? (hovered ? slate[300] : theme.muted)}>{workerStatusIcon ? `${workerStatusIcon.text} ` : ''}</span>
                <span fg={hovered ? slate[300] : theme.muted}>{assigneeLabel}</span>
                {workerTimer ? (
                  <>
                    <span fg={theme.muted}> · </span>
                    {workerResumed ? <span fg={theme.muted}>↺ </span> : null}
                    <span fg={theme.muted}>{workerTimer}</span>
                  </>
                ) : null}
              </text>
            </Button>
          ) : (
            <text style={{ fg: task.assignee.kind === 'user' ? theme.warning ?? theme.foreground : theme.muted }}>
              {truncate(assigneeLabel, agentIdWidth)}
            </text>
          )}
        </box>
      )}
    </box>
  )
}

export function TaskList({ tasks, pushForkOverlay, fileViewerOpen = false }: Props) {
  const theme = useTheme()
  const [expanded, setExpanded] = useState(false)
  const [expandHovered, setExpandHovered] = useState(false)
  const [hoveredTaskId, setHoveredTaskId] = useState<string | null>(null)
  const [now, setNow] = useState(() => Date.now())

  const terminalWidth = useTerminalWidth()
  const box = useBoxWidth()
  const showAssigneeColumn = !fileViewerOpen
  const usableWidth = Math.max(1, (box.width ?? terminalWidth) - 4)
  const taskNameWidth = showAssigneeColumn ? Math.floor(usableWidth * 0.57) : usableWidth
  const agentIdWidth = showAssigneeColumn ? (usableWidth - taskNameWidth) : 0

  const visibleAllTasks = tasks

  const effectiveVisualStates = useMemo(() => computeInheritedVisualStatusMap(visibleAllTasks), [visibleAllTasks])
  const rootSummaries = useMemo(() => buildRootSummaries(visibleAllTasks), [visibleAllTasks])

  const hasWorkingTasks = useMemo(
    () => visibleAllTasks.some(task => task.workerExecution?.state === 'working'),
    [visibleAllTasks]
  )

  useEffect(() => {
    if (tasks.length === 0) return
    const tickMs = hasWorkingTasks ? 200 : 1000
    setNow(Date.now())
    const interval = setInterval(() => setNow(Date.now()), tickMs)
    return () => clearInterval(interval)
  }, [hasWorkingTasks, tasks.length])

  const handleHoverEnd = useCallback(() => setHoveredTaskId(null), [])
  const visibleTasks = expanded ? visibleAllTasks : visibleAllTasks.slice(-COLLAPSED_ROWS)
  const completedCount = visibleAllTasks.filter(task => task.status === 'completed').length
  const activeCount = visibleAllTasks.filter(task => task.status !== 'completed').length

  const stickyRootSummary = useMemo(() => {
    if (expanded) return null

    const collapsedTasks = visibleAllTasks.slice(-COLLAPSED_ROWS)
    if (collapsedTasks.length === 0) return null
    const firstIdx = visibleAllTasks.findIndex(t => t.taskId === collapsedTasks[0]?.taskId)
    if (firstIdx < 0) return null
    const rootIdx = findOwningRootIndex(visibleAllTasks, firstIdx)
    if (rootIdx === null) return null
    const rootTask = visibleAllTasks[rootIdx]
    if (!rootTask || collapsedTasks.some(t => t.taskId === rootTask.taskId)) return null
    return rootSummaries.find(root => root.task.taskId === rootTask.taskId) ?? null
  }, [expanded, rootSummaries, visibleAllTasks])

  if (visibleAllTasks.length === 0) return null

  return (
    <box
      ref={box.ref}
      onSizeChange={box.onSizeChange}
      style={{ flexDirection: 'column', flexShrink: 0, borderStyle: 'single', border: ['left', 'right', 'top', 'bottom'], borderColor: slate[500], customBorderChars: BOX_CHARS, backgroundColor: 'transparent', paddingLeft: 1, paddingRight: 1 }}
    >
      {stickyRootSummary ? (
        <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
          <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0, flexDirection: 'row' }}>
            <TaskNameContent task={stickyRootSummary.task} effectiveStatus={effectiveVisualStates.get(stickyRootSummary.task.taskId) ?? 'pending'} taskNameWidth={taskNameWidth} />
            <HeaderCountsText completed={stickyRootSummary.completed} active={stickyRootSummary.active} theme={theme} />
          </box>
          {showAssigneeColumn && (
            <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row', justifyContent: 'space-between' }}>
              <text style={{ fg: stickyRootSummary.task.assignee.kind === 'user' ? theme.warning ?? theme.foreground : theme.muted }}>
                {truncate(getAssigneeLabel(stickyRootSummary.task), agentIdWidth)}
              </text>
              <Button onClick={() => setExpanded(prev => !prev)} onMouseOver={() => setExpandHovered(true)} onMouseOut={() => setExpandHovered(false)}>
                <text style={{ fg: expandHovered ? theme.foreground : theme.muted }}>{expanded ? 'Collapse all ▼  ' : 'Expand all ▲  '}</text>
              </Button>
            </box>
          )}
        </box>
      ) : (
        <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
          <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0, flexDirection: 'row' }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Task</text>
            <HeaderCountsText completed={completedCount} active={activeCount} theme={theme} />
          </box>
          {showAssigneeColumn && (
            <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row', justifyContent: 'space-between' }}>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Assigned To</text>
              <Button onClick={() => setExpanded(prev => !prev)} onMouseOver={() => setExpandHovered(true)} onMouseOut={() => setExpandHovered(false)}>
                <text style={{ fg: expandHovered ? theme.foreground : theme.muted }}>{expanded ? 'Collapse all ▼  ' : 'Expand all ▲  '}</text>
              </Button>
            </box>
          )}
        </box>
      )}

      <box style={{ flexDirection: 'column' }}>
        {visibleTasks.map(task => (
          <TaskRow
            key={task.taskId}
            task={task}
            effectiveStatus={effectiveVisualStates.get(task.taskId) ?? 'pending'}
            pushForkOverlay={pushForkOverlay}
            hovered={hoveredTaskId === task.taskId}
            onHover={setHoveredTaskId}
            onHoverEnd={handleHoverEnd}
            now={now}
            taskNameWidth={taskNameWidth}
            agentIdWidth={agentIdWidth}
            showAssigneeColumn={showAssigneeColumn}
          />
        ))}
      </box>
    </box>
  )
}
