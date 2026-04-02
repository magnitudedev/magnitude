import { TextAttributes, type ScrollBoxRenderable } from '@opentui/core'
import { blue, slate } from '../../utils/palette'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTheme } from '../../hooks/use-theme'
import { useTerminalWidth } from '../../hooks/use-terminal-width'
import { Button } from '../button'
import { formatSubagentIdWithEmoji } from '../../utils/subagent-role-emoji'
import { BOX_CHARS } from '../../utils/ui-constants'
import {
  computeInheritedVisualStatusMap,
  type VisualStatus,
} from '../../utils/task-visual-status'
import {
  type RootSummary,
  buildRootSummaries,
  findOwningRootIndex,
} from '../../utils/task-tree'
import type { TaskListItem } from './types'

const COLLAPSED_ROWS = 6
const EXPANDED_ROWS = 15

const PULSE_BLUE_SHADES = [
  blue[50],
  blue[100],
  blue[200],
  blue[300],
  blue[400],
  blue[500],
  blue[600],
  blue[700],
  blue[800],
  blue[900],
  blue[800],
  blue[700],
  blue[600],
  blue[500],
  blue[400],
  blue[300],
  blue[200],
  blue[100],
  blue[50],
] as const

type Props = {
  tasks: readonly TaskListItem[]
  pushForkOverlay: (forkId: string) => void
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
  onToggleArchived?: (taskId: string) => void
  archivedExpanded?: boolean
  rootRowRef?: (node: any) => void
}



function truncate(s: string, maxWidth: number) {
  return s.length > maxWidth ? s.slice(0, maxWidth - 1) + '…' : s
}

function getStatusGlyph(status: VisualStatus): '✓' | '◉' | '○' {
  if (status === 'completed') return '✓'
  if (status === 'working' || status === 'assigned') return '◉'
  return '○'
}

function getStatusColor(
  status: VisualStatus,
  now: number,
  theme: ReturnType<typeof useTheme>
): string {
  if (status === 'completed') return theme.success
  if (status === 'working') {
    return PULSE_BLUE_SHADES[Math.floor(now / 200) % PULSE_BLUE_SHADES.length]
  }
  if (status === 'assigned') return blue[500]
  return theme.muted
}

function buildTaskTitleText(task: TaskListItem, opts: { archivedExpanded?: boolean }) {
  const isSummaryRow = task.taskId.startsWith('__archived__')
  const SHOWN_TYPES = new Set(['feature', 'bug', 'refactor'])
  const typeLabel = isSummaryRow ? (opts.archivedExpanded ? '▾ ' : '▸ ') : (SHOWN_TYPES.has(task.type) ? `[${task.type}] ` : '')
  return `${typeLabel}${task.title}`
}

function getTaskIndent(depth: number): string {
  return depth > 0 ? '  '.repeat(depth) : ''
}

function HeaderCountsText({ completed, active, theme }: { completed: number; active: number; theme: ReturnType<typeof useTheme> }) {
  return (
    <text style={{ fg: theme.muted }}>
      {` (${completed} completed, ${active} active)`}
    </text>
  )
}

function getAssigneeLabel(task: TaskListItem, agentIdWidth: number): string {
  return task.assignee.kind === 'lead'
    ? 'lead'
    : task.assignee.kind === 'none'
      ? ''
      : task.assignee.kind === 'user'
        ? 'user'
        : truncate(
            formatSubagentIdWithEmoji(task.assignee.agentId, task.assignee.workerType),
            agentIdWidth
          )
}

type TaskNameContentProps = {
  task: TaskListItem
  effectiveStatus: VisualStatus
  now: number
  taskNameWidth: number
  archivedExpanded?: boolean
  titleSuffix?: string
}

function TaskNameContent({
  task,
  effectiveStatus,
  now,
  taskNameWidth,
  archivedExpanded,
  titleSuffix,
}: TaskNameContentProps) {
  const theme = useTheme()

  const isCompleted = task.status === 'completed' || task.status === 'archived'
  const isSummaryRow = task.taskId.startsWith('__archived__')
  const isArchived = task.status === 'archived' || isSummaryRow
  const indent = getTaskIndent(task.depth)
  const glyphText = isSummaryRow ? '' : `${getStatusGlyph(effectiveStatus)} `
  const prefixWidth = indent.length + glyphText.length
  const titleText = `${buildTaskTitleText(task, { archivedExpanded })}${titleSuffix ?? ''}`
  const taskNameStr = truncate(
    titleText,
    Math.max(1, taskNameWidth - prefixWidth)
  )

  return (
    <>
      {indent.length > 0 && <text style={{ fg: theme.muted }}>{indent}</text>}
      {!isSummaryRow && (
        <text style={{ fg: getStatusColor(effectiveStatus, now, theme) }}>
          {glyphText}
        </text>
      )}
      {isArchived
        ? <text style={{ fg: theme.muted }}>{taskNameStr}</text>
        : isCompleted
          ? <text attributes={TextAttributes.STRIKETHROUGH} style={{ fg: theme.muted }}>{taskNameStr}</text>
          : <text style={{ fg: theme.foreground }}>{taskNameStr}</text>
      }
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
  onToggleArchived,
  archivedExpanded,
  rootRowRef,
}: TaskRowProps) {
  const theme = useTheme()

  const isSummaryRow = task.taskId.startsWith('__archived__')
  const assigneeLabel = getAssigneeLabel(task, agentIdWidth)
  const summaryIndent = getTaskIndent(task.depth)
  const summaryTaskName = truncate(
    `${summaryIndent}${buildTaskTitleText(task, { archivedExpanded })}`,
    Math.max(1, taskNameWidth)
  )

  return (
    <box ref={rootRowRef} style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
      {/* Task name */}
      <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0, flexDirection: 'row' }}>
        {isSummaryRow && onToggleArchived
          ? (
            <Button onClick={() => onToggleArchived(task.taskId)}>
              <text style={{ fg: hovered ? slate[300] : theme.muted }}>{summaryTaskName}</text>
            </Button>
          )
          : (
            <>
              <TaskNameContent
                task={task}
                effectiveStatus={effectiveStatus}
                now={now}
                taskNameWidth={taskNameWidth}
                archivedExpanded={archivedExpanded}
              />
            </>
          )}
      </box>

      {/* Assigned to */}
      <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row' }}>
        {task.assignee.kind === 'worker' && task.workerForkId
          ? (
            <Button
              onClick={() => pushForkOverlay(task.workerForkId!)}
              onMouseOver={() => onHover(task.taskId)}
              onMouseOut={() => onHoverEnd()}
            >
              <text style={{ fg: hovered ? slate[300] : theme.muted }}>
                {assigneeLabel}
              </text>
            </Button>
          )
          : (
            <text style={{ fg: task.assignee.kind === 'user' ? theme.warning ?? theme.foreground : theme.muted }}>
              {assigneeLabel}
            </text>
          )}
      </box>
    </box>
  )
}

export function TaskList({ tasks, pushForkOverlay }: Props) {
  const theme = useTheme()
  const [expanded, setExpanded] = useState(false)
  const [expandHovered, setExpandHovered] = useState(false)
  const [hoveredTaskId, setHoveredTaskId] = useState<string | null>(null)
  const [expandedArchived, setExpandedArchived] = useState<Set<string>>(new Set())
  const [now, setNow] = useState(() => Date.now())
  const [stickyRootTaskId, setStickyRootTaskId] = useState<string | null>(null)
  const scrollRef = useRef<ScrollBoxRenderable | null>(null)
  const rootRowRefs = useRef<Map<string, any>>(new Map())

  const terminalWidth = useTerminalWidth()
  const usableWidth = Math.max(1, terminalWidth - 4)
  const remainingWidth = Math.max(1, usableWidth)
  const taskNameWidth = Math.floor(remainingWidth * 0.57)
  const agentIdWidth = remainingWidth - taskNameWidth

  const toggleArchived = useCallback((taskId: string) => {
    setExpandedArchived(prev => {
      const next = new Set(prev)
      if (next.has(taskId)) next.delete(taskId)
      else next.add(taskId)
      return next
    })
  }, [])

  const visibleAllTasks = useMemo(() => {
    const result: TaskListItem[] = []
    for (const task of tasks) {
      if (task.taskId.startsWith('__archived__')) {
        result.push(task)
      } else if (task.status === 'archived') {
        const summaryId = `__archived__${task.parentId ?? '__root'}`
        if (expandedArchived.has(summaryId)) result.push(task)
      } else {
        result.push(task)
      }
    }
    return result
  }, [tasks, expandedArchived])

  const effectiveVisualStates = useMemo(
    () => computeInheritedVisualStatusMap(visibleAllTasks),
    [visibleAllTasks]
  )

  const rootSummaries = useMemo(
    () => buildRootSummaries(visibleAllTasks),
    [visibleAllTasks]
  )

  const hasWorkingTasks = useMemo(
    () => Array.from(effectiveVisualStates.values()).some(s => s === 'working'),
    [effectiveVisualStates]
  )

  useEffect(() => {
    if (tasks.length === 0) return
    const tickMs = hasWorkingTasks ? 200 : 1000
    setNow(Date.now())
    const interval = setInterval(() => setNow(Date.now()), tickMs)
    return () => clearInterval(interval)
  }, [hasWorkingTasks, tasks.length])

  useEffect(() => {
    if (!expanded) return
    const snap = () => {
      scrollRef.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    }
    const t1 = setTimeout(snap, 0)
    const t2 = setTimeout(snap, 50)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [tasks.length, expanded])

  useEffect(() => {
    if (!expanded || rootSummaries.length === 0) {
      setStickyRootTaskId(null)
      return
    }

    const computeYOffset = (el: any, container: any) => {
      let offset = 0
      let current = el
      while (current && current !== container) {
        const yoga = current.yogaNode ?? current.getLayoutNode?.()
        if (yoga) offset += yoga.getComputedTop()
        current = current.parent
      }
      return offset
    }

    const checkScroll = () => {
      const scrollbox: any = scrollRef.current
      if (!scrollbox?.content) {
        setStickyRootTaskId(null)
        return
      }

      const scrollTop = scrollbox.scrollTop ?? 0
      let active: string | null = null

      for (const root of rootSummaries) {
        const el = rootRowRefs.current.get(root.task.taskId)
        if (!el) continue
        const offsetY = computeYOffset(el, scrollbox.content)
        if (scrollTop > offsetY) {
          active = root.task.taskId
        }
      }

      setStickyRootTaskId(active)
    }

    checkScroll()
    const interval = setInterval(checkScroll, 50)
    return () => clearInterval(interval)
  }, [expanded, rootSummaries])

  const handleHoverEnd = useCallback(() => setHoveredTaskId(null), [])

  const visibleTasks = expanded ? visibleAllTasks : visibleAllTasks.slice(-COLLAPSED_ROWS)
  const completedCount = visibleAllTasks.filter(task => task.status === 'completed').length
  const activeCount = visibleAllTasks.filter(task => task.status === 'working').length

  const stickyRootSummary = useMemo(() => {
    if (expanded) {
      if (!stickyRootTaskId) return null
      return rootSummaries.find(root => root.task.taskId === stickyRootTaskId) ?? null
    }

    const collapsedTasks = visibleAllTasks.slice(-COLLAPSED_ROWS)
    if (collapsedTasks.length === 0) return null

    const firstVisibleTaskId = collapsedTasks[0]?.taskId
    const firstIdx = visibleAllTasks.findIndex(t => t.taskId === firstVisibleTaskId)
    if (firstIdx < 0) return null

    const rootIdx = findOwningRootIndex(visibleAllTasks, firstIdx)
    if (rootIdx === null) return null

    const rootTask = visibleAllTasks[rootIdx]
    if (!rootTask) return null
    if (collapsedTasks.some(t => t.taskId === rootTask.taskId)) return null

    return rootSummaries.find(root => root.task.taskId === rootTask.taskId) ?? null
  }, [expanded, stickyRootTaskId, rootSummaries, visibleAllTasks])

  if (visibleAllTasks.length === 0) return null

  return (
    <box
      style={{
        flexDirection: 'column',
        flexShrink: 0,
        borderStyle: 'single',
        border: ['left', 'right', 'top', 'bottom'],
        borderColor: slate[500],
        customBorderChars: BOX_CHARS,
        backgroundColor: 'transparent',
        paddingLeft: 1,
        paddingRight: 1,
      }}
    >
      {/* Header row */}
      {stickyRootSummary
        ? (
          <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
            <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0, flexDirection: 'row' }}>
              <TaskNameContent
                task={stickyRootSummary.task}
                effectiveStatus={effectiveVisualStates.get(stickyRootSummary.task.taskId) ?? 'pending'}
                now={now}
                taskNameWidth={taskNameWidth}
              />
              <HeaderCountsText completed={stickyRootSummary.completed} active={stickyRootSummary.active} theme={theme} />
            </box>

            <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row', justifyContent: 'space-between' }}>
              <text style={{ fg: stickyRootSummary.task.assignee.kind === 'user' ? theme.warning ?? theme.foreground : theme.muted }}>
                {getAssigneeLabel(stickyRootSummary.task, agentIdWidth)}
              </text>
              <Button
                onClick={() => setExpanded(prev => !prev)}
                onMouseOver={() => setExpandHovered(true)}
                onMouseOut={() => setExpandHovered(false)}
              >
                <text style={{ fg: expandHovered ? theme.foreground : theme.muted }}>
                  {expanded ? 'Collapse all ▼  ' : 'Expand all ▲  '}
                </text>
              </Button>
            </box>
          </box>
        )
        : (
          <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
            <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0, flexDirection: 'row' }}>
              <>
                <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Task</text>
                <HeaderCountsText completed={completedCount} active={activeCount} theme={theme} />
              </>
            </box>

            <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row', justifyContent: 'space-between' }}>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Assigned To</text>
              <Button
                onClick={() => setExpanded(prev => !prev)}
                onMouseOver={() => setExpandHovered(true)}
                onMouseOut={() => setExpandHovered(false)}
              >
                <text style={{ fg: expandHovered ? theme.foreground : theme.muted }}>
                  {expanded ? 'Collapse all ▼  ' : 'Expand all ▲  '}
                </text>
              </Button>
            </box>
          </box>
        )}

      {expanded ? (
        <scrollbox
          ref={scrollRef}
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
          {visibleTasks.map(task => {
            const isRootRow = task.depth === 0 && !task.taskId.startsWith('__archived__')
            return (
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
                onToggleArchived={task.status === 'archived' ? toggleArchived : undefined}
                archivedExpanded={task.status === 'archived' ? expandedArchived.has(task.taskId) : undefined}
                rootRowRef={isRootRow
                  ? (node: any) => {
                      if (node) rootRowRefs.current.set(task.taskId, node)
                      else rootRowRefs.current.delete(task.taskId)
                    }
                  : undefined}
              />
            )
          })}
        </scrollbox>
      ) : (
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
              onToggleArchived={task.status === 'archived' ? toggleArchived : undefined}
              archivedExpanded={task.status === 'archived' ? expandedArchived.has(task.taskId) : undefined}
            />
          ))}
        </box>
      )}
    </box>
  )
}
