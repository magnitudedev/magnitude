import { TextAttributes, type ScrollBoxRenderable } from '@opentui/core'
import { blue, slate } from '../../utils/palette'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTheme } from '../../hooks/use-theme'
import { useTerminalWidth } from '../../hooks/use-terminal-width'
import { Button } from '../button'
import { formatSubagentIdWithEmoji } from '../../utils/subagent-role-emoji'
import { BOX_CHARS } from '../../utils/ui-constants'
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
  pushForkOverlay: (forkId: string) => void
  hovered: boolean
  onHover: (taskId: string) => void
  onHoverEnd: () => void
  now: number
  taskNameWidth: number
  agentIdWidth: number
}

function TaskRow({ task, pushForkOverlay, hovered, onHover, onHoverEnd, now, taskNameWidth, agentIdWidth }: TaskRowProps) {
  const theme = useTheme()

  const truncate = (s: string, maxWidth: number) =>
    s.length > maxWidth ? s.slice(0, maxWidth - 1) + '…' : s

  const isCompleted = task.status === 'completed'
  const isWorking = task.status === 'working'
  const pulseColor =
    PULSE_BLUE_SHADES[Math.floor(now / 200) % PULSE_BLUE_SHADES.length]

  const prefix = `${'  '.repeat(task.depth)}${task.depth > 0 ? '└─ ' : ''}`
  const typeLabel = `[${task.type}] `
  const taskNameStr = truncate(`${prefix}${typeLabel}${task.title}`, taskNameWidth)

  const assigneeLabel = task.assignee.kind === 'lead'
    ? 'lead'
    : task.assignee.kind === 'user'
      ? 'user'
      : truncate(formatSubagentIdWithEmoji(task.assignee.agentId, task.assignee.workerType), agentIdWidth)

  return (
    <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
      {/* Status */}
      <box style={{ width: 2, minWidth: 2, maxWidth: 2, flexShrink: 0 }}>
        {isCompleted
          ? <text style={{ fg: theme.success }}>✓ </text>
          : isWorking
            ? <text style={{ fg: pulseColor }}>◉ </text>
            : <text style={{ fg: theme.muted }}>○ </text>
        }
      </box>

      {/* Task name */}
      <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0 }}>
        {isCompleted
          ? <text attributes={TextAttributes.STRIKETHROUGH} style={{ fg: theme.muted }}>{taskNameStr}</text>
          : <text style={{ fg: theme.foreground }}>{taskNameStr}</text>
        }
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
  const [now, setNow] = useState(() => Date.now())
  const scrollRef = useRef<ScrollBoxRenderable | null>(null)

  const terminalWidth = useTerminalWidth()
  const usableWidth = Math.max(1, terminalWidth - 4)
  const checkboxWidth = 2
  const remainingWidth = Math.max(1, usableWidth - checkboxWidth)
  const taskNameWidth = Math.floor(remainingWidth * 0.57)
  const agentIdWidth = remainingWidth - taskNameWidth

  const hasActiveTasks = tasks.some(t => t.status === 'working')

  useEffect(() => {
    if (tasks.length === 0) return
    const tickMs = hasActiveTasks ? 200 : 1000
    setNow(Date.now())
    const interval = setInterval(() => setNow(Date.now()), tickMs)
    return () => clearInterval(interval)
  }, [hasActiveTasks, tasks.length])

  // On new tasks — snap to bottom
  useEffect(() => {
    if (!expanded) return
    const snap = () => {
      scrollRef.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    }
    const t1 = setTimeout(snap, 0)
    const t2 = setTimeout(snap, 50)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [tasks.length, expanded])

  const handleHoverEnd = useCallback(() => setHoveredTaskId(null), [])

  if (tasks.length === 0) return null

  const visibleTasks = expanded ? tasks : tasks.slice(-COLLAPSED_ROWS)
  const completedCount = tasks.filter(task => task.status === 'completed').length
  const activeCount = tasks.filter(task => task.status === 'working').length

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
      <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
        {/* Spacer for checkbox column */}
        <box style={{ width: checkboxWidth, minWidth: checkboxWidth, maxWidth: checkboxWidth, flexShrink: 0 }} />
        {/* Task column header */}
        <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0, flexDirection: 'row' }}>
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Task</text>
          <text style={{ fg: theme.muted }}>
            {` (${completedCount} completed, ${activeCount} active)`}
          </text>
        </box>
        {/* Assigned to column header — label left, expand/collapse button right */}
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

      {/* Task rows */}
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
          {tasks.map(task => (
            <TaskRow
              key={task.taskId}
              task={task}
              pushForkOverlay={pushForkOverlay}
              hovered={hoveredTaskId === task.taskId}
              onHover={setHoveredTaskId}
              onHoverEnd={handleHoverEnd}
              now={now}
              taskNameWidth={taskNameWidth}
              agentIdWidth={agentIdWidth}
            />
          ))}
        </scrollbox>
      ) : (
        <box style={{ flexDirection: 'column' }}>
          {visibleTasks.map(task => (
            <TaskRow
              key={task.taskId}
              task={task}
              pushForkOverlay={pushForkOverlay}
              hovered={hoveredTaskId === task.taskId}
              onHover={setHoveredTaskId}
              onHoverEnd={handleHoverEnd}
              now={now}
              taskNameWidth={taskNameWidth}
              agentIdWidth={agentIdWidth}
            />
          ))}
        </box>
      )}
    </box>
  )
}
