import { TextAttributes, type ScrollBoxRenderable } from '@opentui/core'
import { blue, slate } from '../../utils/palette'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTheme } from '../../hooks/use-theme'
import { useTerminalWidth } from '../../hooks/use-terminal-width'
import { Button } from '../button'
import { formatElapsedMs } from '../../utils/format-elapsed'
import { formatSubagentIdWithEmoji } from '../../utils/subagent-role-emoji'
import type { TaskItem } from './types'

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
  tasks: readonly TaskItem[]
  pushForkOverlay: (forkId: string) => void
  modeColor: string
  inputBg: string
}

type TaskRowProps = {
  task: TaskItem
  pushForkOverlay: (forkId: string) => void
  hovered: boolean
  onHover: (forkId: string) => void
  onHoverEnd: () => void
  now: number
  taskNameWidth: number
  agentIdWidth: number
}

function TaskRow({ task, pushForkOverlay, hovered, onHover, onHoverEnd, now, taskNameWidth, agentIdWidth }: TaskRowProps) {
  const theme = useTheme()

  const truncate = (s: string, maxWidth: number) =>
    s.length > maxWidth ? s.slice(0, maxWidth - 1) + '…' : s

  const isIdle = task.phase === 'idle'
  const pulseColor =
    PULSE_BLUE_SHADES[Math.floor(now / 200) % PULSE_BLUE_SHADES.length]

  const elapsedMs = isIdle
    ? task.accumulatedActiveMs + (task.completedAt != null ? task.completedAt - task.activeSince : 0)
    : task.accumulatedActiveMs + Math.max(0, now - task.activeSince)
  const timerStr = formatElapsedMs(elapsedMs)

  const taskNameStr = truncate(task.name, taskNameWidth)
  const timerSuffix = ` · ${timerStr}`
  const agentIdMaxChars = Math.max(1, agentIdWidth - timerSuffix.length)
  const subagentLabel =
    formatSubagentIdWithEmoji(task.agentId, task.role) + (task.resumeCount > 0 ? ' (resumed)' : '')
  const agentIdStr = truncate(subagentLabel, agentIdMaxChars)

  return (
    <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
      {/* Checkbox */}
      <box style={{ width: 2, minWidth: 2, maxWidth: 2, flexShrink: 0 }}>
        {isIdle
          ? <text style={{ fg: theme.success }}>✓ </text>
          : <text style={{ fg: pulseColor }}>◉ </text>
        }
      </box>

      {/* Task name */}
      <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0 }}>
        {isIdle
          ? <text attributes={TextAttributes.STRIKETHROUGH} style={{ fg: theme.muted }}>{taskNameStr}</text>
          : <text style={{ fg: theme.foreground }}>{taskNameStr}</text>
        }
      </box>

      {/* Assigned to: agent ID (clickable) + inline timer */}
      <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row' }}>
        <Button
          onClick={() => pushForkOverlay(task.forkId)}
          onMouseOver={() => onHover(task.forkId)}
          onMouseOut={() => onHoverEnd()}
        >
          <text style={{ fg: hovered ? slate[300] : theme.muted }}>
            {agentIdStr}
          </text>
        </Button>
        <text style={{ fg: theme.muted }}>{timerSuffix}</text>
      </box>
    </box>
  )
}

export function TaskList({ tasks, pushForkOverlay, modeColor, inputBg }: Props) {
  const theme = useTheme()
  const [expanded, setExpanded] = useState(false)
  const [expandHovered, setExpandHovered] = useState(false)
  const [hoveredForkId, setHoveredForkId] = useState<string | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const scrollRef = useRef<ScrollBoxRenderable | null>(null)

  const terminalWidth = useTerminalWidth()
  const usableWidth = Math.max(1, terminalWidth - 4)
  const checkboxWidth = 2
  const remainingWidth = Math.max(1, usableWidth - checkboxWidth)
  const taskNameWidth = Math.floor(remainingWidth * 0.57)
  const agentIdWidth = remainingWidth - taskNameWidth

  const hasActiveTasks = tasks.some(t => t.phase === 'active')

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

  const handleHoverEnd = useCallback(() => setHoveredForkId(null), [])

  if (tasks.length === 0) return null

  const visibleTasks = expanded ? tasks : tasks.slice(-COLLAPSED_ROWS)

  return (
    <box style={{ flexDirection: 'column', flexShrink: 0 }}>
      {/* Header row */}
      <box style={{ flexDirection: 'row', height: 1, minHeight: 1, maxHeight: 1 }}>
        {/* Spacer for checkbox column */}
        <box style={{ width: checkboxWidth, minWidth: checkboxWidth, maxWidth: checkboxWidth, flexShrink: 0 }} />
        {/* Task column header */}
        <box style={{ width: taskNameWidth, minWidth: taskNameWidth, maxWidth: taskNameWidth, flexShrink: 0 }}>
          <text style={{ fg: theme.muted }} attributes={TextAttributes.BOLD}>Task</text>
        </box>
        {/* Assigned to column header — label left, expand/collapse button right */}
        <box style={{ width: agentIdWidth, minWidth: agentIdWidth, maxWidth: agentIdWidth, flexShrink: 0, flexDirection: 'row', justifyContent: 'space-between' }}>
          <text style={{ fg: theme.muted }} attributes={TextAttributes.BOLD}>Assigned To</text>
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
              key={task.forkId}
              task={task}
              pushForkOverlay={pushForkOverlay}
              hovered={hoveredForkId === task.forkId}
              onHover={setHoveredForkId}
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
              key={task.forkId}
              task={task}
              pushForkOverlay={pushForkOverlay}
              hovered={hoveredForkId === task.forkId}
              onHover={setHoveredForkId}
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
