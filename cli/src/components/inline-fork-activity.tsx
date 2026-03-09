import { memo, useState, useEffect, useRef } from 'react'
import { TextAttributes } from '@opentui/core'
import type { ForkActivityMessage, ForkActivityToolCounts } from '@magnitudedev/agent'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'
import { BOX_CHARS } from '../utils/ui-constants'

interface InlineForkActivityProps {
  message: ForkActivityMessage
  onExpand: (forkId: string) => void
  onArtifactClick?: (name: string, section?: string) => void
}

function formatElapsedTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}

function getTotalToolCount(counts: ForkActivityToolCounts): number {
  return counts.reads + counts.writes + counts.edits + counts.commands
    + counts.webSearches + counts.artifactWrites + counts.artifactUpdates
    + counts.searches + counts.clicks + counts.navigations + counts.inputs
    + counts.evaluations + counts.other
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

// Pulse animation: ramp up through blue shades then back down
const PULSE_COLORS = [
  '#0c4a6e', '#075985', '#0369a1', '#0284c7', '#0ea5e9', '#38bdf8',
  '#7dd3fc', '#38bdf8', '#0ea5e9', '#0284c7', '#0369a1', '#075985',
]
const PULSE_INTERVAL_MS = 200

// Visible tool categories with their display format
const TOOL_DISPLAY: { key: keyof ForkActivityToolCounts; verb: string; noun: string; nounPlural: string }[] = [
  { key: 'reads', verb: 'Read', noun: 'file', nounPlural: 'files' },
  { key: 'writes', verb: 'Wrote', noun: 'file', nounPlural: 'files' },
  { key: 'edits', verb: 'Edited', noun: 'file', nounPlural: 'files' },
  { key: 'commands', verb: 'Ran', noun: 'command', nounPlural: 'commands' },
  { key: 'webSearches', verb: '', noun: 'web search', nounPlural: 'web searches' },
  { key: 'navigations', verb: 'Visited', noun: 'page', nounPlural: 'pages' },
  { key: 'clicks', verb: 'Clicked', noun: 'element', nounPlural: 'elements' },
  { key: 'inputs', verb: 'Typed', noun: 'input', nounPlural: 'inputs' },
  { key: 'evaluations', verb: 'Ran', noun: 'script', nounPlural: 'scripts' },
]

const ArtifactChip = memo(function ArtifactChip({
  name,
  onClick,
}: {
  name: string
  onClick?: () => void
}) {
  const theme = useTheme()
  const [isHovered, setIsHovered] = useState(false)

  return (
    <Button
      onClick={onClick}
      onMouseOver={() => setIsHovered(true)}
      onMouseOut={() => setIsHovered(false)}
    >
      <text style={{ fg: isHovered ? theme.link : theme.primary, wrapMode: 'none' }}>{'[≡ '}{name}{']'}</text>
    </Button>
  )
})

export const InlineForkActivity = memo(function InlineForkActivity({
  message,
  onExpand,
  onArtifactClick,
}: InlineForkActivityProps) {
  const theme = useTheme()
  const [isDetailsHovered, setIsDetailsHovered] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [pulseIndex, setPulseIndex] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pulseRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const isRunning = message.status === 'running'

  // Live timer for running state
  useEffect(() => {
    if (isRunning) {
      const update = () => {
        setElapsed(Math.floor((Date.now() - message.startedAt) / 1000))
      }
      update()
      timerRef.current = setInterval(update, 1000)
      return () => {
        if (timerRef.current) clearInterval(timerRef.current)
      }
    } else if (message.completedAt) {
      setElapsed(Math.floor((message.completedAt - message.startedAt) / 1000))
    }
  }, [isRunning, message.startedAt, message.completedAt])

  // Pulsing line animation
  useEffect(() => {
    if (isRunning) {
      pulseRef.current = setInterval(() => {
        setPulseIndex(prev => (prev + 1) % PULSE_COLORS.length)
      }, PULSE_INTERVAL_MS)
      return () => {
        if (pulseRef.current) clearInterval(pulseRef.current)
      }
    }
  }, [isRunning])

  // Determine line color
  const lineColor = isRunning
    ? PULSE_COLORS[pulseIndex]
    : message.status === 'completed' ? theme.success : theme.error

  const timeStr = isRunning ? formatElapsedTime(elapsed) : formatDuration(elapsed)
  const totalTools = getTotalToolCount(message.toolCounts)

  // Truncate name
  const displayName = message.name.length > 50 ? message.name.slice(0, 47) + '...' : message.name
  const roleLabel = capitalize(message.role)
  const resumed = (message.resumeCount ?? 0) > 0

  // Build visible tool summary parts
  const toolParts: { verb: string; count: number; noun: string }[] = []
  for (const { key, verb, noun, nounPlural } of TOOL_DISPLAY) {
    const count = message.toolCounts[key]
    if (count > 0) {
      toolParts.push({ verb, count, noun: count === 1 ? noun : nounPlural })
    }
  }

  return (
    <box
      style={{
        alignSelf: 'flex-start',
        marginBottom: 1,
        borderStyle: 'single',
        border: ['left'],
        borderColor: lineColor,
        customBorderChars: { ...BOX_CHARS, vertical: '┃' },
        paddingLeft: 1,
        flexDirection: 'column',
      }}
    >
      {/* Line 1: role agent: name · timer · tools */}
      <box style={{ flexDirection: 'row' }}>
        <text style={{ wrapMode: 'none' }}>
          <span fg={theme.foreground} attributes={TextAttributes.BOLD}>{roleLabel}:</span>
          {' '}
          <span fg={theme.foreground}>{displayName}</span>
          {resumed ? <span fg={theme.muted}> (resumed)</span> : null}
          <span fg={theme.muted}>{' · '}</span>
          <span fg={theme.muted}>{timeStr}</span>
          {totalTools > 0 ? (
            <>
              <span fg={theme.muted}>{' · '}</span>
              <span fg={theme.info}>{totalTools}</span>
              <span fg={theme.muted}>{totalTools === 1 ? ' tool' : ' tools'}</span>
            </>
          ) : null}
        </text>
      </box>

      {/* Line 2: tool summary */}
      <box style={{ flexDirection: 'row' }}>
        <text style={{ wrapMode: 'none' }}>
          {toolParts.length === 0 ? (
            <span fg={theme.muted}>Starting...</span>
          ) : (
            toolParts.map((part, i) => (
              <span key={i}>
                {i > 0 ? <span fg={theme.muted}>, </span> : null}
                {part.verb ? (
                  <>
                    <span fg={theme.muted}>{part.verb} </span>
                    <span fg={theme.info}>{part.count}</span>
                    <span fg={theme.muted}> {part.noun}</span>
                  </>
                ) : (
                  <>
                    <span fg={theme.info}>{part.count}</span>
                    <span fg={theme.muted}> {part.noun}</span>
                  </>
                )}
              </span>
            ))
          )}
        </text>
      </box>

      {/* Line 3: artifact chips */}
      {message.artifactNames.length > 0 && (
        <box style={{ flexDirection: 'row', gap: 1 }}>
          <text style={{ fg: theme.muted, wrapMode: 'none' }}>Artifacts:</text>
          {message.artifactNames.map((name) => (
            <ArtifactChip
              key={name}
              name={name}
              onClick={onArtifactClick ? () => onArtifactClick(name) : undefined}
            />
          ))}
        </box>
      )}

      {/* Clickable details link */}
      <Button
        onClick={() => onExpand(message.forkId)}
        onMouseOver={() => setIsDetailsHovered(true)}
        onMouseOut={() => setIsDetailsHovered(false)}
      >
        <text style={{ fg: isDetailsHovered ? theme.foreground : theme.muted, wrapMode: 'none' }}>
          View details →
        </text>
      </Button>
    </box>
  )
})
