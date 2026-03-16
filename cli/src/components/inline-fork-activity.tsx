import { memo, useState, useEffect, useRef } from 'react'
import { TextAttributes } from '@opentui/core'
import type { ForkActivityMessage, ForkActivityToolCounts } from '@magnitudedev/agent'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'
import { AGENT_BG_COLORS } from '@magnitudedev/agent'
import { BOX_CHARS } from '../utils/ui-constants'
import { violet, rose, indigo } from '../utils/palette'

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
    + counts.webSearches + counts.webFetches + counts.artifactWrites + counts.artifactUpdates
    + counts.searches + counts.clicks + counts.navigations + counts.inputs
    + counts.evaluations + counts.other
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

const PULSE_INTERVAL_MS = 200

// Pulse shades: smooth wave 300→400→500→600→700→600→500→400→300 (9 steps, ~1.8s cycle)
const PULSE_PALETTES: Record<string, string[]> = {
  [violet[500]]: [violet[300], violet[400], violet[500], violet[600], violet[700], violet[600], violet[500], violet[400], violet[300]],
  [rose[500]]:   [rose[300],   rose[400],   rose[500],   rose[600],   rose[700],   rose[600],   rose[500],   rose[400],   rose[300]],
  [indigo[500]]: [indigo[300], indigo[400], indigo[500], indigo[600], indigo[700], indigo[600], indigo[500], indigo[400], indigo[300]],

}

function getPulseColors(agentColor: string): string[] {
  return PULSE_PALETTES[agentColor] ?? [agentColor]
}

// Visible tool categories with their display format
const TOOL_DISPLAY: { key: keyof ForkActivityToolCounts; verb: string; noun: string; nounPlural: string }[] = [
  { key: 'reads', verb: 'Read', noun: 'file', nounPlural: 'files' },
  { key: 'writes', verb: 'Wrote', noun: 'file', nounPlural: 'files' },
  { key: 'edits', verb: 'Edited', noun: 'file', nounPlural: 'files' },
  { key: 'commands', verb: 'Ran', noun: 'command', nounPlural: 'commands' },
  { key: 'webSearches', verb: 'Ran', noun: 'web search', nounPlural: 'web searches' },
  { key: 'webFetches', verb: 'Fetched', noun: 'page', nounPlural: 'pages' },
  { key: 'navigations', verb: 'Visited', noun: 'page', nounPlural: 'pages' },
  { key: 'clicks', verb: 'Clicked', noun: 'element', nounPlural: 'elements' },
  { key: 'inputs', verb: 'Typed', noun: 'input', nounPlural: 'inputs' },
  { key: 'evaluations', verb: 'Ran', noun: 'script', nounPlural: 'scripts' },
]

export const InlineForkActivity = memo(function InlineForkActivity({
  message,
  onExpand,
}: InlineForkActivityProps) {
  const theme = useTheme()
  const [showHovered, setShowHovered] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [pulseIndex, setPulseIndex] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pulseRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const isRunning = message.status === 'running'
  const agentColor = message.agentColor ?? theme.info

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

  // Smooth pulsing border animation
  useEffect(() => {
    if (isRunning) {
      const pulseColors = getPulseColors(agentColor)
      pulseRef.current = setInterval(() => {
        setPulseIndex(prev => (prev + 1) % pulseColors.length)
      }, PULSE_INTERVAL_MS)
      return () => {
        if (pulseRef.current) clearInterval(pulseRef.current)
      }
    }
  }, [isRunning, agentColor])

  // Determine border color: smooth pulse when running, static agent color when done
  const pulseColors = getPulseColors(agentColor)
  const borderColor = isRunning
    ? pulseColors[pulseIndex % pulseColors.length]!
    : agentColor

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
        flexGrow: 1,
        marginBottom: 1,
        marginLeft: 3,
        marginRight: 3,
      }}
    >
      {/* Outer box: border only */}
      <box
        style={{
          borderStyle: 'single',
          border: ['left'],
          borderColor,
          customBorderChars: { ...BOX_CHARS, vertical: '┃' },
        }}
      >
      {/* Inner box: background + padding */}
      <box
        style={{
          flexDirection: 'column',
          backgroundColor: AGENT_BG_COLORS[agentColor] ?? '#151520',
          paddingLeft: 1,
          paddingRight: 2,
        }}
      >
      {/* Line 1: role (name) · timer · tools · Show */}
      <box style={{ flexDirection: 'row' }}>
        <text style={{ wrapMode: 'none' }}>
          <span fg={theme.foreground} attributes={TextAttributes.BOLD}>{roleLabel}</span>
          <span fg={theme.muted}>{' ('}{displayName}{')'}</span>
          {resumed ? <span fg={theme.muted}> (resumed)</span> : null}
          <span fg={theme.muted}>{' · '}{timeStr}</span>
          {totalTools > 0 ? <span fg={theme.muted}>{' · '}</span> : null}
          {totalTools > 0 ? <span fg={theme.info}>{String(totalTools)}</span> : null}
          {totalTools > 0 ? <span fg={theme.muted}>{totalTools === 1 ? ' tool' : ' tools'}</span> : null}
        </text>
        <Button
          onClick={() => onExpand(message.forkId)}
          onMouseOver={() => setShowHovered(true)}
          onMouseOut={() => setShowHovered(false)}
        >
          <text style={{ fg: showHovered ? theme.foreground : theme.muted, wrapMode: 'none' }}>{' · Show'}</text>
        </Button>
      </box>

      {/* Line 2: tool summary — all spans are direct children of <text> */}
      <text style={{ wrapMode: 'none' }}>
        {toolParts.length === 0 ? (
          <span fg={theme.muted}>Starting...</span>
        ) : (
          toolParts.flatMap((part, i) => [
            i > 0 ? <span key={`sep-${i}`} fg={theme.muted}>, </span> : null,
            <span key={`verb-${i}`} fg={theme.muted}>{part.verb} </span>,
            <span key={`count-${i}`} fg={theme.info}>{String(part.count)}</span>,
            <span key={`noun-${i}`} fg={theme.muted}> {part.noun}</span>,
          ].filter(Boolean))
        )}
      </text>
      </box>
      </box>
    </box>
  )
})