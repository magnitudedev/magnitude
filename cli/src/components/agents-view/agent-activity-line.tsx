import { memo, useState, useEffect, useRef } from 'react'
import { TextAttributes } from '@opentui/core'
import type { AgentsViewActivityItem } from '@magnitudedev/agent'
import type { ForkActivityToolCounts } from '@magnitudedev/agent'
import { Button } from '../button'
import { useTheme } from '../../hooks/use-theme'
import { getAgentPalette } from '../../utils/agent-colors'

interface AgentActivityLineProps {
  item: AgentsViewActivityItem
  onForkExpand: (forkId: string) => void
}

const PULSE_INTERVAL_MS = 200

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

const TOOL_DISPLAY: { key: keyof ForkActivityToolCounts; verb: string; noun: string; nounPlural: string }[] = [
  { key: 'reads', verb: 'Read', noun: 'file', nounPlural: 'files' },
  { key: 'writes', verb: 'Wrote', noun: 'file', nounPlural: 'files' },
  { key: 'edits', verb: 'Edited', noun: 'file', nounPlural: 'files' },
  { key: 'commands', verb: 'Ran', noun: 'command', nounPlural: 'commands' },
  { key: 'webSearches', verb: 'Searched', noun: 'web', nounPlural: 'web searches' },
  { key: 'webFetches', verb: 'Fetched', noun: 'page', nounPlural: 'pages' },
  { key: 'navigations', verb: 'Visited', noun: 'page', nounPlural: 'pages' },
  { key: 'clicks', verb: 'Clicked', noun: 'element', nounPlural: 'elements' },
  { key: 'inputs', verb: 'Typed', noun: 'input', nounPlural: 'inputs' },
  { key: 'evaluations', verb: 'Ran', noun: 'script', nounPlural: 'scripts' },
]

export const AgentActivityLine = memo(function AgentActivityLine({
  item,
  onForkExpand,
}: AgentActivityLineProps) {
  const theme = useTheme()
  const [nameHovered, setNameHovered] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [pulseIndex, setPulseIndex] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pulseRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const palette = getAgentPalette(item.colorIndex)
  const isActive = item.status === 'active'

  useEffect(() => {
    if (isActive) {
      const update = () => setElapsed(Math.floor((Date.now() - item.startedAt) / 1000))
      update()
      timerRef.current = setInterval(update, 1000)
      return () => { if (timerRef.current) clearInterval(timerRef.current) }
    } else if (item.completedAt) {
      setElapsed(Math.floor((item.completedAt - item.startedAt) / 1000))
    }
  }, [isActive, item.startedAt, item.completedAt])

  useEffect(() => {
    if (isActive) {
      pulseRef.current = setInterval(() => {
        setPulseIndex(prev => (prev + 1) % palette.pulse.length)
      }, PULSE_INTERVAL_MS)
      return () => { if (pulseRef.current) clearInterval(pulseRef.current) }
    }
  }, [isActive, palette.pulse.length])

  const diamondColor = isActive ? palette.pulse[pulseIndex]! : palette.border
  const timeStr = formatDuration(elapsed)
  const totalTools = getTotalToolCount(item.toolCounts)

  const toolParts: { verb: string; count: number; noun: string }[] = []
  for (const { key, verb, noun, nounPlural } of TOOL_DISPLAY) {
    const count = item.toolCounts[key]
    if (count > 0) {
      toolParts.push({ verb, count, noun: count === 1 ? noun : nounPlural })
    }
  }

  const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

  return (
    <box
      style={{
        marginBottom: 1,
        flexDirection: 'row',
        alignItems: 'flex-start',
      }}
    >
      <text style={{ wrapMode: 'none' }}>
        <span fg={diamondColor}>{'◆ '}</span>
      </text>
      <Button
        onClick={() => onForkExpand(item.forkId)}
        onMouseOver={() => setNameHovered(true)}
        onMouseOut={() => setNameHovered(false)}
      >
        <text style={{ wrapMode: 'none' }}>
          <span fg={nameHovered ? palette.pulse[0] : palette.border} attributes={TextAttributes.BOLD}>{capitalize(item.agentRole)}</span>
          <span fg={theme.muted}>{' ('}{item.agentName}{')'}</span>
        </text>
      </Button>
      <text style={{ wrapMode: 'none' }}>
        <span fg={theme.muted}>{' · '}{timeStr}</span>
        {totalTools > 0 ? (
          <span fg={theme.muted}>{' · '}</span>
        ) : null}
        {totalTools > 0 ? (
          <span fg={theme.info}>{String(totalTools)}</span>
        ) : null}
        {totalTools > 0 ? (
          <span fg={theme.muted}>{totalTools === 1 ? ' tool' : ' tools'}</span>
        ) : null}
        {toolParts.length > 0 ? (
          <span fg={theme.muted}>{' · '}</span>
        ) : null}
        {toolParts.flatMap((part, i) => [
          i > 0 ? <span key={`sep-${i}`} fg={theme.muted}>{', '}</span> : null,
          <span key={`verb-${i}`} fg={theme.muted}>{part.verb}{' '}</span>,
          <span key={`count-${i}`} fg={theme.info}>{String(part.count)}</span>,
          <span key={`noun-${i}`} fg={theme.muted}>{' '}{part.noun}</span>,
        ].filter(Boolean))}
      </text>
    </box>
  )
})