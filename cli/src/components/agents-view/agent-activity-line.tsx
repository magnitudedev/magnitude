import { memo, useState, useEffect } from 'react'
import type { AgentsViewActivityStartItem, ActiveActivityEntry, ForkActivityToolCounts } from '@magnitudedev/agent'
import { Button } from '../button'
import { useTheme } from '../../hooks/use-theme'
import { getAgentColorByRole } from '../../utils/agent-colors'
import { LaneGutter, type LaneEntry } from './lane-gutter'

interface AgentActivityLineProps {
  item: AgentsViewActivityStartItem
  onForkExpand: (forkId: string) => void
  lanes?: LaneEntry[]
  activeEntry?: ActiveActivityEntry
}

const PULSE_INTERVAL_MS = 200
const SWEEP_INTERVAL_MS = 200

function sweepColors(name: string, sweepPos: number, baseColor: string, pulse: string[]): string[] {
  if (name.length <= 1) return [pulse[0]!]
  const period = (name.length - 1) * 2
  const mod = sweepPos % period
  const cursor = mod < name.length ? mod : period - mod
  return Array.from(name, (_, i) => {
    const dist = Math.abs(i - cursor)
    if (dist <= 1) return pulse[0]!
    if (dist === 2) return pulse[1]!
    return baseColor
  })
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000)
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

export const AgentActivityLine = memo(function AgentActivityLine({
  item,
  onForkExpand,
  lanes = [],
  activeEntry,
}: AgentActivityLineProps) {
  const theme = useTheme()
  const [nameHovered, setNameHovered] = useState(false)
  const [pulseIndex, setPulseIndex] = useState(0)
  const [sweepPos, setSweepPos] = useState(0)
  const [now, setNow] = useState(() => Date.now())

  const isFinished = !activeEntry
  const palette = getAgentColorByRole(item.agentRole)

  useEffect(() => {
    if (isFinished) return
    const id = setInterval(() => {
      setPulseIndex(prev => (prev + 1) % palette.pulse.length)
    }, PULSE_INTERVAL_MS)
    return () => clearInterval(id)
  }, [palette.pulse.length, isFinished])

  useEffect(() => {
    if (isFinished) return
    const id = setInterval(() => {
      setSweepPos(prev => prev + 1)
    }, SWEEP_INTERVAL_MS)
    return () => clearInterval(id)
  }, [isFinished])

  useEffect(() => {
    if (isFinished) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [isFinished])

  const elapsedMs = isFinished
    ? (item.completedAt! - item.startedAt) + item.priorElapsedMs
    : (now - (activeEntry?.startedAt ?? item.startedAt)) + item.priorElapsedMs

  const currentToolCounts = isFinished
    ? item.finalToolCounts!
    : (activeEntry?.toolCounts ?? item.priorToolCounts)

  const totalTools = getTotalToolCount(currentToolCounts) + getTotalToolCount(item.priorToolCounts)
  const timeStr = formatDuration(elapsedMs)

  return (
    <box
      style={{
        flexDirection: 'row',
        alignItems: 'stretch',
        minHeight: 1,
      }}
    >
      <LaneGutter lanes={lanes} />
      <text style={{ wrapMode: 'none' }}>
        <span fg={isFinished ? palette.border : palette.pulse[pulseIndex]!}>{'◆ '}</span>
      </text>
      <Button
        onClick={() => onForkExpand(item.forkId)}
        onMouseOver={() => setNameHovered(true)}
        onMouseOut={() => setNameHovered(false)}
      >
        <text style={{ wrapMode: 'none' }}>
          {nameHovered
            ? <span fg={palette.pulse[0]}>{item.agentName}</span>
            : isFinished
              ? <span fg={palette.border}>{item.agentName}</span>
              : (() => {
                  const nameColors = sweepColors(item.agentName, sweepPos, palette.border, palette.pulse)
                  return Array.from(item.agentName).map((ch, i) => (
                    <span key={i} fg={nameColors[i]}>{ch}</span>
                  ))
                })()
          }
        </text>
      </Button>
      <text style={{ wrapMode: 'none' }}>
        {isFinished ? (
          <>
            <span fg={theme.muted}>{' finished ('}{timeStr}</span>
            {totalTools > 0 && <span fg={theme.muted}>{' · '}{String(totalTools)}{totalTools === 1 ? ' tool' : ' tools'}</span>}
            <span fg={theme.muted}>{')'}</span>
          </>
        ) : (
          <>
            <span fg={theme.muted}>{item.isResume ? ' (resumed) running...' : ' running...'}</span>
            <span fg={theme.muted}>{' ('}{timeStr}</span>
            {totalTools > 0 && <span fg={theme.muted}>{' · '}{String(totalTools)}{totalTools === 1 ? ' tool' : ' tools'}</span>}
            <span fg={theme.muted}>{')'}</span>
          </>
        )}
      </text>
    </box>
  )
})