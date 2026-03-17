import { memo, useState, useEffect } from 'react'
import type { AgentsViewActivityStartItem } from '@magnitudedev/agent'
import { Button } from '../button'
import { useTheme } from '../../hooks/use-theme'
import { getAgentColorByRole } from '../../utils/agent-colors'
import { TextAttributes } from '@opentui/core'
import { LaneGutter, type LaneEntry } from './lane-gutter'

interface AgentActivityLineProps {
  item: AgentsViewActivityStartItem
  onForkExpand: (forkId: string) => void
  lanes?: LaneEntry[]
  isFinished?: boolean
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
    if (dist === 0) return pulse[0]!   // center: shade 300
    if (dist === 1) return pulse[1]!   // adjacent: shade 400
    return baseColor                    // everything else: normal color
  })
}




export const AgentActivityLine = memo(function AgentActivityLine({
  item,
  onForkExpand,
  lanes = [],
  isFinished = false,
}: AgentActivityLineProps) {
  const theme = useTheme()
  const [nameHovered, setNameHovered] = useState(false)
  const [pulseIndex, setPulseIndex] = useState(0)
  const [sweepPos, setSweepPos] = useState(0)

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
        <span fg={theme.muted}>{' started'}</span>
      </text>
    </box>
  )
})