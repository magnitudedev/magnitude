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

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

export const AgentActivityLine = memo(function AgentActivityLine({
  item,
  onForkExpand,
  lanes = [],
  isFinished = false,
}: AgentActivityLineProps) {
  const theme = useTheme()
  const [nameHovered, setNameHovered] = useState(false)
  const [pulseIndex, setPulseIndex] = useState(0)

  const palette = getAgentColorByRole(item.agentRole)

  useEffect(() => {
    if (isFinished) return
    const id = setInterval(() => {
      setPulseIndex(prev => (prev + 1) % palette.pulse.length)
    }, PULSE_INTERVAL_MS)
    return () => clearInterval(id)
  }, [palette.pulse.length, isFinished])

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
          <span fg={nameHovered ? palette.pulse[0] : palette.border} attributes={TextAttributes.BOLD}>{capitalize(item.agentRole)}</span>
          <span fg={theme.muted}>{' ('}{item.agentName}{')'}</span>
        </text>
      </Button>
      <text style={{ wrapMode: 'none' }}>
        <span fg={theme.muted}>{' started'}</span>
      </text>
    </box>
  )
})