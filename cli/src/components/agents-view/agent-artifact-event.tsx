import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import type { AgentsViewArtifactItem } from '@magnitudedev/agent'
import { Button } from '../button'
import { useTheme } from '../../hooks/use-theme'
import { getAgentColorByRole } from '../../utils/agent-colors'
import { LaneGutter, type LaneEntry } from './lane-gutter'

interface AgentArtifactEventProps {
  item: AgentsViewArtifactItem
  onArtifactClick?: (name: string, section?: string) => void
  lanes?: LaneEntry[]
}

export const AgentArtifactEvent = memo(function AgentArtifactEvent({
  item,
  onArtifactClick,
  lanes = [],
}: AgentArtifactEventProps) {
  const theme = useTheme()
  const [artifactHovered, setArtifactHovered] = useState(false)
  const palette = getAgentColorByRole(item.agentRole)

  const actionLabel = item.action === 'wrote' ? 'created' : 'updated'

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
        <span fg={palette.border}>{'✎ '}</span>
        <span fg={palette.border}>{item.agentName}</span>
        <span fg={theme.muted}>{' '}{actionLabel}{' '}</span>
      </text>
      <Button
        onClick={onArtifactClick ? () => onArtifactClick(item.artifactName) : undefined}
        onMouseOver={() => setArtifactHovered(true)}
        onMouseOut={() => setArtifactHovered(false)}
      >
        <text style={{ fg: artifactHovered ? theme.link : theme.primary, wrapMode: 'none' }}>{'[≡ '}{item.artifactName}{']'}</text>
      </Button>
    </box>
  )
})