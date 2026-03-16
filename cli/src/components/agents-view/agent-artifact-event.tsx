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

  const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)
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
        <span fg={palette.border} attributes={TextAttributes.BOLD}>{capitalize(item.agentRole)}</span>
        <span fg={theme.muted}>{' ('}{item.agentName}{')'}{' '}{actionLabel}{' '}</span>
      </text>
      <Button
        onClick={onArtifactClick ? () => onArtifactClick(item.artifactName) : undefined}
        onMouseOver={() => setArtifactHovered(true)}
        onMouseOut={() => setArtifactHovered(false)}
      >
        <text style={{ fg: artifactHovered ? theme.link : palette.border, wrapMode: 'none' }}>{'[≡ '}{item.artifactName}{']'}</text>
      </Button>
    </box>
  )
})