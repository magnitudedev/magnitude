import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import type { AgentsViewArtifactItem } from '@magnitudedev/agent'
import { Button } from '../button'
import { useTheme } from '../../hooks/use-theme'
import { getAgentPalette } from '../../utils/agent-colors'

interface AgentArtifactEventProps {
  item: AgentsViewArtifactItem
  onArtifactClick?: (name: string, section?: string) => void
}

export const AgentArtifactEvent = memo(function AgentArtifactEvent({
  item,
  onArtifactClick,
}: AgentArtifactEventProps) {
  const theme = useTheme()
  const [artifactHovered, setArtifactHovered] = useState(false)
  const palette = getAgentPalette(item.colorIndex)

  const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)
  const actionLabel = item.action === 'wrote' ? 'Wrote' : 'Updated'

  return (
    <box
      style={{
        marginBottom: 1,
        flexDirection: 'row',
        alignItems: 'center',
      }}
    >
      <text style={{ fg: theme.muted, wrapMode: 'none' }}>
        <span fg={palette.border}>{'✎ '}</span>
        <span fg={palette.border} attributes={TextAttributes.BOLD}>{capitalize(item.agentRole)}</span>
        <span fg={theme.muted}>{' ('}{item.agentName}{')'}{' · '}{actionLabel}{' '}</span>
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