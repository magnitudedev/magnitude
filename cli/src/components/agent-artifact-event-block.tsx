import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import type { DisplayMessage } from '@magnitudedev/agent'
import { AGENT_BG_COLORS } from '@magnitudedev/agent'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'
import { BOX_CHARS } from '../utils/ui-constants'

type AgentArtifactEventMessage = Extract<DisplayMessage, { type: 'agent_artifact_event' }>

interface AgentArtifactEventBlockProps {
  message: AgentArtifactEventMessage
  onArtifactClick?: (name: string, section?: string) => void
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

const ArtifactLink = memo(function ArtifactLink({
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
      <text style={{ fg: isHovered ? theme.link : theme.primary, wrapMode: 'none' }}>
        {'[≡ '}{name}{']'}
      </text>
    </Button>
  )
})

export const AgentArtifactEventBlock = memo(function AgentArtifactEventBlock({
  message,
  onArtifactClick,
}: AgentArtifactEventBlockProps) {
  const theme = useTheme()

  const agentRole = capitalize(message.agentRole)
  const operationLabel = message.operation === 'wrote' ? 'Wrote' : 'Updated'

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
          borderColor: message.agentColor,
          customBorderChars: { ...BOX_CHARS, vertical: '┃' },
        }}
      >
        {/* Inner box: background + padding */}
        <box
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            backgroundColor: AGENT_BG_COLORS[message.agentColor] ?? '#151520',
            paddingLeft: 1,
            paddingRight: 2,
          }}
        >
          <text style={{ wrapMode: 'none' }}>
            <span fg={theme.foreground} attributes={TextAttributes.BOLD}>{agentRole}</span>
            <span fg={theme.muted}>{' ('}{message.agentName}{')'}</span>
            <span fg={theme.muted}>{` · ${operationLabel} `}</span>
          </text>
          <ArtifactLink
            name={message.artifactName}
            onClick={onArtifactClick ? () => onArtifactClick(message.artifactName) : undefined}
          />
        </box>
      </box>
    </box>
  )
})