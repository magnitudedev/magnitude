import { memo, useState } from 'react'
import type { DisplayMessage } from '@magnitudedev/agent'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'
import { TextAttributes } from '@opentui/core'

type AgentCommunicationMessage = Extract<DisplayMessage, { type: 'agent_communication' }>

interface AgentCommunicationCardProps {
  message: AgentCommunicationMessage
}

export const AgentCommunicationCard = memo(function AgentCommunicationCard({ message }: AgentCommunicationCardProps) {
  const theme = useTheme()
  const [expanded, setExpanded] = useState(false)
  const [hovered, setHovered] = useState(false)

  const agentName = message.agentName ?? message.agentId
  const agentRole = message.agentRole ? message.agentRole.charAt(0).toUpperCase() + message.agentRole.slice(1) : 'Agent'
  const isToAgent = message.direction === 'to_agent'
  const directionArrow = isToAgent ? '→' : '←'
  const directionPrefix = isToAgent ? 'To' : 'From'
  const isOrchestrator = message.agentName === 'Orchestrator' || message.agentRole === 'Orchestrator'

  const toggleColor = hovered ? theme.foreground : theme.muted

  return (
    <box
      style={{
        alignSelf: 'flex-start',
        marginBottom: 1,
        flexDirection: 'column',
      }}
    >
      <box style={{ flexDirection: 'row' }}>
        <Button
          onClick={() => setExpanded(prev => !prev)}
          onMouseOver={() => setHovered(true)}
          onMouseOut={() => setHovered(false)}
        >
          <text style={{ wrapMode: 'none' }}>
            <span fg={theme.info}>{directionArrow} ✉ </span>
            <span fg={theme.muted}>{directionPrefix} </span>
            {isOrchestrator ? (
              <span fg={theme.foreground} attributes={TextAttributes.BOLD}>Orchestrator</span>
            ) : (
              <>
                <span fg={theme.foreground} attributes={TextAttributes.BOLD}>{agentRole}: </span>
                <span fg={theme.foreground}>{agentName}</span>
              </>
            )}
            <span fg={toggleColor}> · {expanded ? 'Hide' : 'Show'}</span>
          </text>
        </Button>
      </box>

      {expanded ? (
        <box style={{ paddingLeft: 3, flexDirection: 'column' }}>
          <text style={{ fg: theme.foreground }}>
            {message.content}
          </text>
        </box>
      ) : null}
    </box>
  )
})