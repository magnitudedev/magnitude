import { TextAttributes } from '@opentui/core'
import type { DisplayMessage, PendingInboundCommunicationDisplay } from '@magnitudedev/agent'
import { useTheme } from '../hooks/use-theme'
import { BOX_CHARS } from '../utils/ui-constants'
import { AgentCommunicationCard } from './agent-communication-card'

interface PendingCommunicationsPanelProps {
  messages: readonly PendingInboundCommunicationDisplay[]
}

export function PendingCommunicationsPanel({ messages }: PendingCommunicationsPanelProps) {
  const theme = useTheme()
  const agentMessages = messages.filter((message) => message.source === 'agent')
  if (agentMessages.length === 0) return null

  return (
    <box style={{ marginBottom: 1, paddingLeft: 1, paddingRight: 1 }}>
      <box
        style={{
          borderStyle: 'single',
          borderColor: theme.border,
          customBorderChars: BOX_CHARS,
          flexDirection: 'column',
          paddingLeft: 1,
          paddingRight: 1,
        }}
      >
        <text style={{ fg: theme.muted }} attributes={TextAttributes.BOLD}>Pending messages</text>
        {agentMessages.map((message) => {
          const cardMessage = {
            id: message.id,
            type: 'agent_communication',
            direction: message.direction,
            agentId: message.agentId,
            agentName: message.agentName,
            agentRole: message.agentRole,
            forkId: message.forkId,
            content: message.content,
            preview: message.preview,
            timestamp: message.timestamp,
          } as Extract<DisplayMessage, { type: 'agent_communication' }>

          return (
            <box key={message.id} style={{ marginTop: 1 }}>
              <AgentCommunicationCard message={cardMessage} />
            </box>
          )
        })}
      </box>
    </box>
  )
}