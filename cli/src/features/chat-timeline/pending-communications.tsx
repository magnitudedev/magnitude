import { TextAttributes } from '@opentui/core'
import { Option } from 'effect'
import type { AgentCommunicationMessage, PendingInboundCommunication } from '@magnitudedev/sdk'
import { useTheme } from '../../hooks/use-theme'
import { BOX_CHARS } from '../../utils/ui-constants'
import { AgentCommunicationCard } from './messages/agent-communication-card'

interface PendingCommunicationsPanelProps {
  messages: readonly PendingInboundCommunication[]
  onFileClick?: (path: string, section?: string) => void
}

export function PendingCommunicationsPanel({ messages, onFileClick }: PendingCommunicationsPanelProps) {
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
          const cardMessage: AgentCommunicationMessage = {
            id: message.id,
            type: 'agent_communication',
            streamId: Option.none(),
            direction: message.direction,
            agentId: message.agentId,
            agentName: message.agentName,
            agentRole: message.agentRole,
            forkId: message.forkId,
            content: message.content,
            preview: message.preview,
            timestamp: message.timestamp,
            status: Option.none(),
          }

          return (
            <box key={message.id} style={{ marginTop: 1 }}>
              <AgentCommunicationCard message={cardMessage} onFileClick={onFileClick} />
            </box>
          )
        })}
      </box>
    </box>
  )
}
