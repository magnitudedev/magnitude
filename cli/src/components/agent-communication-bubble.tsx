import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import type { DisplayMessage } from '@magnitudedev/agent'
import { ORCHESTRATOR_COLOR, AGENT_BG_COLORS } from '@magnitudedev/agent'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'
import { BOX_CHARS } from '../utils/ui-constants'
import { MarkdownContent } from './markdown-content'

type AgentCommunicationMessage = Extract<DisplayMessage, { type: 'agent_communication' }>

interface AgentCommunicationBubbleProps {
  message: AgentCommunicationMessage
  onArtifactClick?: (name: string, section?: string) => void
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

const ArtifactAttachmentLink = memo(function ArtifactAttachmentLink({
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
      <text style={{ fg: isHovered ? theme.link : theme.primary, wrapMode: 'none' }}>{'[≡ '}{name}{']'}</text>
    </Button>
  )
})

function needsCollapse(content: string): boolean {
  const newlineCount = (content.match(/\n/g) ?? []).length
  return newlineCount > 3 || content.length > 300
}

function truncateToLines(content: string, lines: number): string {
  const parts = content.split('\n')
  return parts.slice(0, lines).join('\n')
}

export const AgentCommunicationBubble = memo(function AgentCommunicationBubble({
  message,
  onArtifactClick,
}: AgentCommunicationBubbleProps) {
  const theme = useTheme()
  const [isExpanded, setIsExpanded] = useState(false)
  const [isToggleHovered, setIsToggleHovered] = useState(false)

  const isToAgent = message.direction === 'to_agent'
  const agentName = message.agentName ?? message.agentId
  const agentRole = message.agentRole ? capitalize(message.agentRole) : 'Agent'
  const borderColor = message.agentColor
  const bgColor = AGENT_BG_COLORS[message.agentColor] ?? '#151520'

  const content = String(message.content ?? '')
  const collapsible = needsCollapse(content)
  const displayContent = collapsible && !isExpanded ? truncateToLines(content, 3) : content

  return (
    <box
      style={{
        flexGrow: 1,
        marginBottom: 1,
        marginLeft: 3,
        marginRight: 3,
        flexDirection: 'column',
      }}
    >
      {/* Outer box: border only */}
      <box
        style={{
          borderStyle: 'single',
          border: ['left'],
          borderColor,
          customBorderChars: { ...BOX_CHARS, vertical: '┃' },
        }}
      >
        {/* Inner box: background + padding */}
        <box
          style={{
            flexDirection: 'column',
            backgroundColor: bgColor,
            paddingLeft: 1,
            paddingRight: 2,
          }}
        >
          {/* Header line */}
          <text style={{ wrapMode: 'none' }}>
            {isToAgent ? (
              <>
                <span fg={theme.foreground} attributes={TextAttributes.BOLD}>Orchestrator</span>
                <span fg={theme.muted}>{' → '}</span>
                <span fg={theme.foreground} attributes={TextAttributes.BOLD}>{agentRole}</span>
                <span fg={theme.muted}>{' ('}{agentName}{')'}</span>
              </>
            ) : (
              <>
                <span fg={theme.foreground} attributes={TextAttributes.BOLD}>{agentRole}</span>
                <span fg={theme.muted}>{' ('}{agentName}{')'}</span>
                <span fg={theme.muted}>{' → '}</span>
                <span fg={theme.foreground} attributes={TextAttributes.BOLD}>Orchestrator</span>
              </>
            )}
          </text>

          {/* Content — rendered as markdown */}
          <MarkdownContent
            content={displayContent}
            onOpenArtifact={onArtifactClick}
          />

          {/* Expand/Collapse toggle */}
          {collapsible && (
            <Button
              onClick={() => setIsExpanded(v => !v)}
              onMouseOver={() => setIsToggleHovered(true)}
              onMouseOut={() => setIsToggleHovered(false)}
            >
              <text style={{ fg: isToggleHovered ? theme.foreground : theme.muted, wrapMode: 'none' }}>
                {isExpanded ? '▲ Collapse' : '▼ Expand'}
              </text>
            </Button>
          )}

          {/* Attached artifacts — only for to_agent messages */}
          {isToAgent && message.attachedArtifacts.length > 0 && (
            <box style={{ marginTop: 1, flexDirection: 'row', flexWrap: 'wrap' }}>
              <text style={{ wrapMode: 'none' }}><span fg={theme.muted}>{'Attached: '}</span></text>
              {message.attachedArtifacts.map((name, i) => (
                <box key={name} style={{ flexDirection: 'row' }}>
                  {i > 0 && <text style={{ wrapMode: 'none' }}><span fg={theme.muted}>{' '}</span></text>}
                  <ArtifactAttachmentLink
                    name={name}
                    onClick={onArtifactClick ? () => onArtifactClick(name) : undefined}
                  />
                </box>
              ))}
            </box>
          )}
        </box>
      </box>
    </box>
  )
})