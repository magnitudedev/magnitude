import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import type { AgentsViewMessageItem } from '@magnitudedev/agent'
import { Button } from '../button'
import { MarkdownContent } from '../markdown-content'
import { useTheme } from '../../hooks/use-theme'
import { getAgentPalette, ORCHESTRATOR_PALETTE } from '../../utils/agent-colors'
import { BOX_CHARS } from '../../utils/ui-constants'

interface AgentMessageBubbleProps {
  item: AgentsViewMessageItem
  onArtifactClick?: (name: string, section?: string) => void
}

function isLong(content: string): boolean {
  return content.split('\n').length > 3 || content.length > 300
}

function truncate(content: string): string {
  const lines = content.split('\n')
  if (lines.length > 3) return lines.slice(0, 3).join('\n')
  if (content.length > 300) return content.slice(0, 300)
  return content
}

const ArtifactChip = memo(function ArtifactChip({
  name,
  onClick,
}: {
  name: string
  onClick?: () => void
}) {
  const theme = useTheme()
  const [hovered, setHovered] = useState(false)

  return (
    <Button
      onClick={onClick}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
    >
      <text style={{ fg: hovered ? theme.link : theme.primary, wrapMode: 'none' }}>{'[≡ '}{name}{']'}</text>
    </Button>
  )
})

export const AgentMessageBubble = memo(function AgentMessageBubble({
  item,
  onArtifactClick,
}: AgentMessageBubbleProps) {
  const theme = useTheme()
  const [collapsed, setCollapsed] = useState(true)
  const [expandHovered, setExpandHovered] = useState(false)

  const isSenderOrchestrator = item.direction === 'to_agent'
  const senderPalette = isSenderOrchestrator
    ? ORCHESTRATOR_PALETTE
    : getAgentPalette(item.fromColorIndex!)
  const borderColor = senderPalette.border
  const bgColor = senderPalette.bg

  const long = isLong(item.content)
  const displayContent = long && collapsed ? truncate(item.content) : item.content

  // Header: "Orchestrator → Role (name)" or "Role (name) → Orchestrator"
  const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)
  const fromLabel = item.fromColorIndex === null
    ? 'Orchestrator'
    : capitalize(item.fromRole)
  const fromName = item.fromColorIndex === null ? null : item.fromName
  const toLabel = item.toColorIndex === null
    ? 'Orchestrator'
    : capitalize(item.toRole)
  const toName = item.toColorIndex === null ? null : item.toName

  return (
    <box
      style={{
        marginBottom: 1,
        borderStyle: 'single',
        border: ['left'],
        borderColor,
        customBorderChars: { ...BOX_CHARS, vertical: '┃' },
      }}
    >
      <box
        style={{
          paddingLeft: 1,
          flexDirection: 'column',
          backgroundColor: bgColor,
        }}
      >
      {/* Header */}
      <text style={{ wrapMode: 'none' }}>
        {isSenderOrchestrator ? (
          <>
            <span fg={theme.foreground} attributes={TextAttributes.BOLD}>{fromLabel}</span>
            <span fg={theme.muted}>{' → '}</span>
            <span fg={theme.foreground} attributes={TextAttributes.BOLD}>{toLabel}</span>
            {toName ? <span fg={theme.muted}>{' ('}{toName}{')'}</span> : null}
          </>
        ) : (
          <>
            <span fg={theme.foreground} attributes={TextAttributes.BOLD}>{fromLabel}</span>
            {fromName ? <span fg={theme.muted}>{' ('}{fromName}{')'}</span> : null}
          </>
        )}
      </text>

      {/* Content */}
      <MarkdownContent
        content={displayContent}
        onOpenArtifact={onArtifactClick}
      />

      {/* Expand/collapse toggle */}
      {long ? (
        <Button
          onClick={() => setCollapsed(c => !c)}
          onMouseOver={() => setExpandHovered(true)}
          onMouseOut={() => setExpandHovered(false)}
        >
          <text style={{ fg: expandHovered ? theme.foreground : theme.muted, wrapMode: 'none' }}>
            {collapsed ? '▼ Expand' : '▲ Collapse'}
          </text>
        </Button>
      ) : null}

      {/* Attached artifacts */}
      {item.attachedArtifacts.length > 0 ? (
        <box style={{ flexDirection: 'row', gap: 1 }}>
          <text style={{ fg: theme.muted, wrapMode: 'none' }}>Attached:</text>
          {item.attachedArtifacts.map((name) => (
            <ArtifactChip
              key={name}
              name={name}
              onClick={onArtifactClick ? () => onArtifactClick(name) : undefined}
            />
          ))}
        </box>
      ) : null}
      </box>
    </box>
  )
})