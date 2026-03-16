import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import type { AgentsViewMessageItem } from '@magnitudedev/agent'
import { Button } from '../button'
import { MarkdownContent } from '../markdown-content'
import { useTheme } from '../../hooks/use-theme'
import { getAgentColorByRole } from '../../utils/agent-colors'
import { BOX_CHARS } from '../../utils/ui-constants'
import { LaneGutter, type LaneEntry } from './lane-gutter'

interface AgentMessageBubbleProps {
  item: AgentsViewMessageItem
  onArtifactClick?: (name: string, section?: string) => void
  lanes?: LaneEntry[]
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
  lanes = [],
}: AgentMessageBubbleProps) {
  const theme = useTheme()
  const [collapsed, setCollapsed] = useState(true)
  const [expandHovered, setExpandHovered] = useState(false)

  const isSenderOrchestrator = item.direction === 'to_agent'

  // Derive palettes from role
  const orchestratorPalette = getAgentColorByRole('orchestrator')
  const agentPalette = isSenderOrchestrator
    ? getAgentColorByRole(item.toRole)
    : getAgentColorByRole(item.fromRole)


  const long = isLong(item.content)
  const displayContent = long && collapsed ? truncate(item.content) : item.content

  const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

  return (
    <box style={{ flexDirection: 'row', alignItems: 'stretch' }}>
      <LaneGutter lanes={lanes} />
      <box
        style={{
          flexGrow: 1,
        }}
      >
      <box
        style={{
          flexDirection: 'column',
        }}
      >
      {/* Header */}
      <text style={{ wrapMode: 'none' }}>
        <span fg={theme.muted}>{'⌲ '}</span>
        {isSenderOrchestrator ? (
          <>
            <span fg={orchestratorPalette.border} attributes={TextAttributes.BOLD}>{'Orchestrator'}</span>
            <span fg={theme.muted}>{' → '}</span>
            <span fg={agentPalette.border} attributes={TextAttributes.BOLD}>{capitalize(item.toRole)}</span>
            <span fg={theme.muted}>{' ('}{item.toName}{')'}</span>
          </>
        ) : (
          <>
            <span fg={agentPalette.border} attributes={TextAttributes.BOLD}>{capitalize(item.fromRole)}</span>
            <span fg={theme.muted}>{' ('}{item.fromName}{')'}</span>
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
    </box>
  )
})