import { memo, useMemo, useState } from 'react'
import type { DisplayMessage } from '@magnitudedev/agent'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'
import { useTerminalWidth } from '../hooks/use-terminal-width'
import { TextAttributes } from '@opentui/core'
import { formatSubagentIdWithEmoji } from '../utils/subagent-role-emoji'
import { MarkdownContent } from '../markdown/markdown-content'
import { PREVIEW_LINE_CAP } from '../utils/strings'

type AgentCommunicationMessage = Extract<DisplayMessage, { type: 'agent_communication' }>

export function getCommunicationPreview(
  content: string,
  _contentWidth: number,
): { previewLines: string[]; hasOverflow: boolean } {
  const lines = content.split('\n')
  return {
    previewLines: lines.slice(0, PREVIEW_LINE_CAP),
    hasOverflow: lines.length > PREVIEW_LINE_CAP,
  }
}

interface AgentCommunicationCardProps {
  message: AgentCommunicationMessage
  widthAdjustment?: number
  onFileClick?: (path: string, section?: string) => void
}

export const AgentCommunicationCard = memo(function AgentCommunicationCard({
  message,
  widthAdjustment = 0,
  onFileClick,
}: AgentCommunicationCardProps) {
  const theme = useTheme()
  const [expanded, setExpanded] = useState(false)
  const [hovered, setHovered] = useState(false)
  const terminalWidth = useTerminalWidth()
  const contentWidth = Math.max(1, terminalWidth - 6 - widthAdjustment)

  const { previewContent, hasOverflow } = useMemo(() => {
    const { previewLines, hasOverflow } = getCommunicationPreview(message.content, contentWidth)
    return {
      previewContent: previewLines.join('\n'),
      hasOverflow,
    }
  }, [message.content, contentWidth])

  return (
    <box style={{ alignSelf: 'flex-start', flexDirection: 'column' }}>
      <box>
        <text>
          <span attributes={TextAttributes.BOLD}>✉ </span>
          {message.direction === 'from_agent' ? (
            <>
              <span fg={theme.info} attributes={TextAttributes.BOLD}>Lead</span>
              <span attributes={TextAttributes.BOLD}> → </span>
              <span fg={theme.secondary} attributes={TextAttributes.BOLD}>
                {formatSubagentIdWithEmoji(message.agentId, message.agentRole)}
              </span>
            </>
          ) : (
            <>
              <span fg={theme.secondary} attributes={TextAttributes.BOLD}>
                {formatSubagentIdWithEmoji(message.agentId, message.agentRole)}
              </span>
              <span attributes={TextAttributes.BOLD}> → </span>
              <span fg={theme.info} attributes={TextAttributes.BOLD}>Lead</span>
            </>
          )}
        </text>
      </box>

      <box style={{ paddingLeft: 2, flexDirection: 'column' }}>
        {expanded ? (
          <MarkdownContent content={message.content} contentWidth={contentWidth} onOpenFile={onFileClick} />
        ) : (
          <MarkdownContent content={previewContent} contentWidth={contentWidth} onOpenFile={onFileClick} />
        )}

        {hasOverflow ? (
          <Button
            onClick={() => setExpanded(prev => !prev)}
            onMouseOver={() => setHovered(true)}
            onMouseOut={() => setHovered(false)}
          >
            <text style={{ fg: hovered ? theme.foreground : theme.muted }}>
              {expanded ? '⌃ Collapse' : '⌄ Expand'}
            </text>
          </Button>
        ) : null}
      </box>
    </box>
  )
})
