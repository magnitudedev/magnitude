import { memo, useMemo, useState } from 'react'
import { Option } from 'effect'
import type { DisplayMessage } from '@magnitudedev/sdk'
import { Button } from '../../../components/button'
import { useTheme } from '../../../hooks/use-theme'
import { useTerminalWidth } from '../../../hooks/use-terminal-width'
import { TextAttributes } from '@opentui/core'
import { MarkdownContent } from '../../../markdown/markdown-content'
import { PREVIEW_LINE_CAP } from '@magnitudedev/client-common'

const EXPANDED_LINE_CAP = 300

type AgentCommunicationMessage = Extract<DisplayMessage, { type: 'agent_communication' }>

export function truncateContentLines(
  content: string,
  maxLines: number,
): { text: string; hiddenCount: number; wasTruncated: boolean } {
  const lines = content.split('\n')
  if (lines.length <= maxLines) {
    return { text: content, hiddenCount: 0, wasTruncated: false }
  }
  return {
    text: lines.slice(0, maxLines).join('\n'),
    hiddenCount: lines.length - maxLines,
    wasTruncated: true,
  }
}

const ExpandedContent = memo(function ExpandedContent({
  content,
  contentWidth,
  onFileClick,
}: {
  content: string
  contentWidth: number
  onFileClick?: (path: string, section?: string) => void
}) {
  const theme = useTheme()
  const { text, hiddenCount, wasTruncated } = useMemo(
    () => truncateContentLines(content, EXPANDED_LINE_CAP),
    [content],
  )

  return (
    <box style={{ flexDirection: 'column' }}>
      <MarkdownContent content={text} contentWidth={contentWidth} onOpenFile={onFileClick} />
      {wasTruncated && (
        <text style={{ fg: theme.muted }}>
          …{hiddenCount} lines hidden. Content capped at {EXPANDED_LINE_CAP} lines
        </text>
      )}
    </box>
  )
})

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

  const agentRole = Option.getOrNull(message.agentRole)
  const roleDisplay = agentRole
    ? `${agentRole.charAt(0).toUpperCase()}${agentRole.slice(1)}`
    : message.agentId

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
                {roleDisplay}
              </span>
            </>
          ) : (
            <>
              <span fg={theme.secondary} attributes={TextAttributes.BOLD}>
                {roleDisplay}
              </span>
              <span attributes={TextAttributes.BOLD}> → </span>
              <span fg={theme.info} attributes={TextAttributes.BOLD}>Lead</span>
            </>
          )}
        </text>
      </box>

      <box style={{ paddingLeft: 2, flexDirection: 'column' }}>
        {expanded ? (
          <ExpandedContent content={message.content} contentWidth={contentWidth} onFileClick={onFileClick} />
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
