import { memo, useMemo, useState } from 'react'
import type { DisplayMessage } from '@magnitudedev/agent'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'
import { useTerminalWidth } from '../hooks/use-terminal-width'
import { PREVIEW_LINE_CAP, wrapTextToVisualLines } from '../utils/strings'
import { TextAttributes } from '@opentui/core'

type AgentCommunicationMessage = Extract<DisplayMessage, { type: 'agent_communication' }>

export function getCommunicationPreview(
  content: string,
  contentWidth: number,
): { previewLines: string[]; hasOverflow: boolean } {
  const wrappedLines = wrapTextToVisualLines(content, contentWidth)
  return {
    previewLines: wrappedLines.slice(0, PREVIEW_LINE_CAP),
    hasOverflow: wrappedLines.length > PREVIEW_LINE_CAP,
  }
}

interface AgentCommunicationCardProps {
  message: AgentCommunicationMessage
}

export const AgentCommunicationCard = memo(function AgentCommunicationCard({ message }: AgentCommunicationCardProps) {
  const theme = useTheme()
  const [expanded, setExpanded] = useState(false)
  const [hovered, setHovered] = useState(false)
  const terminalWidth = useTerminalWidth()
  const contentWidth = Math.max(1, terminalWidth - 6)

  const { previewLines, hasOverflow } = useMemo(
    () => getCommunicationPreview(message.content, contentWidth),
    [message.content, contentWidth]
  )

  return (
    <box style={{ alignSelf: 'flex-start', marginBottom: 1, flexDirection: 'column' }}>
      <box>
        <text>
          <span attributes={TextAttributes.BOLD}>✉ </span>
          {message.direction === 'from_agent' ? (
            <>
              <span fg={theme.info} attributes={TextAttributes.BOLD}>Main agent</span>
              <span attributes={TextAttributes.BOLD}> → </span>
              <span fg={theme.secondary} attributes={TextAttributes.BOLD}>{message.agentId}</span>
            </>
          ) : (
            <>
              <span fg={theme.secondary} attributes={TextAttributes.BOLD}>{message.agentId}</span>
              <span attributes={TextAttributes.BOLD}> → </span>
              <span fg={theme.info} attributes={TextAttributes.BOLD}>Main agent</span>
            </>
          )}
        </text>
      </box>

      <box style={{ paddingLeft: 2, flexDirection: 'column' }}>
        {expanded ? (
          <text style={{ fg: theme.foreground }}>{message.content}</text>
        ) : (
          previewLines.map((line, index) => (
            <text key={`${message.id}-line-${index}`} style={{ fg: theme.foreground }}>{line}</text>
          ))
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