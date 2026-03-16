import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'
import { getAgentPalette } from '../utils/agent-colors'

interface AgentNotificationProps {
  type: 'started' | 'completed'
  agentRole: string
  agentName: string
  colorIndex: number
  durationSeconds?: number
  totalTools?: number
  onViewAgents: () => void
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}

export const AgentNotification = memo(function AgentNotification({
  type,
  agentRole,
  agentName,
  colorIndex,
  durationSeconds,
  totalTools,
  onViewAgents,
}: AgentNotificationProps) {
  const theme = useTheme()
  const [isLinkHovered, setIsLinkHovered] = useState(false)
  const palette = getAgentPalette(colorIndex)

  return (
    <box style={{ flexDirection: 'row', marginBottom: 1 }}>
      <text style={{ wrapMode: 'none' }}>
        {'  '}
        <span fg={palette.border}>{'◆'}</span>
        {' '}
        <span fg={palette.border} attributes={TextAttributes.BOLD}>{agentRole.charAt(0).toUpperCase() + agentRole.slice(1)}</span>
        <span fg={theme.muted}>{' ('}{agentName}{')'}</span>
        {' '}
        <span fg={theme.muted}>{type === 'started' ? 'Started' : 'Finished'}</span>
        {type === 'completed' && durationSeconds !== undefined ? (
          <span fg={theme.muted}>{' · '}{formatDuration(durationSeconds)}</span>
        ) : null}
        {type === 'completed' && totalTools !== undefined ? (
          <span fg={theme.muted}>{' · '}</span>
        ) : null}
        {type === 'completed' && totalTools !== undefined ? (
          <span fg={theme.info}>{String(totalTools)}</span>
        ) : null}
        {type === 'completed' && totalTools !== undefined ? (
          <span fg={theme.muted}>{totalTools === 1 ? ' tool' : ' tools'}</span>
        ) : null}
        <span fg={theme.muted}>{' · '}</span>
      </text>
      <Button
        onClick={onViewAgents}
        onMouseOver={() => setIsLinkHovered(true)}
        onMouseOut={() => setIsLinkHovered(false)}
      >
        <text style={{ fg: isLinkHovered ? theme.foreground : theme.muted, wrapMode: 'none' }}>
          {'View agents →'}
        </text>
      </Button>
    </box>
  )
})