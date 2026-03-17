import { memo, useState, useEffect, useMemo } from 'react'
import type { AgentsViewArtifactItem } from '@magnitudedev/agent'
import { getLatestInProgressArtifactStream } from '@magnitudedev/agent'
import { Button } from '../button'
import { useTheme } from '../../hooks/use-theme'
import { getAgentColorByRole } from '../../utils/agent-colors'
import { LaneGutter, type LaneEntry } from './lane-gutter'

interface AgentArtifactEventProps {
  item: AgentsViewArtifactItem
  onArtifactClick?: (name: string, section?: string) => void
  lanes?: LaneEntry[]
  subscribeForkDisplay: (forkId: string | null, cb: (state: any) => void) => () => void
}

export const AgentArtifactEvent = memo(function AgentArtifactEvent({
  item,
  onArtifactClick,
  lanes = [],
  subscribeForkDisplay,
}: AgentArtifactEventProps) {
  const theme = useTheme()
  const [artifactHovered, setArtifactHovered] = useState(false)
  const [liveContent, setLiveContent] = useState<string | null>(null)
  const palette = getAgentColorByRole(item.agentRole)

  useEffect(() => {
    if (item.phase !== 'streaming') {
      setLiveContent(null)
      return
    }
    const unsubscribe = subscribeForkDisplay(item.forkId, (state) => {
      const stream = getLatestInProgressArtifactStream(state, item.artifactName)
      if (stream?.preview?.mode === 'write') {
        setLiveContent(stream.preview.contentSoFar)
      } else if (stream?.preview?.mode === 'update') {
        setLiveContent(stream.preview.newStringSoFar || stream.preview.oldStringSoFar)
      } else {
        setLiveContent(null)
      }
    })
    return unsubscribe
  }, [item.forkId, item.artifactName, item.phase, subscribeForkDisplay])

  const isStreaming = item.phase === 'streaming'
  const actionLabel = isStreaming
    ? (item.action === 'wrote' ? 'writing' : 'updating')
    : (item.action === 'wrote' ? 'created' : 'updated')

  const previewLines = useMemo(() => {
    if (!liveContent) return []
    const lines = liveContent.split('\n').filter(l => l.trim().length > 0)
    return lines.slice(-2)
  }, [liveContent])

  return (
    <box style={{ flexDirection: 'row', alignItems: 'stretch' }}>
      <LaneGutter lanes={lanes} />
      <box style={{ flexDirection: 'column', flexGrow: 1, minWidth: 0 }}>
        <box
          style={{
            flexDirection: 'row',
            alignItems: 'stretch',
            minHeight: 1,
          }}
        >
          <text style={{ wrapMode: 'none' }}>
            <span fg={palette.border}>{'✎ '}</span>
            <span fg={palette.border}>{item.agentName}</span>
            <span fg={theme.muted}>{' '}{actionLabel}{' '}</span>
          </text>
          <Button
            onClick={onArtifactClick ? () => onArtifactClick(item.artifactName) : undefined}
            onMouseOver={() => setArtifactHovered(true)}
            onMouseOut={() => setArtifactHovered(false)}
          >
            <text style={{ fg: artifactHovered ? theme.link : theme.primary, wrapMode: 'none' }}>{'[≡ '}{item.artifactName}{']'}</text>
          </Button>
          {isStreaming && <text style={{ fg: theme.muted, wrapMode: 'none' }}>{'...'}</text>}
        </box>
        {isStreaming && previewLines.map((line, i) => (
          <box key={i} style={{ flexDirection: 'row', minHeight: 1, overflow: 'hidden' }}>
            <text style={{ flexShrink: 0, width: 2, fg: theme.muted, wrapMode: 'none' }}>{'│ '}</text>
            <text style={{ flexShrink: 1, fg: theme.muted, wrapMode: 'none', overflow: 'hidden' }}>{line}</text>
          </box>
        ))}
      </box>
    </box>
  )
})