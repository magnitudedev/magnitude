import { memo, useState, useEffect, useRef } from 'react'
import { TextAttributes } from '@opentui/core'
import type { AgentsViewState, AgentsViewActivityStartItem, AgentsViewMessageItem, AgentsViewArtifactItem } from '@magnitudedev/agent'
import { getAgentColorByRole } from '../utils/agent-colors'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'

interface AgentSummaryBarProps {
  agentsViewState: AgentsViewState
  onViewAll: () => void
  onArtifactClick: (name: string) => void
  activeTab?: 'main' | 'agents'
  variant?: 'default' | 'main-content'
}

const PULSE_INTERVAL_MS = 200

function formatElapsed(startedAt: number): string {
  const seconds = Math.floor((Date.now() - startedAt) / 1000)
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}


function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + '…' : s
}

const ArtifactChip = memo(function ArtifactChip({
  artifact,
  index,
  isRecent,
  onArtifactClick,
}: {
  artifact: AgentsViewArtifactItem
  index: number
  isRecent: boolean
  onArtifactClick: (name: string) => void
}) {
  const theme = useTheme()
  const [hovered, setHovered] = useState(false)

  const defaultColor = theme.primary
  const hoverColor = theme.link
  const recentColor = getAgentColorByRole(artifact.agentRole).border
  const color = hovered ? hoverColor : isRecent ? recentColor : defaultColor

  return (
    <Button
      onClick={() => onArtifactClick(artifact.artifactName)}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
    >
      <text style={{ wrapMode: 'none' }}>
        {index > 0 ? <span fg={theme.muted}>{' · '}</span> : null}
        <span fg={color}>{'[≡ '}{artifact.artifactName}{']'}</span>
      </text>
    </Button>
  )
})

const RunningAgentChip = memo(function RunningAgentChip({ item }: { item: AgentsViewActivityStartItem }) {
  const [pulseIndex, setPulseIndex] = useState(0)
  const [elapsed, setElapsed] = useState(() => formatElapsed(item.startedAt))
  const pulseRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const palette = getAgentColorByRole(item.agentRole)

  useEffect(() => {
    pulseRef.current = setInterval(() => {
      setPulseIndex(prev => (prev + 1) % palette.pulse.length)
    }, PULSE_INTERVAL_MS)
    return () => { if (pulseRef.current) clearInterval(pulseRef.current) }
  }, [palette.pulse.length])

  useEffect(() => {
    const update = () => setElapsed(formatElapsed(item.startedAt))
    update()
    timerRef.current = setInterval(update, 1000)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [item.startedAt])

  return (
    <text style={{ wrapMode: 'none' }}>
      <span fg={palette.pulse[pulseIndex]!}>{'◆ '}</span>
      <span fg={palette.border}>{item.agentName}</span>
      <span fg={'#888888'}>{' '}{elapsed}{'  '}</span>
    </text>
  )
})

export const AgentSummaryBar = memo(function AgentSummaryBar({
  agentsViewState,
  onViewAll,
  onArtifactClick,
  activeTab = 'main',
  variant = 'default',
}: AgentSummaryBarProps) {
  const theme = useTheme()
  const [viewAllHovered, setViewAllHovered] = useState(false)
  const [recentArtifacts, setRecentArtifacts] = useState<Set<string>>(new Set())

  const items = agentsViewState.items

  // Derive running agents (start items without a corresponding end item)
  const runningForkIds = new Set<string>()
  for (const item of items) {
    if (item.type === 'agents_view_activity_start') runningForkIds.add(item.forkId)
    if (item.type === 'agents_view_activity_end') runningForkIds.delete(item.forkId)
  }
  const runningAgents = items.filter(
    (item): item is AgentsViewActivityStartItem =>
      item.type === 'agents_view_activity_start' && runningForkIds.has(item.forkId)
  )

  // Derive last message
  const lastMessage = [...items].reverse().find(
    (item): item is AgentsViewMessageItem => item.type === 'agents_view_message'
  ) ?? null

  // Derive artifacts (deduplicated by name, keep latest)
  const artifactMap = new Map<string, AgentsViewArtifactItem>()
  for (const item of items) {
    if (item.type === 'agents_view_artifact') {
      artifactMap.set(item.artifactName, item)
    }
  }
  const artifacts = Array.from(artifactMap.values()).slice(-3)

  // Track recently updated artifacts for highlight
  const prevArtifactNamesRef = useRef<Set<string>>(new Set(artifacts.map(a => a.artifactName)))
  useEffect(() => {
    const currentNames = new Set(artifacts.map(a => a.artifactName))
    const newNames = new Set<string>()
    for (const name of currentNames) {
      if (!prevArtifactNamesRef.current.has(name)) {
        newNames.add(name)
      }
    }
    prevArtifactNamesRef.current = currentNames
    if (newNames.size > 0) {
      setRecentArtifacts(prev => {
        const next = new Set(prev)
        for (const n of newNames) next.add(n)
        return next
      })
      const timeout = setTimeout(() => {
        setRecentArtifacts(prev => {
          const next = new Set(prev)
          for (const n of newNames) next.delete(n)
          return next
        })
      }, 2000)
      return () => clearTimeout(timeout)
    }
  }, [artifacts.length])

  const LABEL_WIDTH = 18
  const pl = variant === 'main-content' ? 1 : 2

  if (variant === 'main-content') {
    return (
      <>
        {/* Line 2: Running subagents */}
        <box style={{ flexDirection: 'row', paddingLeft: pl, paddingRight: 2 }}>
          <text style={{ wrapMode: 'none', width: LABEL_WIDTH }}>
            <span fg={theme.muted}>{'Subagents:  '}</span>
          </text>
          {runningAgents.length === 0 ? (
            <text><span fg={theme.muted}>{'None currently running'}</span></text>
          ) : (
            <box style={{ flexDirection: 'row', overflow: 'hidden' }}>
              {runningAgents.map(agent => (
                <RunningAgentChip key={agent.forkId} item={agent} />
              ))}
            </box>
          )}
        </box>

        {/* Line 3: Last message */}
        <box style={{ flexDirection: 'row', paddingLeft: pl, paddingRight: 2 }}>
          <text style={{ wrapMode: 'none', width: LABEL_WIDTH }}>
            <span fg={theme.muted}>{'Last Message: '}</span>
          </text>
          {lastMessage === null ? (
            <text><span fg={theme.muted}>{'—'}</span></text>
          ) : (
            <text style={{ wrapMode: 'none' }}>
              <span fg={getAgentColorByRole(lastMessage.direction === 'from_agent' ? lastMessage.fromRole : 'orchestrator').border}>
                {'⌲ '}{lastMessage.direction === 'from_agent' ? `${lastMessage.fromName}` : 'Orchestrator'}{': '}
              </span>
              <span fg={theme.muted}>{'"'}{truncate(lastMessage.content.split('\n')[0]!, 80)}{'"'}</span>
            </text>
          )}
        </box>

        {/* Line 4: Artifacts */}
        <box style={{ flexDirection: 'row', paddingLeft: pl, paddingRight: 2 }}>
          <text style={{ wrapMode: 'none', width: LABEL_WIDTH }}>
            <span fg={theme.muted}>{'Recent Artifacts: '}</span>
          </text>
          {artifacts.length === 0 ? (
            <text><span fg={theme.muted}>{'—'}</span></text>
          ) : (
            <box style={{ flexDirection: 'row', overflow: 'hidden' }}>
              {artifacts.map((artifact, i) => (
                <ArtifactChip
                  key={artifact.artifactName}
                  artifact={artifact}
                  index={i}
                  isRecent={recentArtifacts.has(artifact.artifactName)}
                  onArtifactClick={onArtifactClick}
                />
              ))}
            </box>
          )}
        </box>
      </>
    )
  }

  return (
    <box
      style={{
        flexShrink: 0,
        flexDirection: 'column',
        borderStyle: 'single',
        border: ['top'],
        borderColor: theme.border,
        customBorderChars: {
          topLeft: '─', bottomLeft: '', topRight: '─', bottomRight: '',
          horizontal: '─', vertical: ' ', topT: '─', bottomT: '',
          leftT: '', rightT: '', cross: '',
        },
        marginBottom: 1,
      }}
    >
      {/* Line 1: View all activity (only when not on agents tab) */}
      {activeTab !== 'agents' && (
        <box style={{ flexDirection: 'row', justifyContent: 'flex-end', paddingRight: 2 }}>
          <Button
            onClick={onViewAll}
            onMouseOver={() => setViewAllHovered(true)}
            onMouseOut={() => setViewAllHovered(false)}
          >
            <text style={{ wrapMode: 'none' }}>
              <span fg={viewAllHovered ? theme.primary : theme.muted}>{'View all activity →'}</span>
            </text>
          </Button>
        </box>
      )}

      {/* Line 2: Running subagents */}
      <box style={{ flexDirection: 'row', paddingLeft: 2, paddingRight: 2 }}>
        <text style={{ wrapMode: 'none', width: LABEL_WIDTH }}>
          <span fg={theme.muted}>{'Subagents:  '}</span>
        </text>
        {runningAgents.length === 0 ? (
          <text><span fg={theme.muted}>{'None currently running'}</span></text>
        ) : (
          <box style={{ flexDirection: 'row', overflow: 'hidden' }}>
            {runningAgents.map(agent => (
              <RunningAgentChip key={agent.forkId} item={agent} />
            ))}
          </box>
        )}
      </box>

      {/* Line 3: Last message */}
      <box style={{ flexDirection: 'row', paddingLeft: 2, paddingRight: 2 }}>
        <text style={{ wrapMode: 'none', width: LABEL_WIDTH }}>
          <span fg={theme.muted}>{'Last Message: '}</span>
        </text>
        {lastMessage === null ? (
          <text><span fg={theme.muted}>{'—'}</span></text>
        ) : (
          <text style={{ wrapMode: 'none' }}>
            <span fg={getAgentColorByRole(lastMessage.direction === 'from_agent' ? lastMessage.fromRole : 'orchestrator').border}>
              {'⌲ '}{lastMessage.direction === 'from_agent' ? `${lastMessage.fromName}` : 'Orchestrator'}{': '}
            </span>
            <span fg={theme.muted}>{'"'}{truncate(lastMessage.content.split('\n')[0]!, 80)}{'"'}</span>
          </text>
        )}
      </box>

      {/* Line 4: Artifacts */}
      <box style={{ flexDirection: 'row', paddingLeft: 2, paddingRight: 2 }}>
        <text style={{ wrapMode: 'none', width: LABEL_WIDTH }}>
          <span fg={theme.muted}>{'Recent Artifacts: '}</span>
        </text>
        {artifacts.length === 0 ? (
          <text><span fg={theme.muted}>{'—'}</span></text>
        ) : (
          <box style={{ flexDirection: 'row', overflow: 'hidden' }}>
            {artifacts.map((artifact, i) => (
              <ArtifactChip
                key={artifact.artifactName}
                artifact={artifact}
                index={i}
                isRecent={recentArtifacts.has(artifact.artifactName)}
                onArtifactClick={onArtifactClick}
              />
            ))}
          </box>
        )}
      </box>
    </box>
  )
})