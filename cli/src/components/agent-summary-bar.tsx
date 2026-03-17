import { memo, useState, useEffect, useRef } from 'react'
import type { MutableRefObject } from 'react'
import { TextAttributes } from '@opentui/core'
import type { AgentsViewState, AgentsViewActivityStartItem, AgentsViewMessageItem, AgentsViewArtifactItem } from '@magnitudedev/agent'
import { getAgentColorByRole } from '../utils/agent-colors'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'

export interface SummaryBarSeenTracker {
  seenMessageIds: MutableRefObject<Set<string>>
  seenActivityStartIds: MutableRefObject<Set<string>>
  seenArtifactEventIds: MutableRefObject<Set<string>>
}

interface AgentSummaryBarProps {
  agentsViewState: AgentsViewState
  onViewAll: () => void
  onArtifactClick: (name: string) => void
  seenTracker: SummaryBarSeenTracker
  activeTab?: 'main' | 'agents'
  variant?: 'default' | 'main-content'
}

const PULSE_INTERVAL_MS = 200
const SWEEP_INTERVAL_MS = 200
const BLINK_DURATION_MS = 1500

function sweepColors(name: string, sweepPos: number, baseColor: string, pulse: string[]): string[] {
  if (name.length <= 1) return [pulse[0]!]
  const period = (name.length - 1) * 2
  const mod = sweepPos % period
  const cursor = mod < name.length ? mod : period - mod
  return Array.from(name, (_, i) => {
    const dist = Math.abs(i - cursor)
    if (dist <= 1) return pulse[0]!
    if (dist === 2) return pulse[1]!
    return baseColor
  })
}

function formatElapsed(startedAt: number, priorElapsedMs = 0): string {
  const seconds = Math.floor((Date.now() - startedAt + priorElapsedMs) / 1000)
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}

function messagePreview(content: string, max: number): string {
  const trimmed = content.trimEnd()
  const firstLine = trimmed.split('\n')[0]!
  const isComplete = firstLine === trimmed
  if (isComplete) {
    return firstLine.length > max ? firstLine.slice(0, max - 1) + '…' : firstLine
  } else {
    return firstLine.length > max ? firstLine.slice(0, max - 1) + '…' : firstLine + '…'
  }
}

/**
 * Runs a single flash sequence when `active` becomes true or `trigger` changes (while > 0).
 * Returns blinkOn: true = highlight phase, false = settled phase.
 */
function useBlinkSequence(active: boolean, trigger?: number): boolean {
  const [blinkOn, setBlinkOn] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const startBlink = () => {
    setBlinkOn(true)
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    timeoutRef.current = setTimeout(() => {
      setBlinkOn(false)
      timeoutRef.current = null
    }, BLINK_DURATION_MS)
  }

  useEffect(() => {
    if (active) {
      startBlink()
    } else {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
      setBlinkOn(false)
    }
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
    }
  }, [active])

  // Re-trigger when trigger counter increments (and active)
  useEffect(() => {
    if (trigger !== undefined && trigger > 0 && active) {
      startBlink()
    }
  }, [trigger, active])

  return blinkOn
}

const ArtifactChip = memo(function ArtifactChip({
  artifact,
  index,
  isBlinking,
  onArtifactClick,
}: {
  artifact: AgentsViewArtifactItem
  index: number
  isBlinking: boolean
  onArtifactClick: (name: string) => void
}) {
  const theme = useTheme()
  const [hovered, setHovered] = useState(false)
  const blinkOn = useBlinkSequence(isBlinking)

  const defaultColor = theme.primary
  const hoverColor = theme.link
  const color = hovered
    ? hoverColor
    : isBlinking
      ? (blinkOn ? theme.link : theme.muted)
      : defaultColor

  return (
    <Button
      onClick={() => onArtifactClick(artifact.artifactName)}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
    >
      <text style={{ wrapMode: 'none' }}>
        {index > 0 ? <span fg={theme.muted}>{' · '}</span> : null}
        <span fg={color} attributes={isBlinking && blinkOn ? TextAttributes.BOLD : TextAttributes.NONE}>{'[≡ '}{artifact.artifactName}{']'}</span>
      </text>
    </Button>
  )
})

const RunningAgentChip = memo(function RunningAgentChip({ item, isBlinking }: { item: AgentsViewActivityStartItem; isBlinking: boolean }) {
  const [pulseIndex, setPulseIndex] = useState(0)
  const [sweepPos, setSweepPos] = useState(0)
  const [elapsed, setElapsed] = useState(() => formatElapsed(item.startedAt, item.priorElapsedMs ?? 0))
  const pulseRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const sweepRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const palette = getAgentColorByRole(item.agentRole)
  const blinkOn = useBlinkSequence(isBlinking)

  useEffect(() => {
    pulseRef.current = setInterval(() => {
      setPulseIndex(prev => (prev + 1) % palette.pulse.length)
    }, PULSE_INTERVAL_MS)
    return () => { if (pulseRef.current) clearInterval(pulseRef.current) }
  }, [palette.pulse.length])

  useEffect(() => {
    sweepRef.current = setInterval(() => {
      setSweepPos(prev => prev + 1)
    }, SWEEP_INTERVAL_MS)
    return () => { if (sweepRef.current) clearInterval(sweepRef.current) }
  }, [])

  useEffect(() => {
    const update = () => setElapsed(formatElapsed(item.startedAt, item.priorElapsedMs ?? 0))
    update()
    timerRef.current = setInterval(update, 1000)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [item.startedAt, item.priorElapsedMs])

  const nameColors = isBlinking
    ? Array.from(item.agentName, () => blinkOn ? palette.pulse[0]! : palette.border)
    : sweepColors(item.agentName, sweepPos, palette.border, palette.pulse)

  return (
    <text style={{ wrapMode: 'none' }}>
      <span fg={isBlinking ? palette.pulse[0]! : palette.pulse[pulseIndex]!} attributes={isBlinking && blinkOn ? TextAttributes.BOLD : TextAttributes.NONE}>{'◆ '}</span>
      {Array.from(item.agentName).map((ch, i) => (
        <span key={i} fg={nameColors[i]} attributes={isBlinking && blinkOn ? TextAttributes.BOLD : TextAttributes.NONE}>{ch}</span>
      ))}
      <span fg={'#888888'}>{' '}{elapsed}{item.isResume ? ' (resumed)' : ''}{'  '}</span>
    </text>
  )
})

export const AgentSummaryBar = memo(function AgentSummaryBar({
  agentsViewState,
  onViewAll,
  onArtifactClick,
  seenTracker,
  activeTab = 'main',
  variant = 'default',
}: AgentSummaryBarProps) {
  const theme = useTheme()
  const [viewAllHovered, setViewAllHovered] = useState(false)
  const [blinkingAgents, setBlinkingAgents] = useState<Set<string>>(new Set())
  const [isMessageBlinking, setIsMessageBlinking] = useState(false)
  const [messageBlinkTrigger, setMessageBlinkTrigger] = useState(0)
  const [blinkingArtifacts, setBlinkingArtifacts] = useState<Set<string>>(new Set())
  const agentBlinkTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  const artifactBlinkTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  const messageBlinkTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const items = agentsViewState.items

  // Derive running agents from active item ids (one chip per active run)
  const startItemsById = new Map<string, AgentsViewActivityStartItem>()
  for (const item of items) {
    if (item.type === 'agents_view_activity_start') {
      startItemsById.set(item.id, item)
    }
  }
  const runningAgents = Array.from(agentsViewState.activeActivityIds.values())
    .map(entry => startItemsById.get(entry.itemId))
    .filter((item): item is AgentsViewActivityStartItem => item !== undefined)

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

  // Track message blink
  const lastMessageKey = lastMessage ? lastMessage.id : null
  useEffect(() => {
    if (lastMessageKey !== null && !seenTracker.seenMessageIds.current.has(lastMessageKey)) {
      seenTracker.seenMessageIds.current.add(lastMessageKey)
      if (messageBlinkTimerRef.current) clearTimeout(messageBlinkTimerRef.current)
      setIsMessageBlinking(true)
      setMessageBlinkTrigger(prev => prev + 1)
      messageBlinkTimerRef.current = setTimeout(() => {
        setIsMessageBlinking(false)
        messageBlinkTimerRef.current = null
      }, BLINK_DURATION_MS)
    }
  }, [lastMessageKey, seenTracker])

  // Track agent blink
  const runningAgentsKey = runningAgents.map(a => a.id).join('|')
  useEffect(() => {
    const newIds = new Set<string>()
    for (const agent of runningAgents) {
      if (!seenTracker.seenActivityStartIds.current.has(agent.id)) {
        seenTracker.seenActivityStartIds.current.add(agent.id)
        newIds.add(agent.id)
      }
    }

    if (newIds.size > 0) {
      setBlinkingAgents(prev => {
        const next = new Set(prev)
        for (const id of newIds) next.add(id)
        return next
      })
      for (const id of newIds) {
        const existing = agentBlinkTimersRef.current.get(id)
        if (existing) clearTimeout(existing)
        const t = setTimeout(() => {
          setBlinkingAgents(prev => { const next = new Set(prev); next.delete(id); return next })
          agentBlinkTimersRef.current.delete(id)
        }, BLINK_DURATION_MS)
        agentBlinkTimersRef.current.set(id, t)
      }
    }
  }, [runningAgentsKey, runningAgents, seenTracker])

  // Track artifact blink
  const artifactsIdentityKey = artifacts.map(a => a.id).join('|')
  useEffect(() => {
    const newArtifactNames = new Set<string>()
    for (const artifact of artifacts) {
      if (!seenTracker.seenArtifactEventIds.current.has(artifact.id)) {
        seenTracker.seenArtifactEventIds.current.add(artifact.id)
        newArtifactNames.add(artifact.artifactName)
      }
    }

    if (newArtifactNames.size > 0) {
      setBlinkingArtifacts(prev => {
        const next = new Set(prev)
        for (const n of newArtifactNames) next.add(n)
        return next
      })
      for (const name of newArtifactNames) {
        const existing = artifactBlinkTimersRef.current.get(name)
        if (existing) clearTimeout(existing)
        const t = setTimeout(() => {
          setBlinkingArtifacts(prev => { const next = new Set(prev); next.delete(name); return next })
          artifactBlinkTimersRef.current.delete(name)
        }, BLINK_DURATION_MS)
        artifactBlinkTimersRef.current.set(name, t)
      }
    }
  }, [artifactsIdentityKey, artifacts, seenTracker])

  const messageBlinkOn = useBlinkSequence(isMessageBlinking, messageBlinkTrigger)

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
                <RunningAgentChip key={agent.id} item={agent} isBlinking={blinkingAgents.has(agent.id)} />
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
              <span
                fg={isMessageBlinking
                  ? (messageBlinkOn
                    ? getAgentColorByRole(lastMessage.direction === 'from_agent' ? lastMessage.fromRole : 'orchestrator').pulse[0]
                    : getAgentColorByRole(lastMessage.direction === 'from_agent' ? lastMessage.fromRole : 'orchestrator').border)
                  : getAgentColorByRole(lastMessage.direction === 'from_agent' ? lastMessage.fromRole : 'orchestrator').border}
                attributes={isMessageBlinking && messageBlinkOn ? TextAttributes.BOLD : TextAttributes.NONE}

              >
                {'⌲ '}{lastMessage.direction === 'from_agent' ? `${lastMessage.fromName}` : 'Orchestrator'}{': '}
              </span>
              <span
                fg={isMessageBlinking ? (messageBlinkOn ? theme.foreground : theme.muted) : theme.muted}
                attributes={isMessageBlinking && messageBlinkOn ? TextAttributes.BOLD : TextAttributes.NONE}

              >
                {'"'}{messagePreview(lastMessage.content, 80)}{'"'}
              </span>
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
                  isBlinking={blinkingArtifacts.has(artifact.artifactName)}
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
              <RunningAgentChip key={agent.id} item={agent} isBlinking={blinkingAgents.has(agent.id)} />
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
            <span
              fg={isMessageBlinking
                ? (messageBlinkOn
                  ? getAgentColorByRole(lastMessage.direction === 'from_agent' ? lastMessage.fromRole : 'orchestrator').pulse[0]
                  : getAgentColorByRole(lastMessage.direction === 'from_agent' ? lastMessage.fromRole : 'orchestrator').border)
                : getAgentColorByRole(lastMessage.direction === 'from_agent' ? lastMessage.fromRole : 'orchestrator').border}
              attributes={isMessageBlinking && messageBlinkOn ? TextAttributes.BOLD : TextAttributes.NONE}

            >
              {'⌲ '}{lastMessage.direction === 'from_agent' ? `${lastMessage.fromName}` : 'Orchestrator'}{': '}
            </span>
            <span
              fg={isMessageBlinking ? (messageBlinkOn ? theme.foreground : theme.muted) : theme.muted}
              attributes={isMessageBlinking && messageBlinkOn ? TextAttributes.BOLD : TextAttributes.NONE}

            >
              {'"'}{messagePreview(lastMessage.content, 80)}{'"'}
            </span>
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
                isBlinking={blinkingArtifacts.has(artifact.artifactName)}
                onArtifactClick={onArtifactClick}
              />
            ))}
          </box>
        )}
      </box>
    </box>
  )
})