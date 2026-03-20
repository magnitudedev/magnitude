import { memo, useEffect, useRef, useState, useCallback } from 'react'
import { TextAttributes } from '@opentui/core'
import type { ThinkBlockMessage, ThinkBlockStep } from '@magnitudedev/agent'
import { Button } from './button'
import { AgentCommunicationCard } from './agent-communication-card'
import { useTheme } from '../hooks/use-theme'
import { orange } from '../utils/theme'
import { ShimmerText } from './shimmer-text'
import { MiniWave } from './mini-wave'
import { renderRegistry, clusterRenderRegistry } from '../visuals/registry'
import { selectLatestLiveActivityFromThinkSteps } from '../utils/live-activity'

const SHIMMER_INTERVAL_MS = 160

interface ThinkBlockProps {
  block: ThinkBlockMessage
  isCollapsed: boolean
  onToggle: () => void
  timerStartTime?: number | null
  hideHeader?: boolean
  onHeaderRef?: (ref: any) => void
  pendingApproval?: boolean
  onFileClick?: (path: string, section?: string) => void
  isInterrupted?: boolean
}

const formatElapsedTime = (seconds: number): string => {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}

const SubagentStartedRow = ({ step }: { step: Extract<ThinkBlockStep, { type: 'subagent_started' }> }) => {
  const theme = useTheme()
  return (
    <text>
      <span style={{ fg: orange[400] }}>▶ </span>
      <span style={{ fg: theme.muted }}>Subagent </span>
      <span style={{ fg: theme.muted }}>started</span>
      <span style={{ fg: theme.muted }}>: </span>
      <span style={{ fg: theme.foreground }}>{step.subagentId}</span>
      {step.resumed && <span style={{ fg: theme.muted }}> (resumed)</span>}
      <span style={{ fg: theme.muted }}> — {step.title}</span>
    </text>
  )
}

const SubagentFinishedRow = ({ step }: { step: Extract<ThinkBlockStep, { type: 'subagent_finished' }> }) => {
  const theme = useTheme()
  const duration = formatDuration(Math.floor(step.cumulativeTotalTimeMs / 1000))
  const tools = step.cumulativeTotalToolsUsed
  return (
    <text>
      <span style={{ fg: theme.success }}>✓ </span>
      <span style={{ fg: theme.muted }}>Subagent </span>
      <span style={{ fg: theme.muted }}>finished</span>
      <span style={{ fg: theme.muted }}>: </span>
      <span style={{ fg: theme.foreground }}>{step.subagentId}</span>
      {step.resumed && <span style={{ fg: theme.muted }}> (resumed)</span>}
      <span style={{ fg: theme.muted }}> · {step.resumed ? '↺ ' : ''}{duration} · </span>
      <span style={{ fg: theme.primary }}>{tools}</span>
      <span style={{ fg: theme.muted }}> {tools === 1 ? 'tool' : 'tools'}</span>
    </text>
  )
}

const SubagentKilledRow = ({ step }: { step: Extract<ThinkBlockStep, { type: 'subagent_killed' | 'subagent_user_killed' }> }) => {
  const theme = useTheme()
  const message = step.type === 'subagent_user_killed' ? 'Subagent killed by user: ' : 'Subagent killed: '
  return (
    <text>
      <span style={{ fg: theme.error }}>■ </span>
      <span style={{ fg: theme.muted }}>{message}</span>
      <span style={{ fg: theme.foreground }}>{step.subagentId}</span>
      <span style={{ fg: theme.muted }}> - {step.title}</span>
    </text>
  )
}

function buildSummary(steps: readonly { type: string; toolKey?: string }[]): string {
  let webSearches = 0
  let commands = 0
  let reads = 0
  let writes = 0
  let searches = 0
  let edits = 0
  let gathers = 0
  let clicks = 0
  let navigations = 0
  let inputs = 0
  let evaluations = 0
  let subagentStarted = 0
  let subagentFinished = 0
  let subagentKilled = 0
  for (const step of steps) {
    if (step.type === 'subagent_started') {
      subagentStarted++
      continue
    }
    if (step.type === 'subagent_finished') {
      subagentFinished++
      continue
    }
    if (step.type === 'subagent_killed') {
      subagentKilled++
      continue
    }
    if (step.type !== 'tool') continue
    if (step.toolKey === 'webSearch' || step.toolKey === 'webFetch') webSearches++
    else if (step.toolKey === 'shell') commands++
    else if (step.toolKey === 'fileRead') reads++
    else if (step.toolKey === 'fileWrite') writes++
    else if (step.toolKey === 'fileSearch') searches++
    else if (step.toolKey === 'fileEdit') edits++
    else if (step.toolKey === 'gather') gathers++
    else if (step.toolKey === 'click' || step.toolKey === 'doubleClick' || step.toolKey === 'rightClick' || step.toolKey === 'drag') clicks++
    else if (step.toolKey === 'navigate' || step.toolKey === 'goBack' || step.toolKey === 'switchTab' || step.toolKey === 'newTab') navigations++
    else if (step.toolKey === 'type') inputs++
    else if (step.toolKey === 'evaluate') evaluations++
  }
  const parts: string[] = []
  if (webSearches > 0) parts.push(`${webSearches} ${webSearches === 1 ? 'web search' : 'web searches'}`)
  if (commands > 0) parts.push(`${commands} ${commands === 1 ? 'command' : 'commands'}`)
  if (reads > 0) parts.push(`${reads} ${reads === 1 ? 'read' : 'reads'}`)
  if (writes > 0) parts.push(writes + ' ' + (writes === 1 ? 'write' : 'writes'))
  if (edits > 0) parts.push(edits + ' ' + (edits === 1 ? 'edit' : 'edits'))
  if (searches > 0) parts.push(`${searches} ${searches === 1 ? 'search' : 'searches'}`)
  if (gathers > 0) parts.push(`${gathers} gather`)
  if (clicks > 0) parts.push(`${clicks} ${clicks === 1 ? 'click' : 'clicks'}`)
  if (navigations > 0) parts.push(`${navigations} ${navigations === 1 ? 'navigation' : 'navigations'}`)
  if (inputs > 0) parts.push(`${inputs} ${inputs === 1 ? 'input' : 'inputs'}`)
  if (evaluations > 0) parts.push(`${evaluations} ${evaluations === 1 ? 'eval' : 'evals'}`)
  if (subagentStarted > 0) parts.push(`${subagentStarted} ${subagentStarted === 1 ? 'subagent started' : 'subagents started'}`)
  if (subagentFinished > 0) parts.push(`${subagentFinished} ${subagentFinished === 1 ? 'subagent finished' : 'subagents finished'}`)
  if (subagentKilled > 0) parts.push(`${subagentKilled} ${subagentKilled === 1 ? 'subagent killed' : 'subagents killed'}`)
  return parts.length > 0 ? ` (${parts.join(', ')})` : ''
}

// =============================================================================
// Cluster-based step grouping
// =============================================================================

interface StepGroup {
  /** Cluster key from the visual definition, or null for thinking/unregistered tools */
  cluster: string | null
  steps: ThinkBlockStep[]
}

/**
 * Group consecutive steps by visual cluster.
 * Consecutive tool steps with the same non-null cluster share a group.
 * Thinking steps and tools without a cluster get their own singleton group.
 */
function groupByCluster(steps: readonly ThinkBlockStep[]): StepGroup[] {
  const groups: StepGroup[] = []
  for (const step of steps) {
    const cluster = step.cluster ?? null
    const syntheticCluster = step.type === 'subagent_started' || step.type === 'subagent_finished' || step.type === 'subagent_killed' || step.type === 'subagent_user_killed'
      ? '__subagent_lifecycle__'
      : cluster
    const last = groups[groups.length - 1]
    if (last && syntheticCluster !== null && last.cluster === syntheticCluster) {
      last.steps.push(step)
    } else {
      groups.push({ cluster: syntheticCluster, steps: [step] })
    }
  }
  return groups
}

// =============================================================================
// Step rendering — delegates to visual registry
// =============================================================================

const ToolStepView = memo(function ToolStepView({
  step,
  isExpanded,
  onToggle,
  onFileClick,
}: {
  step: ThinkBlockStep
  isExpanded: boolean
  onToggle: () => void
  onFileClick?: (name: string, section?: string) => void
}) {
  const theme = useTheme()
  const render = step.toolKey ? renderRegistry.get(step.toolKey) : undefined

  if (render && step.visualState !== undefined) {
    return <>{render({ state: step.visualState, isExpanded, onToggle, stepResult: step.result, onFileClick })}</>
  }

  // Fallback for tools without a visual definition
  return (
    <text>
      <span style={{ fg: theme.warning }}>{step.label}</span>
      {step.result && (
        <span style={{ fg: step.result.status === 'success' ? theme.success : theme.error }}>
          {' '}[{step.result.status}]
        </span>
      )}
    </text>
  )
})

const THINKING_FADE_WINDOW = 15
const THINKING_TICK_MS = 33
const THINKING_LINEAR_DRAIN = 8

const ThinkingStep = memo(function ThinkingStep({ content, label, isActive, isInterrupted }: { content: string; label?: string; isActive: boolean; isInterrupted?: boolean }) {
  const theme = useTheme()
  const [displayedLength, setDisplayedLength] = useState(content.length)
  const isLinearDrainRef = useRef(!isActive)

  useEffect(() => {
    if (isInterrupted || !isActive) setDisplayedLength(content.length)
  }, [isInterrupted, isActive, content.length])

  useEffect(() => {
    isLinearDrainRef.current = !isActive
  }, [isActive])

  useEffect(() => {
    if (!isActive && displayedLength >= content.length) return

    const interval = setInterval(() => {
      setDisplayedLength((prev) => {
        const target = content.length
        if (prev >= target) return prev
        if (isLinearDrainRef.current) {
          return Math.min(target, prev + THINKING_LINEAR_DRAIN)
        }
        const remaining = target - prev
        const speed = Math.max(1, Math.floor(remaining * 0.15))
        return Math.min(target, prev + speed)
      })
    }, THINKING_TICK_MS)

    return () => clearInterval(interval)
  }, [content.length, displayedLength, isActive])

  const displayed = content.slice(0, displayedLength)
  const isAnimating = displayedLength < content.length

  if (!isAnimating) {
    return (
      <text attributes={TextAttributes.ITALIC}>

        <span style={{ fg: theme.muted }}>{displayed}</span>
      </text>
    )
  }

  const fadeWindowStart = Math.max(0, displayedLength - THINKING_FADE_WINDOW)
  const settled = displayed.slice(0, fadeWindowStart)
  const fading = displayed.slice(fadeWindowStart)

  return (
    <text attributes={TextAttributes.ITALIC}>
      <span style={{ fg: theme.muted }}>{settled}</span>
      <span style={{ fg: theme.border }} attributes={TextAttributes.DIM}>
        {fading}
      </span>
    </text>
  )
})

// =============================================================================
// Cluster container styling
// =============================================================================

function ClusterContainer({
  cluster,
  isFirstGroup,
  children,
}: {
  cluster: string | null
  isFirstGroup: boolean
  children: React.ReactNode
}) {
  const theme = useTheme()

  return (
    <box
      style={{
        flexDirection: 'column',
        marginTop: isFirstGroup ? 0 : 1,
        ...(cluster === 'shell'
          ? {
              backgroundColor: theme.terminalBg,
              paddingRight: 1,
            }
          : {}),
      }}
    >
      {children}
    </box>
  )
}

// =============================================================================
// Group rendering — cluster renderer or per-step fallback
// =============================================================================

const StepGroupView = memo(function StepGroupView({
  group,
  expandedSteps,
  toggleStep,
  onFileClick,
  isActive,
  isInterrupted,
  lastThinkingStepId,
}: {
  group: StepGroup
  expandedSteps: Set<string>
  toggleStep: (id: string) => void
  onFileClick?: (path: string, section?: string) => void
  isActive?: boolean
  isInterrupted?: boolean
  lastThinkingStepId?: string
}) {
  const theme = useTheme()
  const clusterRenderer = group.cluster
    ? clusterRenderRegistry.get(group.cluster)
    : undefined

  if (clusterRenderer) {
    const clusterSteps = group.steps
      .filter(s => s.type === 'tool' && s.visualState !== undefined)
      .map(s => ({ id: s.id, visualState: s.visualState!, result: s.result }))

    return (
      <>
        {clusterRenderer({
          steps: clusterSteps,
          expandedSteps,
          onToggleStep: toggleStep,
          onFileClick: onFileClick,
        })}
      </>
    )
  }

  return (
    <>
      {group.steps.map((step) => {
        if (step.type === 'thinking') {
          const isLastThinkingStep = step.id === lastThinkingStepId
          return (
            <box key={step.id}>
              <ThinkingStep content={step.content ?? ''} label={step.label} isActive={(isActive ?? false) && isLastThinkingStep} isInterrupted={isInterrupted} />
            </box>
          )
        }

        if (step.type === 'subagent_started') {
          return <SubagentStartedRow key={step.id} step={step} />
        }

        if (step.type === 'subagent_finished') {
          return <SubagentFinishedRow key={step.id} step={step} />
        }

        if (step.type === 'subagent_killed' || step.type === 'subagent_user_killed') {
          return <SubagentKilledRow key={step.id} step={step} />
        }

        if (step.type === 'communication') {
          const message = {
            id: step.id,
            type: 'agent_communication',
            direction: step.direction,
            agentId: step.agentId,
            agentName: step.agentName,
            agentRole: step.agentRole,
            forkId: step.forkId ?? null,
            content: step.content,
            preview: step.preview || step.content,
            timestamp: step.timestamp,
          }

          return <AgentCommunicationCard key={step.id} message={message} widthAdjustment={2} />
        }

        return (
          <ToolStepView
            key={step.id}
            step={step}
            isExpanded={expandedSteps.has(step.id)}
            onToggle={() => toggleStep(step.id)}
            onFileClick={onFileClick}
          />
        )
      })}
    </>
  )
})

// =============================================================================
// Sticky Working Header
// =============================================================================

interface StickyWorkingHeaderProps {
  timerStartTime: number | null
  onToggle: () => void
  pendingApproval?: boolean
}

export const StickyWorkingHeader = memo(function StickyWorkingHeader({
  timerStartTime,
  onToggle,
  pendingApproval,
}: StickyWorkingHeaderProps) {
  const theme = useTheme()
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [isHovered, setIsHovered] = useState(false)

  useEffect(() => {
    if (!timerStartTime) {
      setElapsedSeconds(0)
      return
    }

    const updateElapsed = () => {
      const now = Date.now()
      const elapsed = Math.floor((now - timerStartTime) / 1000)
      setElapsedSeconds(elapsed)
    }

    updateElapsed()

    // Freeze timer during pending approval
    if (pendingApproval) return

    const interval = setInterval(updateElapsed, 1000)

    return () => clearInterval(interval)
  }, [timerStartTime, pendingApproval])

  const textColor = isHovered ? theme.primary : theme.muted
  const shimmerColor = isHovered ? theme.primary : theme.secondary

  return (
    <Button
      onClick={onToggle}
      style={{ flexDirection: 'row' }}
      onMouseOver={() => setIsHovered(true)}
      onMouseOut={() => setIsHovered(false)}
    >
      <text style={{ fg: textColor }}>
        <span>{'\u25be '}</span>
        {pendingApproval ? (
          <span>Waiting for approval...</span>
        ) : (
          <>
            <ShimmerText
              text="Working "
              interval={SHIMMER_INTERVAL_MS}
              primaryColor={shimmerColor}
            />
            <MiniWave color={shimmerColor} />
          </>
        )}
        {elapsedSeconds > 0 && (
          <span style={{ fg: theme.secondary }}> {formatElapsedTime(elapsedSeconds)}</span>
        )}
      </text>
    </Button>
  )
})

// =============================================================================
// ThinkBlock
// =============================================================================

export const ThinkBlock = memo(function ThinkBlock({
  block,
  isCollapsed,
  onToggle,
  timerStartTime,
  hideHeader,
  onHeaderRef,
  pendingApproval,
  onFileClick,
  isInterrupted
}: ThinkBlockProps) {
  const theme = useTheme()
  const isActive = block.status === 'active'
  const isEmpty = block.steps.length === 0
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(() => new Set())
  const [isHeaderHovered, setIsHeaderHovered] = useState(false)

  const toggleStep = useCallback((id: string) => {
    setExpandedSteps(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const preview = selectLatestLiveActivityFromThinkSteps(block.steps) ?? 'Thinking...'

  // Summary for completed blocks
  const summary = buildSummary(block.steps)
  const completedDuration = block.completedAt ? Math.floor((block.completedAt - block.timestamp) / 1000) : 0

  // Group steps by visual cluster
  const groups = groupByCluster(block.steps)
  const lastStepId = block.steps.length > 0 ? block.steps[block.steps.length - 1].id : undefined

  // Timer logic
  useEffect(() => {
    if (!timerStartTime || !isActive) {
      setElapsedSeconds(0)
      return
    }

    const updateElapsed = () => {
      const now = Date.now()
      const elapsed = Math.floor((now - timerStartTime) / 1000)
      setElapsedSeconds(elapsed)
    }

    updateElapsed()

    // Freeze timer during pending approval
    if (pendingApproval) return

    const interval = setInterval(updateElapsed, 1000)

    return () => clearInterval(interval)
  }, [timerStartTime, isActive, pendingApproval])

  return (
    <box ref={onHeaderRef} style={{ marginBottom: 1, flexDirection: 'column' }}>
      {/* Header - clickable to toggle (hidden when rendered externally as sticky) */}
      {!hideHeader && (
        <Button
          onClick={() => { setExpandedSteps(new Set()); onToggle() }}
          style={{ flexDirection: 'row', alignSelf: 'flex-start' }}
          onMouseOver={() => setIsHeaderHovered(true)}
          onMouseOut={() => setIsHeaderHovered(false)}
        >
          <text style={{ fg: isHeaderHovered ? theme.primary : theme.muted }}>
            <span>{isCollapsed ? '\u25b8' : '\u25be'} </span>
            {isActive ? (
              <>
                {pendingApproval ? (
                  <span>Waiting for approval...</span>
                ) : (
                  <>
                    <ShimmerText
                      text="Working "
                      interval={SHIMMER_INTERVAL_MS}
                      primaryColor={isHeaderHovered ? theme.primary : theme.secondary}
                    />
                    <MiniWave color={isHeaderHovered ? theme.primary : theme.secondary} />
                  </>
                )}
                {elapsedSeconds > 0 && (
                  <span style={{ fg: theme.secondary }}> {formatElapsedTime(elapsedSeconds)}</span>
                )}
              </>
            ) : (
              <>
                <span attributes={TextAttributes.BOLD}>
                  Completed{completedDuration > 0 ? ` in ${formatDuration(completedDuration)}` : ''}{summary}
                </span>
                <span style={{ fg: isHeaderHovered ? theme.primary : theme.muted }}> · {isCollapsed ? 'Show' : 'Hide'}</span>
              </>
            )}
            {isCollapsed && isActive && !isEmpty && (
              <span style={{ fg: theme.secondary }}> — {preview}</span>
            )}
          </text>
        </Button>
      )}

      {/* Expanded content — cluster-grouped, registry-driven rendering */}
      {!isCollapsed && (
        <box style={{ paddingLeft: 2, flexDirection: 'column' }}>
          {groups.map((group, gi) => (
            <ClusterContainer key={gi} cluster={group.cluster} isFirstGroup={gi === 0}>
              <StepGroupView
                group={group}
                expandedSteps={expandedSteps}
                toggleStep={toggleStep}
                onFileClick={onFileClick}
                isActive={isActive}
                isInterrupted={isInterrupted}
                lastThinkingStepId={lastStepId}
              />
            </ClusterContainer>
          ))}
        </box>
      )}
    </box>
  )
})
