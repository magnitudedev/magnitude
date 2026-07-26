import { memo, useSyncExternalStore } from 'react'
import { Option } from 'effect'
import type {
  DisplayActorWork,
  DisplayModelRequestActivity,
  InterruptedMessage,
} from '@magnitudedev/sdk'
import { useTheme } from '../../hooks/use-theme'
import {
  slate,
  subscribeAnimationTick,
  getAnimationTickSnapshot,
  type LocalModelLoadActivity,
} from '@magnitudedev/client-common'
import { red } from '../../utils/theme'
import { modelRequestProgressSegments } from '../chat-timeline/model-request-progress'

const WORKING_PULSE_COLORS = [
  slate[100], slate[200], slate[300], slate[400], slate[500],
  slate[400], slate[300], slate[200],
] as const

// Smooth pulse: 400 → 300 → 400 with computed intermediates
// slate[400]=#94a3b8  slate[300]=#cbd5e1
const THINKING_PULSE_COLORS = [
  slate[400],      // 0%   #94a3b8
  '#a2b0c3',       // 25%
  '#b0bccd',       // 50%
  '#bdc9d7',       // 75%
  slate[300],      // 100% #cbd5e1 (peak)
  '#bdc9d7',       // 75%
  '#b0bccd',       // 50%
  '#a2b0c3',       // 25%
] as const

const BRAILLE_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'] as const

interface ActivityRailProps {
  work: DisplayActorWork | null
  width: number
  waitsForGenerationProgress: boolean
  modelLoadActivity: LocalModelLoadActivity | null
  modelRequestActivity: DisplayModelRequestActivity | null
  interruptedMessage?: InterruptedMessage | null
  advisorModelName?: string | null
}

const REQUEST_VISIBILITY_DELAY_MS = 500
const LOW_MEMORY_MODEL_STOPPED_MESSAGE =
  "Model stopped · Low memory - close memory-intensive apps and try again"

function formatElapsed(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

export const ActivityRail = memo(function ActivityRail({
  work,
  width,
  waitsForGenerationProgress,
  modelLoadActivity,
  modelRequestActivity,
  interruptedMessage,
  advisorModelName,
}: ActivityRailProps) {
  const theme = useTheme()
  const tick = useSyncExternalStore(subscribeAnimationTick, getAnimationTickSnapshot, getAnimationTickSnapshot)

  const active = work?.phase === 'working'
  const activity = work?.activity ?? null
  const hasSpinner = activity?.kind === 'tool' && Option.getOrNull(activity.decorator) === 'spinner'
  const hasActivity = activity !== null
  const isAdvisor = activity?.kind === 'advisor'

  // Derive animation indices from tick (80ms per tick)
  // Thinking pulse: 250ms → ~3 ticks per step
  const thinkingPulseIndex = (hasActivity && (active || isAdvisor)) ? Math.floor(tick / 3) % THINKING_PULSE_COLORS.length : 0
  // Dot pulse: 300ms → ~4 ticks per step
  const dotPulseIndex = active ? Math.floor(tick / 4) % WORKING_PULSE_COLORS.length : 0
  // Braille: 80ms → 1 tick per step
  const brailleIndex = (hasSpinner && active) ? tick % BRAILLE_FRAMES.length : 0
  const loadingBrailleIndex = tick % BRAILLE_FRAMES.length

  if (modelLoadActivity !== null) {
    if (modelLoadActivity._tag === "Blocked") {
      return (
        <box style={{ height: 1, flexShrink: 0 }}>
          <text style={{ fg: theme.warning }}>{LOW_MEMORY_MODEL_STOPPED_MESSAGE}</text>
        </box>
      )
    }
    return (
      <box style={{ height: 1, flexShrink: 0 }}>
        <text>
          <span style={{ fg: theme.primary }}>{BRAILLE_FRAMES[loadingBrailleIndex]}</span>
          {' '}
          <span style={{ fg: theme.foreground }}>Loading model</span>
          <span style={{ fg: theme.muted }}>{` · ${modelLoadActivity.percentage}%`}</span>
        </text>
      </box>
    )
  }

  if (modelRequestActivity !== null) {
    if (Date.now() - modelRequestActivity.startedAt < REQUEST_VISIBILITY_DELAY_MS) {
      return <box style={{ height: 1, flexShrink: 0 }} />
    }
    const progress = modelRequestProgressSegments(modelRequestActivity)
    const compactLabel = width < 60
      && progress.label === 'Loading conversation into the model'
      ? 'Loading conversation'
      : progress.label
    return (
      <box style={{ height: 1, flexShrink: 0 }}>
        <text>
          <span style={{ fg: theme.primary }}>{BRAILLE_FRAMES[loadingBrailleIndex]}</span>
          {' '}
          <span style={{ fg: theme.foreground }}>{compactLabel}</span>
          {progress.detail && (
            <span style={{ fg: theme.muted }}>{` · ${progress.detail}`}</span>
          )}
          {progress.trailing && width >= 72 && (
            <span style={{ fg: theme.muted }}>{` · ${progress.trailing}`}</span>
          )}
        </text>
      </box>
    )
  }

  if (
    active
    && waitsForGenerationProgress
    && work.respondingSince === undefined
  ) {
    return <box style={{ height: 1, flexShrink: 0 }} />
  }

  // Active: show running timer
  if (active) {
    const respondingSince = work.respondingSince ?? work.activeSince
    const responseElapsedMs = Math.max(0, Date.now() - (respondingSince ?? Date.now()))
    const responseElapsedSeconds = Math.floor(responseElapsedMs / 1000)
    return (
      <box style={{ height: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.muted }}>
          <span style={{ fg: isAdvisor ? slate[600] : WORKING_PULSE_COLORS[dotPulseIndex] }}>{'●'}</span>
          {` Working... ${formatElapsed(responseElapsedSeconds)}`}
          {work.activeChildCount > 0 && (
            <>
              {' · '}
              {`${work.activeChildCount} worker${work.activeChildCount === 1 ? '' : 's'} running`}
            </>
          )}
          {hasSpinner && (
            <>
              {' · '}
              <span style={{ fg: theme.muted }}>{BRAILLE_FRAMES[brailleIndex]}</span>
              {' '}
              {activity!.kind === 'tool' && activity.message}
            </>
          )}
          {isAdvisor && (
            <>
              {' · '}
              <span style={{ fg: THINKING_PULSE_COLORS[thinkingPulseIndex] }}>
                {activity!.message}{advisorModelName ? ` (${advisorModelName})` : ''}
              </span>
            </>
          )}
          {hasActivity && !hasSpinner && !isAdvisor && (
            <>
              {' · '}
              <span style={{ fg: THINKING_PULSE_COLORS[thinkingPulseIndex] }}>
                {activity!.message}
              </span>
            </>
          )}
        </text>
      </box>
    )
  }

  // Chain inactive but activity present — show activity standalone (takes priority over completed/interrupted)
  if (!active && hasActivity) {
    return (
      <box style={{ height: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.muted }}>
          <span style={{ fg: isAdvisor ? theme.muted : THINKING_PULSE_COLORS[thinkingPulseIndex] }}>{'●'}</span>
          {' '}
          <span style={{ fg: THINKING_PULSE_COLORS[thinkingPulseIndex] }}>
            {activity!.message}{isAdvisor && advisorModelName ? ` (${advisorModelName})` : ''}
          </span>
        </text>
      </box>
    )
  }

  // Interrupted state: show interrupt text in place of the work summary
  if (interruptedMessage) {
    let interruptText: string
    if (interruptedMessage.context === 'fork') {
      interruptText = '■ Agent stopped'
    } else if (interruptedMessage.allKilled) {
      interruptText = '■ All agents interrupted. What would you like to do?'
    } else {
      interruptText = '■ Lead interrupted. What would you like to do?'
    }
    return (
      <box style={{ height: 1, flexShrink: 0 }}>
        <text style={{ fg: red[400] }}>{interruptText}</text>
      </box>
    )
  }

  return <box style={{ height: 1, flexShrink: 0 }} />
})
