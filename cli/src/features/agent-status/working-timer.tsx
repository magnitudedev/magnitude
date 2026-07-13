import { memo, useSyncExternalStore } from 'react'
import { Option } from 'effect'
import type { DisplayActorWork, InterruptedMessage } from '@magnitudedev/sdk'
import { useTheme } from '../../hooks/use-theme'
import { slate, subscribeAnimationTick, getAnimationTickSnapshot } from '@magnitudedev/client-common'
import { red } from '../../utils/theme'

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

interface WorkingTimerProps {
  work: DisplayActorWork | null
  interruptedMessage?: InterruptedMessage | null
  advisorModelName?: string | null
}

function formatElapsed(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds} second${seconds === 1 ? '' : 's'}`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  if (remainingSeconds === 0) return `${minutes} minute${minutes === 1 ? '' : 's'}`
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
}

function buildSummaryLine(durationSeconds: number): string {
  return `Worked for ${formatDuration(durationSeconds)}`
}

export const WorkingTimer = memo(function WorkingTimer({
  work,
  interruptedMessage,
  advisorModelName,
}: WorkingTimerProps) {
  const theme = useTheme()
  const tick = useSyncExternalStore(subscribeAnimationTick, getAnimationTickSnapshot, getAnimationTickSnapshot)

  const active = work?.phase === 'working'
  const activity = work?.activity ?? null
  const hasSpinner = activity?.kind === 'tool' && Option.getOrNull(activity.decorator) === 'spinner'
  const hasActivity = activity !== null
  const isAdvisor = activity?.kind === 'advisor'

  // Derive animation indices from tick (80ms per tick)
  // Elapsed: current chain only — no accumulatedMs prefix
  const elapsedMs = active && work
    ? Math.max(0, Date.now() - (work.activeSince ?? Date.now()))
    : 0
  const elapsedSeconds = Math.floor(elapsedMs / 1000)
  // Thinking pulse: 250ms → ~3 ticks per step
  const thinkingPulseIndex = (hasActivity && (active || isAdvisor)) ? Math.floor(tick / 3) % THINKING_PULSE_COLORS.length : 0
  // Dot pulse: 300ms → ~4 ticks per step
  const dotPulseIndex = active ? Math.floor(tick / 4) % WORKING_PULSE_COLORS.length : 0
  // Braille: 80ms → 1 tick per step
  const brailleIndex = (hasSpinner && active) ? tick % BRAILLE_FRAMES.length : 0

  // Active: show running timer
  if (active) {
    return (
      <box style={{ flexShrink: 0, paddingLeft: 2, paddingTop: 0, paddingBottom: 0 }}>
        <text style={{ fg: theme.muted }}>
          <span style={{ fg: isAdvisor ? slate[600] : WORKING_PULSE_COLORS[dotPulseIndex] }}>{'●'}</span>
          {` Working... ${formatElapsed(elapsedSeconds)}`}
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
      <box style={{ flexShrink: 0, paddingLeft: 2, paddingTop: 0, paddingBottom: 0 }}>
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
      <box style={{ flexShrink: 0, paddingLeft: 2, paddingTop: 0, paddingBottom: 0 }}>
        <text style={{ fg: red[400] }}>{interruptText}</text>
      </box>
    )
  }

  // Completed: show persistent summary
  if (!active && work && work.lastWorkMs > 0) {
    const durationSeconds = Math.floor(work.lastWorkMs / 1000)
    return (
      <box style={{ flexShrink: 0, paddingLeft: 2, paddingTop: 0, paddingBottom: 0 }}>
        <text style={{ fg: theme.muted }}>
          <span style={{ fg: slate[600] }}>{'●'}</span>
          {' '}
          {buildSummaryLine(durationSeconds)}
        </text>
      </box>
    )
  }

  return null
})
