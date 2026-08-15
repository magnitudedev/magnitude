import { memo } from 'react'
import { Option } from 'effect'
import { TextAttributes } from '@opentui/core'
import type { DisplayRootStatus, InterruptedMessage, SlotId } from '@magnitudedev/sdk'
import {
  displayRootStatusElapsedMs,
  animationPulse,
  modelReleaseReasonLabel,
  rootDetailSegments,
  useStabilizedRootDetail,
  type LocalModelLoadActivity,
} from '@magnitudedev/client-common'
import { useTheme } from '../../hooks/use-theme'
import { spinnerFrameForStep } from '../../hooks/use-spinner-frame'
import { useAnimationStep, useAnimationTime } from '../../hooks/use-animation-time'
import { Button } from '../../components/button'

const ACTIVE_PULSE_DURATION_MS = 1_200

const LOW_MEMORY_MODEL_STOPPED_MESSAGE =
  'Model stopped · Low memory - close memory-intensive apps and try again'

interface ActivityRailProps {
  status: DisplayRootStatus | null
  width: number
  modelLoadActivity: LocalModelLoadActivity | null
  onStopModel: (slotId: SlotId) => void
  interruptedMessage?: InterruptedMessage | null
}

function formatElapsed(totalMs: number): string {
  const totalSeconds = Math.floor(totalMs / 1_000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

export const ActivityRail = memo(function ActivityRail({
  status,
  width,
  modelLoadActivity,
  onStopModel,
  interruptedMessage,
}: ActivityRailProps) {
  const theme = useTheme()
  const modelResidency = modelLoadActivity === null
    ? null
    : modelLoadActivity.residency
  const active = status?._tag === 'Working'
  const stabilizedDetail = useStabilizedRootDetail(status)
  const pulseAnimated = modelResidency?._tag === 'Stopping'
    || (active && modelResidency?._tag !== 'Loading')
  const animationTime = useAnimationTime(pulseAnimated)
  const loadingSpinnerStep = useAnimationStep(modelResidency?._tag === 'Loading', 80)
  const pulseProgress = animationPulse(animationTime, ACTIVE_PULSE_DURATION_MS)
  const pulseColor = theme.neutralPulse[Math.min(
    theme.neutralPulse.length - 1,
    Math.floor(pulseProgress * theme.neutralPulse.length),
  )]!

  if (modelLoadActivity !== null && modelResidency !== null) {
    if (modelResidency._tag === 'Failed') {
      return (
        <box style={{ height: 1, flexShrink: 0 }}>
          <text style={{ fg: theme.status.warning }}>
            <span style={{ fg: theme.status.failure }}>■</span>
            {` ${LOW_MEMORY_MODEL_STOPPED_MESSAGE}`}
          </text>
        </box>
      )
    }
    if (modelResidency._tag === 'Stopping') {
      return (
        <box style={{ height: 1, flexShrink: 0 }}>
          <text>
            <span style={{ fg: pulseColor }}>■</span>
            {' '}
            <span style={{ fg: theme.text.body }}>Stopping model</span>
            <span style={{ fg: theme.text.metadata }}>{` · ${modelReleaseReasonLabel(modelResidency.reason)}`}</span>
          </text>
        </box>
      )
    }
    if (modelResidency._tag === 'Loading') {
      const percentage = Math.min(100, Math.max(0, Math.round(
        Option.getOrElse(modelResidency.progress, () => 0) * 100,
      )))
      return (
        <box style={{ height: 1, flexShrink: 0, flexDirection: 'row' }}>
          <text>
            <span style={{ fg: theme.accent }}>{spinnerFrameForStep(loadingSpinnerStep)}</span>
            {' '}
            <span style={{ fg: theme.text.body }}>Loading model</span>
            <span style={{ fg: theme.text.metadata }}>{` · ${percentage}%`}</span>
          </text>
          <Button onClick={() => onStopModel(modelLoadActivity.slotId)}>
            <text style={{ fg: theme.text.metadata }} attributes={TextAttributes.DIM}>{' · Stop'}</text>
          </Button>
        </box>
      )
    }
  }

  if (status?._tag === 'Working' && stabilizedDetail !== null) {
    const detail = rootDetailSegments(stabilizedDetail)
    const elapsed = formatElapsed(displayRootStatusElapsedMs(status, Date.now()))
    return (
      <box style={{ height: 1, flexShrink: 0 }}>
        <text>
          <span style={{ fg: pulseColor }}>●</span>
          {' '}
          <span style={{ fg: theme.text.body }}>Working</span>
          <span style={{ fg: theme.text.metadata }}>{` · ${elapsed}`}</span>
          {detail.keyword !== null && (
            <>
              <span style={{ fg: theme.text.metadata }}>{' · '}</span>
              <span style={{ fg: pulseColor }}>{detail.keyword}</span>
            </>
          )}
          {detail.detail !== null && (
            <span style={{ fg: theme.text.metadata }}>{` · ${detail.detail}`}</span>
          )}
          {detail.trailing !== null && width >= 72 && (
            <span style={{ fg: theme.text.metadata }}>{` · ${detail.trailing}`}</span>
          )}
        </text>
      </box>
    )
  }

  if (interruptedMessage) {
    const interruptText = interruptedMessage.context === 'fork'
      ? '■ Agent stopped'
      : interruptedMessage.allKilled
        ? '■ All agents interrupted. What would you like to do?'
        : '■ Lead interrupted. What would you like to do?'
    return (
      <box style={{ height: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.status.interrupted }}>{interruptText}</text>
      </box>
    )
  }

  return <box style={{ height: 1, flexShrink: 0 }} />
})
