import { memo } from 'react'
import { Option } from 'effect'
import { TextAttributes } from '@opentui/core'
import type { DisplayRootStatus, InterruptedMessage, ModelInstanceId } from '@magnitudedev/sdk'
import {
  displayRootStatusElapsedMs,
  modelReleaseReasonLabel,
  rootDetailSegments,
  slate,
  useStabilizedRootDetail,
  type LocalModelLoadActivity,
} from '@magnitudedev/client-common'
import { useTheme } from '../../hooks/use-theme'
import { red } from '../../utils/theme'
import { spinnerFrameForTick } from '../../hooks/use-spinner-frame'
import { useAnimationTick } from '../../hooks/use-animation-tick'
import { Button } from '../../components/button'

const ACTIVE_PULSE_COLORS = [
  slate[100], slate[200], slate[300], slate[400], slate[500],
  slate[400], slate[300], slate[200],
] as const

const LOW_MEMORY_MODEL_STOPPED_MESSAGE =
  'Model stopped · Low memory - close memory-intensive apps and try again'

interface ActivityRailProps {
  status: DisplayRootStatus | null
  width: number
  modelLoadActivity: LocalModelLoadActivity | null
  onStopModel: (instanceId: ModelInstanceId) => void
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
  const modelLifecycle = modelLoadActivity === null
    ? null
    : Option.getOrThrow(modelLoadActivity.instance).lifecycle
  const active = status?._tag === 'Working'
  const stabilizedDetail = useStabilizedRootDetail(status)
  const modelAnimated = modelLifecycle?._tag === 'Loading' || modelLifecycle?._tag === 'Stopping'
  const tick = useAnimationTick(modelAnimated || active)
  const pulseColor = ACTIVE_PULSE_COLORS[Math.floor(tick / 4) % ACTIVE_PULSE_COLORS.length]

  if (modelLoadActivity !== null && modelLifecycle !== null) {
    const instance = Option.getOrThrow(modelLoadActivity.instance)
    if (
      modelLifecycle._tag === 'Failed'
      || modelLifecycle._tag === 'Stopped' && modelLifecycle.reason === 'memory_pressure'
    ) {
      return (
        <box style={{ height: 1, flexShrink: 0 }}>
          <text style={{ fg: theme.warning }}>
            <span style={{ fg: red[400] }}>■</span>
            {` ${LOW_MEMORY_MODEL_STOPPED_MESSAGE}`}
          </text>
        </box>
      )
    }
    if (modelLifecycle._tag === 'Stopping') {
      return (
        <box style={{ height: 1, flexShrink: 0 }}>
          <text>
            <span style={{ fg: pulseColor }}>■</span>
            {' '}
            <span style={{ fg: theme.foreground }}>Stopping model</span>
            <span style={{ fg: theme.muted }}>{` · ${modelReleaseReasonLabel(modelLifecycle.reason)}`}</span>
          </text>
        </box>
      )
    }
    if (modelLifecycle._tag === 'Loading') {
      const percentage = Math.min(100, Math.max(0, Math.round(
        Option.getOrElse(modelLifecycle.progress, () => 0) * 100,
      )))
      return (
        <box style={{ height: 1, flexShrink: 0, flexDirection: 'row' }}>
          <text>
            <span style={{ fg: theme.primary }}>{spinnerFrameForTick(tick)}</span>
            {' '}
            <span style={{ fg: theme.foreground }}>Loading model</span>
            <span style={{ fg: theme.muted }}>{` · ${percentage}%`}</span>
          </text>
          <Button onClick={() => onStopModel(instance.id)}>
            <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{' · Stop'}</text>
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
          <span style={{ fg: theme.foreground }}>Working</span>
          <span style={{ fg: theme.muted }}>{` · ${elapsed}`}</span>
          {detail.keyword !== null && (
            <>
              <span style={{ fg: theme.muted }}>{' · '}</span>
              <span style={{ fg: pulseColor }}>{detail.keyword}</span>
            </>
          )}
          {detail.detail !== null && (
            <span style={{ fg: theme.muted }}>{` · ${detail.detail}`}</span>
          )}
          {detail.trailing !== null && width >= 72 && (
            <span style={{ fg: theme.muted }}>{` · ${detail.trailing}`}</span>
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
        <text style={{ fg: red[400] }}>{interruptText}</text>
      </box>
    )
  }

  return <box style={{ height: 1, flexShrink: 0 }} />
})
