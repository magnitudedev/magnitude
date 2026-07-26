import { useState, useSyncExternalStore, type ReactNode } from 'react'
import { Option } from 'effect'
import type { LocalInferenceView } from '@magnitudedev/client-common'
import {
  getAnimationTickSnapshot,
  subscribeAnimationTick,
} from '@magnitudedev/client-common'
import { ProviderIdSchema } from '@magnitudedev/sdk'
import { Button } from '../../components/button'
import { useTheme } from '../../hooks/use-theme'
import { BOX_CHARS } from '../../utils/ui-constants'
import { orange, slate } from '../../utils/theme'

const LOCAL_PROVIDER_ID = ProviderIdSchema.make('local')
const LOADING_FRAMES = ['◐', '◓', '◑', '◒'] as const

export type InferenceRuntimeStatus = 'running' | 'starting' | 'stopping' | 'idle' | 'stopped' | 'checking'

export interface InferenceRuntimeBadgeView {
  readonly status: InferenceRuntimeStatus
  readonly memoryLabel: string | null
}

export const inferenceRuntimeTextColors = (hovered: boolean) => hovered
  ? { label: slate[300], separator: slate[400], value: slate[200] }
  : { label: slate[400], separator: slate[500], value: slate[300] }

const compactGiB = (bytes: number): string =>
  (bytes / 1024 ** 3).toFixed(1).replace(/\.0$/, '')

const runtimeMemoryBytes = (state: LocalInferenceView): number | null =>
  Option.match(state.hardware.residentMemory, {
    onNone: () => null,
    onSome: ({ domains }) => domains.reduce(
      (total, domain) => total
        + domain.modelBytes
        + domain.contextBytes
        + domain.computeBytes
        + domain.auxiliaryBytes,
      0,
    ),
  })

export const deriveInferenceRuntimeBadgeView = (
  state: LocalInferenceView | null,
): InferenceRuntimeBadgeView => {
  if (state === null) return { status: 'checking', memoryLabel: null }

  const localSlots = [state.slots.slots.primary, state.slots.slots.secondary]
    .filter((slot) => slot._tag !== 'Unassigned' && slot.selection.providerId === LOCAL_PROVIDER_ID)
  if (localSlots.some((slot) => slot._tag === 'LoadingLocalModel')) {
    return { status: 'starting', memoryLabel: null }
  }
  if (localSlots.some((slot) => slot._tag === 'UnloadingLocalModel')) {
    return { status: 'stopping', memoryLabel: null }
  }

  const memoryBytes = runtimeMemoryBytes(state)
  if (memoryBytes !== null) {
    return { status: 'running', memoryLabel: `${compactGiB(memoryBytes)} GB` }
  }
  if (Option.isSome(state.hardware.runtimeFailure)) {
    return { status: 'stopped', memoryLabel: null }
  }
  if (localSlots.some((slot) => slot._tag === 'Ready')) {
    return { status: 'starting', memoryLabel: null }
  }
  return { status: 'idle', memoryLabel: null }
}

const inferenceRuntimeValueLabel = (view: InferenceRuntimeBadgeView): string =>
  view.status === 'running'
    ? view.memoryLabel ?? 'Running'
    : view.status === 'starting'
      ? 'Starting…'
      : view.status === 'stopping'
        ? 'Stopping…'
        : view.status === 'stopped'
          ? 'Stopped'
          : view.status === 'checking'
            ? 'Checking…'
            : 'Idle'

export const inferenceRuntimeBadgeLabel = (
  view: InferenceRuntimeBadgeView,
  compact: boolean,
): string => `${compact ? 'Local' : 'Local inference'}  ·  ${inferenceRuntimeValueLabel(view)}`

export function InferenceRuntimeBadge({
  view,
  compact,
  onOpenHardware,
}: {
  readonly view: InferenceRuntimeBadgeView
  readonly compact: boolean
  readonly onOpenHardware: () => void
}): ReactNode {
  const theme = useTheme()
  const [hovered, setHovered] = useState(false)
  const tick = useSyncExternalStore(
    subscribeAnimationTick,
    getAnimationTickSnapshot,
    getAnimationTickSnapshot,
  )
  const transitional = view.status === 'starting'
    || view.status === 'stopping'
    || view.status === 'checking'
  const glyph = transitional
    ? LOADING_FRAMES[Math.floor(tick / 2) % LOADING_FRAMES.length]
    : view.status === 'idle' ? '○' : '●'
  const color = transitional
    ? orange[400]
    : view.status === 'running'
      ? theme.success
      : view.status === 'stopped'
        ? theme.error
        : theme.muted
  const valueLabel = inferenceRuntimeValueLabel(view)
  const textColors = inferenceRuntimeTextColors(hovered)

  return (
    <Button
      onClick={onOpenHardware}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      style={{
        borderStyle: 'single',
        borderColor: theme.border,
        customBorderChars: BOX_CHARS,
        backgroundColor: theme.terminalDetectedBg ?? 'transparent',
        paddingLeft: 1,
        paddingRight: 1,
        flexShrink: 0,
      }}
    >
      <text>
        <span style={{ fg: color }}>{glyph}</span>
        <span style={{ fg: textColors.label }}>{` ${compact ? 'Local' : 'Local inference'}`}</span>
        <span style={{ fg: textColors.separator }}>{'  ·  '}</span>
        <span style={{ fg: textColors.value }}>{valueLabel}</span>
      </text>
    </Button>
  )
}

export function InferenceRuntimeBadgeOverlay(props: {
  readonly view: InferenceRuntimeBadgeView
  readonly compact: boolean
  readonly onOpenHardware: () => void
}): ReactNode {
  return (
    <box style={{ position: 'absolute', top: 0, right: 2, zIndex: 100 }}>
      <InferenceRuntimeBadge {...props} />
    </box>
  )
}
