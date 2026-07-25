import { useSyncExternalStore } from 'react'
import {
  getAnimationTickSnapshot,
  subscribeAnimationTick,
} from '@magnitudedev/client-common'
import { useTheme } from '../../hooks/use-theme'
import type { LocalInferenceFooterView } from '../local-inference/footer-status'

const LOADING_FRAMES = ['◐', '◓', '◑', '◒'] as const

export function ResidencyIndicator({
  residency,
}: {
  readonly residency: NonNullable<LocalInferenceFooterView['residency']>
}) {
  const theme = useTheme()
  const tick = useSyncExternalStore(
    subscribeAnimationTick,
    getAnimationTickSnapshot,
    getAnimationTickSnapshot,
  )

  if (residency === 'loaded') return <text style={{ fg: theme.success }}>●</text>
  if (residency === 'not_loaded') return <text style={{ fg: theme.muted }}>○</text>
  return <text style={{ fg: theme.warning }}>{LOADING_FRAMES[Math.floor(tick / 2) % LOADING_FRAMES.length]}</text>
}
