import { useAnimationStep } from '../../hooks/use-animation-time'
import { useTheme } from '../../hooks/use-theme'
import type { LocalInferenceFooterView } from '../local-inference/footer-status'

const LOADING_FRAMES = ['◐', '◓', '◑', '◒'] as const

export function ResidencyIndicator({
  residency,
}: {
  readonly residency: NonNullable<LocalInferenceFooterView['residency']>
}) {
  const theme = useTheme()
  const animationStep = useAnimationStep(residency === 'loading', 160)

  if (residency === 'loaded') return <text style={{ fg: theme.success }}>●</text>
  if (residency === 'not_loaded') return <text style={{ fg: theme.muted }}>○</text>
  return <text style={{ fg: theme.warning }}>{LOADING_FRAMES[animationStep % LOADING_FRAMES.length]}</text>
}
