import { TextAttributes } from '@opentui/core'
import { useAnimationStep } from '../../hooks/use-animation-time'
import { useTheme } from '../../hooks/use-theme'

interface ContextUsageProps {
  readonly tokenUsage: number | null
  readonly hardCap: number | null
  readonly isCompacting?: boolean
}

const formatContextTokens = (tokens: number): string => tokens >= 1000
  ? `${Math.round(tokens / 1000)}k`
  : `${Math.round(tokens)}`

export const formatContextUsage = (
  tokenUsage: number | null,
  hardCap: number | null,
): string => {
  if (tokenUsage === null) {
    return hardCap === null
      ? '—'
      : `— / ${formatContextTokens(hardCap)}`
  }
  const used = formatContextTokens(tokenUsage)
  return hardCap === null
    ? used
    : `${used} / ${formatContextTokens(hardCap)} (${Math.round((tokenUsage / hardCap) * 100)}%)`
}

export const contextUsageWidth = (
  tokenUsage: number | null,
  hardCap: number | null,
  isCompacting: boolean,
): number => formatContextUsage(tokenUsage, hardCap).length + (isCompacting ? 8 : 0)

export function ContextUsage({
  tokenUsage,
  hardCap,
  isCompacting = false,
}: ContextUsageProps) {
  const theme = useTheme()
  const animationStep = useAnimationStep(isCompacting, 240)
  const display = formatContextUsage(tokenUsage, hardCap)

  if (!isCompacting) return <text style={{ fg: theme.text.metadata }}>{display}</text>

  const frame = animationStep % 6
  const active = (index: number) =>
    index <= frame && frame <= index + 2 ? TextAttributes.NONE : TextAttributes.DIM

  return (
    <text style={{ fg: theme.text.metadata }}>
      <span attributes={active(0)}>{'>'}</span>
      <span attributes={active(1)}>{'>'}</span>
      <span attributes={active(2)}>{'>'}</span>
      {` ${display} `}
      <span attributes={active(2)}>{'<'}</span>
      <span attributes={active(1)}>{'<'}</span>
      <span attributes={active(0)}>{'<'}</span>
    </text>
  )
}
