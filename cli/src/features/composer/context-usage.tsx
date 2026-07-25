import { TextAttributes } from '@opentui/core'
import { useSyncExternalStore } from 'react'
import {
  formatTokensCompact,
  getAnimationTickSnapshot,
  subscribeAnimationTick,
} from '@magnitudedev/client-common'
import { useTheme } from '../../hooks/use-theme'

interface ContextUsageProps {
  readonly tokenUsage: number | null
  readonly hardCap: number | null
  readonly isCompacting?: boolean
}

export const formatContextUsage = (
  tokenUsage: number | null,
  hardCap: number | null,
): string => {
  if (tokenUsage === null) return 'ctx —'
  const used = formatTokensCompact(tokenUsage)
  return hardCap === null
    ? `ctx ${used}`
    : `ctx ${used} (${Math.round((tokenUsage / hardCap) * 100)}%)`
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
  const tick = useSyncExternalStore(
    subscribeAnimationTick,
    getAnimationTickSnapshot,
    getAnimationTickSnapshot,
  )
  const display = formatContextUsage(tokenUsage, hardCap)

  if (!isCompacting) return <text style={{ fg: theme.muted }}>{display}</text>

  const frame = Math.floor(tick / 3) % 6
  const active = (index: number) =>
    index <= frame && frame <= index + 2 ? TextAttributes.NONE : TextAttributes.DIM

  return (
    <text style={{ fg: theme.muted }}>
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
