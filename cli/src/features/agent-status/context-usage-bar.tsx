import React, { useSyncExternalStore } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../../hooks/use-theme'
import { subscribeAnimationTick, getAnimationTickSnapshot } from '@magnitudedev/client-common'

interface ContextUsageBarProps {
  tokenUsage: number | null
  hardCap: number | null
  isCompacting?: boolean
}

const formatTokens = (n: number): string => {
  if (n >= 1000) {
    const value = (n / 1000).toFixed(1)
    return (value.endsWith('.0') ? value.slice(0, -2) : value) + 'k'
  }
  return String(n)
}

export const formatContextUsage = (
  tokenUsage: number | null,
  hardCap: number | null,
): string => tokenUsage == null
  ? (hardCap == null ? '-' : '-/' + formatTokens(hardCap))
  : (() => {
      const tokens = hardCap == null
        ? formatTokens(tokenUsage) + '/Unknown'
        : formatTokens(tokenUsage) + '/' + formatTokens(hardCap)
      return hardCap == null
        ? tokens
        : Math.round((tokenUsage / hardCap) * 100) + '% ' + tokens
    })()

export const contextUsageWidth = (
  tokenUsage: number | null,
  hardCap: number | null,
  isCompacting: boolean,
): number => formatContextUsage(tokenUsage, hardCap).length + (isCompacting ? 8 : 0)

/**
 * Context usage indicator rendered on the right side of the agent-status row.
 * Shows "1% 45k/272k".
 *
 * During compaction, inward-pointing arrows animate on each side of
 * the percent and token count.
 */
export const ContextUsageBar = ({ tokenUsage, hardCap, isCompacting = false }: ContextUsageBarProps) => {
  const theme = useTheme()
  const tick = useSyncExternalStore(subscribeAnimationTick, getAnimationTickSnapshot, getAnimationTickSnapshot)
  // 200ms / 80ms per tick ≈ 3 ticks per frame
  const frame = isCompacting ? Math.floor(tick / 3) : 0

  const displayText = formatContextUsage(tokenUsage, hardCap)

  // Normal (non-compacting) rendering
  if (!isCompacting) {
    return <text style={{ fg: theme.muted }}>{displayText}</text>
  }

  // Compacting rendering: 6-frame inward-sweeping arrow wave
  // Wave moves from outer edges toward center, then resets
  // Left (>>>) positions: 0=outer, 1=mid, 2=inner
  // Right (<<<) positions: 0=inner, 1=mid, 2=outer
  type A = 'dim' | 'bright'
  const left: [A, A, A][] = [
    ['bright', 'dim',    'dim'   ],
    ['bright', 'bright', 'dim'   ],
    ['bright', 'bright', 'bright'],
    ['dim',    'bright', 'bright'],
    ['dim',    'dim',    'bright'],
    ['dim',    'dim',    'dim'   ],
  ]
  const right: [A, A, A][] = [
    ['dim',    'dim',    'bright'],
    ['dim',    'bright', 'bright'],
    ['bright', 'bright', 'bright'],
    ['bright', 'bright', 'dim'   ],
    ['bright', 'dim',    'dim'   ],
    ['dim',    'dim',    'dim'   ],
  ]
  const f = frame % 6
  const attr = (v: A) => v === 'bright' ? TextAttributes.NONE : TextAttributes.DIM

  return (
    <text>
      <span fg={theme.muted} attributes={attr(left[f][0])}>{'>'}</span>
      <span fg={theme.muted} attributes={attr(left[f][1])}>{'>'}</span>
      <span fg={theme.muted} attributes={attr(left[f][2])}>{'>'}</span>
      <span fg={theme.muted}>{' ' + displayText + ' '}</span>
      <span fg={theme.muted} attributes={attr(right[f][0])}>{'<'}</span>
      <span fg={theme.muted} attributes={attr(right[f][1])}>{'<'}</span>
      <span fg={theme.muted} attributes={attr(right[f][2])}>{'<'}</span>
    </text>
  )
}
