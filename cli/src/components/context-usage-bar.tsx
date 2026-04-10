import React, { useState, useEffect } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'

interface ContextUsageBarProps {
  tokenUsage: number | null
  hardCap: number | null
  isCompacting?: boolean
}

/**
 * Context usage indicator, rendered inline next to the PWD in the footer.
 * Shows " · 1%  45k/272k" (middle-dot separator, double space between % and tokens).
 *
 * During compaction, inward-pointing arrows animate on each side of
 * the percent and token count.
 */
export const ContextUsageBar = ({ tokenUsage, hardCap, isCompacting = false }: ContextUsageBarProps) => {
  const theme = useTheme()
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    if (!isCompacting) {
      setFrame(0)
      return
    }
    const interval = setInterval(() => setFrame(f => f + 1), 200)
    return () => clearInterval(interval)
  }, [isCompacting])

  const formatTokens = (n: number) => {
    if (n >= 1000) {
      const v = (n / 1000).toFixed(1)
      return (v.endsWith('.0') ? v.slice(0, -2) : v) + 'k'
    }
    return String(n)
  }

  const displayText = tokenUsage == null
    ? (hardCap == null ? '-' : '-/' + formatTokens(hardCap))
    : (() => {
      const tokensStr = hardCap == null
        ? formatTokens(tokenUsage) + '/Unknown'
        : formatTokens(tokenUsage) + '/' + formatTokens(hardCap)
      return hardCap == null
        ? tokensStr
        : Math.round((tokenUsage / hardCap) * 100) + '% ' + tokensStr
    })()

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
