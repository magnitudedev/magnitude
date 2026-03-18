import { memo, useEffect, useState } from 'react'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../button'
import { TAB_BORDER_CHARS } from '../../utils/ui-constants'
import { slate } from '../../utils/palette'

import type { SubagentTabItem } from './types'

const TAB_INNER_WIDTH = 36 // inner content width (chars), excluding border+padding

const PULSE_SLATE_SHADES = [
  slate[200],
  slate[300],
  slate[400],
  slate[500],
  slate[600],
  slate[700],
  slate[600],
  slate[500],
  slate[400],
  slate[300],
  slate[200],
] as const

type Props = {
  tabs: readonly SubagentTabItem[]
  selectedForkId: string | null // null = Main selected
  onSelect: (forkId: string | null) => void
}

function formatElapsedMs(elapsedMs: number): string {
  const elapsed = Math.max(0, Math.floor(elapsedMs / 1000))
  const m = Math.floor(elapsed / 60)
  const s = elapsed % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

function truncate(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str
  return str.slice(0, maxLen - 1) + '…'
}

function padRight(str: string, len: number): string {
  if (str.length >= len) return str
  return str + ' '.repeat(len - str.length)
}

export const SubagentTabBar = memo(function SubagentTabBar({ tabs, selectedForkId, onSelect }: Props) {
  const theme = useTheme()
  const [now, setNow] = useState(() => Date.now())
  const [hoveredId, setHoveredId] = useState<null | string | 'main'>(null)

  const hasActiveTabs = tabs.some((tab) => tab.phase === 'active')

  useEffect(() => {
    if (tabs.length === 0) return
    const tickMs = hasActiveTabs ? 200 : 1000
    setNow(Date.now())
    const interval = setInterval(() => setNow(Date.now()), tickMs)
    return () => clearInterval(interval)
  }, [hasActiveTabs, tabs.length])

  const mainSelected = selectedForkId === null
  const mainHovered = hoveredId === 'main'

  const mainTextColor = mainSelected || mainHovered ? theme.link : theme.primary
  const mainBorderColor = mainSelected || mainHovered ? theme.link : theme.border

  return (
    <box style={{ flexDirection: 'row', flexWrap: 'wrap', flexShrink: 0, alignItems: 'flex-start' }}>
      <Button
        style={{ alignSelf: 'flex-start' }}
        onClick={() => onSelect(null)}
        onMouseOver={() => setHoveredId('main')}
        onMouseOut={() => setHoveredId(null)}
      >
        <box
          style={{
            borderStyle: 'single',
            border: ['left', 'right', 'top', 'bottom'],
            borderColor: mainBorderColor,
            customBorderChars: TAB_BORDER_CHARS,
          }}
        >
          <box style={{ paddingLeft: 1, paddingRight: 1, flexDirection: 'column', height: 2, minHeight: 2, maxHeight: 2 }}>
            <box style={{ flexDirection: 'row', height: 1 }}>
              <text style={{ fg: mainTextColor }}>Main Agent</text>
            </box>
            <box style={{ flexDirection: 'row', height: 1 }}>
              <text style={{ fg: mainTextColor }}> </text>
            </box>
          </box>
        </box>
      </Button>

      {tabs.length === 0 ? (
        <box style={{ paddingLeft: 3, paddingRight: 1, flexDirection: 'column', height: 3, minHeight: 3, maxHeight: 3 }}>
          <text style={{ fg: theme.foreground }}> </text>
          <text style={{ fg: theme.foreground }}>No subagents yet.</text>
          <text style={{ fg: theme.muted }}>Subagent activity will appear here.</text>
        </box>
      ) : tabs.map((tab) => {
        const isIdle = tab.phase === 'idle'
        const isSelected = selectedForkId === tab.forkId
        const isHovered = hoveredId === tab.forkId

        const tabBorderColor = isSelected || isHovered ? theme.foreground : theme.border
        const tabTextColor = isSelected || isHovered ? theme.foreground : theme.muted
        const tabIdColor = theme.foreground
        const tabRestLine1Color = isSelected || isHovered ? theme.foreground : theme.muted

        const elapsedMs = tab.phase === 'active'
          ? tab.accumulatedActiveMs + Math.max(0, now - tab.activeSince)
          : tab.accumulatedActiveMs
        const timer = formatElapsedMs(elapsedMs)
        const resumedMark = tab.resumeCount > 0 ? '↺ ' : ''
        const detailsPart = `• ${tab.toolCount} tools • ${resumedMark}${timer}`
        const line1Prefix = isIdle ? '○ ' : '● '
        const nameMaxLen = TAB_INNER_WIDTH - line1Prefix.length - detailsPart.length - 1
        const namePart = truncate(tab.agentId, Math.max(1, nameMaxLen))
        const line1Rest = ` ${detailsPart}`
        // const line2 = truncate(tab.toolSummaryLine, TAB_INNER_WIDTH)
        const line2 = truncate(tab.statusLine, TAB_INNER_WIDTH)
        const isRunning = tab.phase === 'active'
        const pulseColor = PULSE_SLATE_SHADES[Math.floor(now / 200) % PULSE_SLATE_SHADES.length]
        const dotColor = isRunning ? pulseColor : slate[500]

        return (
          <Button
            key={tab.forkId}
            onClick={() => onSelect(tab.forkId)}
            onMouseOver={() => setHoveredId(tab.forkId)}
            onMouseOut={() => setHoveredId(null)}
          >
            <box
              style={{
                flexDirection: 'row',
                alignItems: 'flex-start',
                borderStyle: 'single',
                border: ['left', 'right', 'top', 'bottom'],
                borderColor: tabBorderColor,
                customBorderChars: TAB_BORDER_CHARS,
              }}
            >
              <box style={{ paddingLeft: 1, paddingRight: 1, flexDirection: 'column', width: TAB_INNER_WIDTH + 2, height: 2, minHeight: 2, maxHeight: 2 }}>
                <box style={{ flexDirection: 'row' }}>
                  <text style={{ fg: dotColor }}>{line1Prefix}</text>
                  <text style={{ fg: tabIdColor }}>{namePart}</text>
                  <text style={{ fg: tabRestLine1Color }}>{line1Rest}</text>
                </box>
                {/* <text style={{ fg: tabTextColor }}>{line2}</text> */}
                <text style={{ fg: tabTextColor }}>{line2}</text>
              </box>
            </box>
          </Button>
        )
      })}
    </box>
  )
})