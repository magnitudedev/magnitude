import { memo, useEffect, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../button'
import { TAB_BORDER_CHARS } from '../../utils/ui-constants'

import type { SubagentTabItem } from './types'

const TAB_INNER_WIDTH = 36 // inner content width (chars), excluding border+padding

type Props = {
  tabs: readonly SubagentTabItem[]
  selectedForkId: string | null // null = Main selected
  onSelect: (forkId: string | null) => void
}

function formatElapsed(startedAt: number, now: number): string {
  const elapsed = Math.max(0, Math.floor((now - startedAt) / 1000))
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

  useEffect(() => {
    if (tabs.length === 0) return
    const interval = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(interval)
  }, [tabs.length])

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
          <text style={{ fg: theme.foreground }}>No active subagents.</text>
          <text style={{ fg: theme.muted }}>Live subagent activity will appear here.</text>
        </box>
      ) : tabs.map((tab) => {
        const exiting = tab.phase === 'exiting'
        const isSelected = selectedForkId === tab.forkId
        const isHovered = hoveredId === tab.forkId
        const attrs = exiting ? TextAttributes.DIM : undefined

        const tabBorderColor = isSelected || isHovered ? theme.foreground : theme.border
        const tabTextColor = isSelected || isHovered ? theme.foreground : theme.muted
        const tabIdColor = theme.foreground
        const tabRestLine1Color = isSelected || isHovered ? theme.foreground : theme.muted

        const timer = formatElapsed(tab.startedAt, now)
        const detailsPart = `• ${tab.toolCount} tools • ${timer}`
        const nameMaxLen = TAB_INNER_WIDTH - detailsPart.length - 1
        const namePart = truncate(tab.agentId, Math.max(1, nameMaxLen))
        const line1Rest = ` ${detailsPart}`
        // const line2 = truncate(tab.toolSummaryLine, TAB_INNER_WIDTH)
        const line2 = truncate(tab.statusLine, TAB_INNER_WIDTH)

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
                  <text style={{ fg: tabIdColor }} attributes={attrs}>{namePart}</text>
                  <text style={{ fg: tabRestLine1Color }} attributes={attrs}>{line1Rest}</text>
                </box>
                {/* <text style={{ fg: tabTextColor }} attributes={attrs}>{line2}</text> */}
                <text style={{ fg: tabTextColor }} attributes={attrs}>{line2}</text>
              </box>
            </box>
          </Button>
        )
      })}
    </box>
  )
})