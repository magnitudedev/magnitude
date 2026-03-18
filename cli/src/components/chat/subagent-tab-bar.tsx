import { memo, useEffect, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../button'
import { BOX_CHARS } from '../../utils/ui-constants'
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

  if (tabs.length === 0) return null

  const mainSelected = selectedForkId === null
  const mainHovered = hoveredId === 'main'

  const mainBorderColor = mainSelected
    ? theme.primary
    : mainHovered
    ? theme.link
    : theme.border

  const mainFg = mainSelected
    ? theme.primary
    : mainHovered
    ? theme.link
    : theme.muted

  return (
    <box style={{ paddingLeft: 2, paddingRight: 2, paddingBottom: 1, flexDirection: 'row', gap: 1, flexWrap: 'wrap' }}>
      {/* Main tab */}
      <Button
        onClick={() => onSelect(null)}
        onMouseOver={() => setHoveredId('main')}
        onMouseOut={() => setHoveredId(null)}
      >
        <box style={{
          borderStyle: 'single',
          border: true,
          borderColor: mainBorderColor,
          customBorderChars: BOX_CHARS,
          paddingLeft: 1,
          paddingRight: 1,
          flexDirection: 'column',
        }}>
          <text style={{ fg: mainFg }}>Main</text>
          <text style={{ fg: mainFg }}>{' '}</text>
        </box>
      </Button>

      {/* Subagent tabs */}
      {tabs.map((tab) => {
        const exiting = tab.phase === 'exiting'
        const isSelected = selectedForkId === tab.forkId
        const isHovered = hoveredId === tab.forkId

        const active = isSelected || isHovered

        const borderColor = exiting ? theme.border : active ? theme.foreground : theme.border
        const nameFg = exiting ? theme.muted : active ? theme.foreground : theme.foreground
        const metaFg = exiting ? theme.muted : active ? theme.foreground : theme.muted
        const attrs = exiting ? TextAttributes.DIM : undefined

        const timer = formatElapsed(tab.startedAt, now)
        const toolStr = `${tab.toolCount}t`

        // Line 1: name (left) + timer + toolStr (right), padded to TAB_INNER_WIDTH
        const rightPart = `${timer} ${toolStr}`
        const nameMaxLen = TAB_INNER_WIDTH - rightPart.length - 1
        const namePart = truncate(tab.name, nameMaxLen)
        const line1 = padRight(namePart, TAB_INNER_WIDTH - rightPart.length) + rightPart

        // Line 2: status line, truncated
        const line2 = truncate(tab.statusLine, TAB_INNER_WIDTH)

        return (
          <Button
            key={tab.forkId}
            onClick={() => onSelect(tab.forkId)}
            onMouseOver={() => setHoveredId(tab.forkId)}
            onMouseOut={() => setHoveredId(null)}
          >
            <box style={{
              borderStyle: 'single',
              border: true,
              borderColor,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
              flexDirection: 'column',
              width: TAB_INNER_WIDTH + 4, // +2 padding +2 border
            }}>
              <box style={{ flexDirection: 'row', height: 1 }}>
                <text style={{ fg: nameFg }} attributes={attrs}>{namePart}</text>
                <text style={{ fg: metaFg }} attributes={attrs}>{' '.repeat(Math.max(0, TAB_INNER_WIDTH - namePart.length - rightPart.length))}{rightPart}</text>
              </box>
              <text style={{ fg: metaFg }} attributes={attrs}>{line2}</text>
            </box>
          </Button>
        )
      })}
    </box>
  )
})