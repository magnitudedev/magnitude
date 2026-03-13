import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'

export type FileMentionMenuItem = {
  path: string
  warning?: boolean
}

interface FileMentionMenuProps {
  isOpen: boolean
  items: FileMentionMenuItem[]
  overflowCount?: number
  selectedIndex: number
  onSelect: (item: FileMentionMenuItem) => void
  onHoverIndex?: (index: number) => void
}

export const FileMentionMenu = memo(function FileMentionMenu({
  isOpen,
  items,
  overflowCount = 0,
  selectedIndex,
  onSelect,
  onHoverIndex,
}: FileMentionMenuProps) {
  const theme = useTheme()

  if (!isOpen) return null

  return (
    <box style={{ flexDirection: 'column', paddingBottom: 1, maxHeight: 16, overflow: 'hidden' }}>
      <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
        Recent
      </text>
      <text style={{ fg: theme.muted, paddingLeft: 1 }} attributes={TextAttributes.DIM}>
        (none yet)
      </text>

      <text style={{ fg: theme.muted, paddingTop: 1 }} attributes={TextAttributes.DIM}>
        Files
      </text>

      {items.length === 0 ? (
        <text style={{ fg: theme.muted, paddingLeft: 1 }} attributes={TextAttributes.DIM}>
          No matching files
        </text>
      ) : (
        <>
          {items.map((item, index) => {
            const isSelected = index === selectedIndex
            return (
              <Button
                key={item.path}
                onClick={() => onSelect(item)}
                onMouseOver={() => onHoverIndex?.(index)}
                style={{
                  flexDirection: 'row',
                  paddingLeft: 1,
                  paddingRight: 1,
                  backgroundColor: isSelected ? theme.surface : undefined,
                }}
              >
                <text style={{ fg: theme.primary }}>
                  <span attributes={TextAttributes.BOLD}>@{item.path}</span>
                </text>
                {item.warning && (
                  <text style={{ fg: theme.warning }}>
                    {' '}{'[>500KB]'}
                  </text>
                )}
              </Button>
            )
          })}
          {overflowCount > 0 && (
            <text style={{ fg: theme.muted, paddingLeft: 1 }} attributes={TextAttributes.DIM}>
              … and {overflowCount} more
            </text>
          )}
        </>
      )}
    </box>
  )
})