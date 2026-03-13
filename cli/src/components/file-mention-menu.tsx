import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'

export type FileMentionMenuItem = {
  path: string
  kind: 'file' | 'directory'
  contentType: 'text' | 'image' | 'directory'
  warning?: boolean
}

interface FileMentionMenuProps {
  isOpen: boolean
  query?: string
  items: FileMentionMenuItem[]
  recentItems?: FileMentionMenuItem[]
  overflowCount?: number
  selectedIndex: number
  onSelect: (item: FileMentionMenuItem) => void
  onHoverIndex?: (index: number) => void
}

export const FileMentionMenu = memo(function FileMentionMenu({
  isOpen,
  query = '',
  items,
  recentItems = [],
  overflowCount = 0,
  selectedIndex,
  onSelect,
  onHoverIndex,
}: FileMentionMenuProps) {
  const theme = useTheme()

  if (!isOpen) return null

  const hasQuery = query.trim().length > 0

  return (
    <box style={{ flexDirection: 'column', paddingBottom: 1, maxHeight: 16, overflow: 'hidden' }}>
      {!hasQuery && (
        <>
          <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
            Recent
          </text>
          {recentItems.length === 0 ? (
            <text style={{ fg: theme.muted, paddingLeft: 1 }} attributes={TextAttributes.DIM}>
              (none yet)
            </text>
          ) : null}
        </>
      )}

      <text style={{ fg: theme.muted, paddingTop: hasQuery ? 0 : 1 }} attributes={TextAttributes.DIM}>
        Files & directories
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
                key={`${item.kind}:${item.path}`}
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
                {item.kind === 'directory' && (
                  <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
                    {' '}{'[dir]'}
                  </text>
                )}
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