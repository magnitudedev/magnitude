import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../../hooks/use-theme'
import { useLocalWidth } from '../../hooks/use-local-width'
import { Button } from '../../components/button'
import {
  getDisplayWidth,
  truncateToDisplayWidth,
  type SlashCommandDefinition,
} from '@magnitudedev/client-common'

interface SlashCommandMenuProps {
  commands: SlashCommandDefinition[]
  selectedIndex: number
  onSelect: (command: SlashCommandDefinition) => void
  onHoverIndex?: (index: number) => void
}

export function getSlashCommandRowLayout(
  command: SlashCommandDefinition,
  menuWidth: number,
) {
  const label = `/${command.label}`
  const descriptionWidth = Math.max(
    0,
    menuWidth - 2 - getDisplayWidth(label) - 1,
  )
  const description = truncateToDisplayWidth(
    command.description.replace(/\s+/gu, ' ').trim(),
    descriptionWidth,
  )

  return { label, description, descriptionWidth }
}

export const SlashCommandMenu = memo(function SlashCommandMenu({
  commands,
  selectedIndex,
  onSelect,
  onHoverIndex,
}: SlashCommandMenuProps) {
  const theme = useTheme()
  const menu = useLocalWidth()

  if (commands.length === 0) return null

  return (
    <box
      ref={menu.ref}
      onSizeChange={menu.onSizeChange}
      style={{ flexDirection: 'column', paddingBottom: 1 }}
    >
      {commands.map((cmd, index) => {
        const isSelected = index === selectedIndex
        const { label, description, descriptionWidth } = getSlashCommandRowLayout(
          cmd,
          menu.width ?? 0,
        )
        return (
          <Button
            key={cmd.id}
            onClick={() => onSelect(cmd)}
            onMouseOver={() => onHoverIndex?.(index)}
            style={{
              flexDirection: 'row',
              height: 1,
              overflow: 'hidden',
              paddingLeft: 1,
              paddingRight: 1,
              backgroundColor: isSelected ? theme.background.selected : undefined,
            }}
          >
            <text style={{ fg: theme.accent, flexShrink: 0 }}>
              <span attributes={TextAttributes.BOLD}>{label}</span>
            </text>
            {description && (
              <>
                <text style={{ width: 1, flexShrink: 0 }}> </text>
                <text style={{ fg: theme.text.metadata, width: descriptionWidth, flexShrink: 0 }}>
                  {description}
                </text>
              </>
            )}
          </Button>
        )
      })}
    </box>
  )
})
