import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import type { SlashCommandDefinition } from '../commands/slash-commands'

interface SlashCommandMenuProps {
  commands: SlashCommandDefinition[]
  selectedIndex: number
  onSelect: (command: SlashCommandDefinition) => void
}

export const SlashCommandMenu = memo(function SlashCommandMenu({
  commands,
  selectedIndex,
  onSelect,
}: SlashCommandMenuProps) {
  const theme = useTheme()

  if (commands.length === 0) return null

  return (
    <box style={{ flexDirection: 'column', paddingBottom: 1 }}>
      {commands.map((cmd, index) => {
        const isSelected = index === selectedIndex
        return (
          <Button
            key={cmd.id}
            onClick={() => onSelect(cmd)}
            style={{
              flexDirection: 'row',
              paddingLeft: 1,
              paddingRight: 1,
              backgroundColor: isSelected ? theme.surface : undefined,
            }}
          >
            <text style={{ fg: theme.primary }}>
              <span attributes={TextAttributes.BOLD}>/{cmd.label}</span>
            </text>
            <text style={{ fg: theme.muted }}>
              {' '}{cmd.description}
            </text>
          </Button>
        )
      })}
    </box>
  )
})
