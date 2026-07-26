import { TextAttributes } from '@opentui/core'
import type { ReasoningEffort } from '@magnitudedev/sdk'
import stringWidth from 'string-width'
import { violet } from '../../utils/theme'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../../components/button'

export interface ThinkingSelectorOption {
  readonly value: ReasoningEffort
  readonly label: string
}

export const thinkingSelectorWidth = (
  options: readonly ThinkingSelectorOption[],
): number => 5 + options.reduce(
  (width, option, index) => width + stringWidth(option.label) + (index === 0 ? 0 : 2),
  0,
)

export const moveThinkingPreview = (
  index: number,
  direction: -1 | 1,
  optionCount: number,
): number => optionCount === 0
  ? 0
  : (index + direction + optionCount) % optionCount

export function ThinkingSelector({
  options,
  previewIndex,
  onPreview,
  onCommit,
}: {
  readonly options: readonly ThinkingSelectorOption[]
  readonly previewIndex: number
  readonly onPreview: (index: number) => void
  readonly onCommit: (index: number) => void
}) {
  const theme = useTheme()

  return (
    <box style={{ flexDirection: 'row', alignItems: 'center', flexShrink: 0 }}>
      <box style={{ width: 2, flexShrink: 0 }} />
      <text style={{ fg: theme.muted }}>{'>'}</text>
      <box style={{ width: 2, flexShrink: 0 }} />
      {options.map((option, index) => {
        const selected = index === previewIndex
        return (
          <box key={option.value} style={{ flexDirection: 'row', flexShrink: 0 }}>
            {index > 0 && <box style={{ width: 2, flexShrink: 0 }} />}
            <Button
              onClick={() => onCommit(index)}
              onMouseOver={() => onPreview(index)}
            >
              <text style={{ fg: selected ? violet[200] : theme.foreground }}>
                <span attributes={selected ? TextAttributes.UNDERLINE : TextAttributes.NONE}>
                  {option.label}
                </span>
              </text>
            </Button>
          </box>
        )
      })}
    </box>
  )
}
