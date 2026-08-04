import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../../hooks/use-theme'
import { useAnimationStep } from '../../hooks/use-animation-time'

interface AutopilotPreviewMessageProps {
  content: string
  startTime: number // Date.now() when countdown started
  duration?: number // countdown duration in ms, default 3000
}

export const AutopilotPreviewMessage = memo(function AutopilotPreviewMessage({
  content,
  startTime,
  duration = 3000,
}: AutopilotPreviewMessageProps) {
  const theme = useTheme()
  useAnimationStep(true, 80)
  const now = Date.now()
  const progress = Math.min(1, (now - startTime) / duration)

  // Dim→solid transition: below 0.5 is DIM, above 0.5 is NONE
  const attributes = progress < 0.5 ? TextAttributes.DIM : TextAttributes.NONE

  return (
    <box style={{ flexDirection: 'row', marginBottom: 1, paddingLeft: 1 }}>
      <text style={{ fg: theme.muted }} attributes={attributes}>
        {'↑ '}
      </text>
      <text style={{ fg: theme.muted }} attributes={attributes}>
        {content}
      </text>
    </box>
  )
})
