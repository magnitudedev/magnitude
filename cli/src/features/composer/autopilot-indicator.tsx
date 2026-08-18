import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { Button } from '../../components/button'
import { spinnerFrameForStep } from '../../hooks/use-spinner-frame'
import { useAnimationStep } from '../../hooks/use-animation-time'
import { useTheme } from '../../hooks/use-theme'

interface AutopilotIndicatorProps {
  enabled: boolean
  generating: boolean
  onToggle: () => void
}

export const AutopilotIndicator = memo(function AutopilotIndicator({
  enabled,
  generating,
  onToggle,
}: AutopilotIndicatorProps) {
  const theme = useTheme()
  const [hovered, setHovered] = useState(false)
  const animationStep = useAnimationStep(enabled || generating, generating ? 80 : 240)

  const iconContent = generating
    ? spinnerFrameForStep(animationStep)
    : '●'

  const iconColor = generating
    ? theme.text.body
    : enabled
      ? theme.activityPulse[animationStep % theme.activityPulse.length]!
      : theme.status.inactive

  const textColor = hovered ? theme.text.body : enabled ? theme.text.body : theme.text.supporting
  const textAttributes = (!enabled && !hovered) ? TextAttributes.DIM : TextAttributes.NONE

  return (
    <Button
      onClick={onToggle}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      cursor="pointer"
    >
      <text style={{ fg: textColor }} attributes={textAttributes}>
        <span style={{ fg: iconColor }}>{iconContent + ' '}</span>
        {enabled ? 'Autopilot ON' : 'Autopilot OFF'}
      </text>
    </Button>
  )
})
