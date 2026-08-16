import { useTheme } from '../hooks/use-theme'

interface ProgressBarProps {
  /** 0..1 — clamped */
  value: number
  width: number
}

export const ProgressBar = ({ value, width }: ProgressBarProps) => {
  const theme = useTheme()
  const clamped = Math.max(0, Math.min(1, value))
  const filled = Math.round(clamped * width)
  const empty = Math.max(0, width - filled)
  return (
    <text>
      <span fg={theme.accent}>{'█'.repeat(filled)}</span>
      <span fg={theme.text.supporting}>{'░'.repeat(empty)}</span>
    </text>
  )
}
