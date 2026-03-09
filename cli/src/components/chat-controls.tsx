import { memo, useState } from 'react'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'

interface HistoryLoadMoreProps {
  hiddenCount: number
  onLoadMore: () => void
}

export const LoadPreviousButton = memo(function LoadPreviousButton({
  hiddenCount,
  onLoadMore
}: HistoryLoadMoreProps) {
  const theme = useTheme()
  const [isHovered, setIsHovered] = useState(false)

  return (
    <Button
      onClick={onLoadMore}
      onMouseOver={() => setIsHovered(true)}
      onMouseOut={() => setIsHovered(false)}
      style={{
        marginBottom: 1,
        paddingLeft: 1,
        paddingRight: 1,
      }}
    >
      <text style={{ fg: isHovered ? theme.foreground : theme.muted }}>
        ↑ Load {hiddenCount} previous message{hiddenCount === 1 ? '' : 's'}
      </text>
    </Button>
  )
})