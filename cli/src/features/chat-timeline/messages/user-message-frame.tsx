import type { ReactNode } from 'react'
import { BOX_CHARS } from '../../../utils/ui-constants'

interface UserMessageFrameProps {
  readonly borderColor: string
  readonly backgroundColor: string
  readonly children: ReactNode
}

export const UserMessageFrame = ({
  borderColor,
  backgroundColor,
  children,
}: UserMessageFrameProps) => (
  <box
    style={{
      borderStyle: 'single',
      border: ['left'],
      borderColor,
      customBorderChars: { ...BOX_CHARS, vertical: '┃' },
    }}
  >
    <box
      style={{
        flexDirection: 'row',
        backgroundColor,
        paddingTop: 1,
        paddingBottom: 1,
        paddingLeft: 1,
        paddingRight: 2,
        flexGrow: 1,
      }}
    >
      {children}
    </box>
  </box>
)
