import type { ReactNode } from 'react'
import { BOX_CHARS } from '../../../utils/ui-constants'

interface UserMessageFrameProps {
  readonly borderColor: string
  readonly backgroundColor: string
  readonly children: ReactNode
}

const edgeChars = (vertical: '╻' | '╹') => ({
  topLeft: '',
  bottomLeft: '',
  topRight: '',
  bottomRight: '',
  horizontal: ' ',
  vertical,
  topT: '',
  bottomT: '',
  leftT: '',
  rightT: '',
  cross: '',
})

const capChars = (horizontal: '▄' | '▀') => ({
  topLeft: '',
  bottomLeft: '',
  topRight: '',
  bottomRight: '',
  horizontal,
  vertical: ' ',
  topT: '',
  bottomT: '',
  leftT: '',
  rightT: '',
  cross: '',
})

export const UserMessageFrame = ({
  borderColor,
  backgroundColor,
  children,
}: UserMessageFrameProps) => (
  <>
    <box
      style={{
        height: 1,
        borderStyle: 'single',
        border: ['left'],
        borderColor,
        customBorderChars: edgeChars('╻'),
      }}
    >
      <box
        style={{
          height: 1,
          flexGrow: 1,
          borderStyle: 'single',
          border: ['top'],
          borderColor: backgroundColor,
          customBorderChars: capChars('▄'),
        }}
      />
    </box>
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
          paddingLeft: 1,
          paddingRight: 2,
          flexGrow: 1,
        }}
      >
        {children}
      </box>
    </box>
    <box
      style={{
        height: 1,
        borderStyle: 'single',
        border: ['left'],
        borderColor,
        customBorderChars: edgeChars('╹'),
      }}
    >
      <box
        style={{
          height: 1,
          flexGrow: 1,
          borderStyle: 'single',
          border: ['bottom'],
          borderColor: backgroundColor,
          customBorderChars: capChars('▀'),
        }}
      />
    </box>
  </>
)
