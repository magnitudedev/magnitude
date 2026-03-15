import { type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { usePasteHandler } from '../hooks/use-paste-handler'
import { useTheme } from '../hooks/use-theme'
import { readClipboardText } from '../utils/clipboard'
import { InputCursor } from './multiline-input'

interface SingleLineInputProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  focused?: boolean
  masked?: boolean
}

function clampCursor(position: number, length: number): number {
  return Math.max(0, Math.min(length, position))
}

const CONTROL_CHAR_REGEX = /[\x00-\x1f\x7f]/

export function SingleLineInput({
  value,
  onChange,
  placeholder,
  focused = true,
  masked = false,
}: SingleLineInputProps) {
  const theme = useTheme()
  const [cursorPosition, setCursorPosition] = useState(value.length)

  const previousValueRef = useRef(value)
  const internalEditRef = useRef(false)

  useEffect(() => {
    const prev = previousValueRef.current
    const changedExternally = !internalEditRef.current

    if (changedExternally && value.length !== prev.length) {
      setCursorPosition(value.length)
    } else {
      setCursorPosition((prevCursor) => clampCursor(prevCursor, value.length))
    }

    internalEditRef.current = false
    previousValueRef.current = value
  }, [value])

  const applyChange = useCallback(
    (nextValue: string, nextCursor: number) => {
      internalEditRef.current = true
      onChange(nextValue)
      setCursorPosition(clampCursor(nextCursor, nextValue.length))
    },
    [onChange],
  )

  const insertAtCursor = useCallback(
    (text: string) => {
      if (!text) return
      const clamped = clampCursor(cursorPosition, value.length)
      const nextValue = value.slice(0, clamped) + text + value.slice(clamped)
      applyChange(nextValue, clamped + text.length)
    },
    [applyChange, cursorPosition, value],
  )

  const handlePasteText = useCallback(
    (eventText?: string) => {
      const text = eventText ?? readClipboardText()
      if (text) insertAtCursor(text)
    },
    [insertAtCursor],
  )

  const { handlePasteKey, handlePasteEvent } = usePasteHandler({
    enabled: focused,
    onPaste: handlePasteText,
  })

  useKeyboard(
    useCallback(
      (key: KeyEvent) => {
        if (!focused) return

        const clamped = clampCursor(cursorPosition, value.length)

        if (key.name === 'left' && !key.ctrl && !key.meta && !key.option) {
          key.preventDefault?.()
          setCursorPosition(clampCursor(clamped - 1, value.length))
          return
        }

        if (key.name === 'right' && !key.ctrl && !key.meta && !key.option) {
          key.preventDefault?.()
          setCursorPosition(clampCursor(clamped + 1, value.length))
          return
        }

        if (
          key.name === 'home' ||
          (key.ctrl && !key.meta && !key.option && (key.name ?? '').toLowerCase() === 'a')
        ) {
          key.preventDefault?.()
          setCursorPosition(0)
          return
        }

        if (
          key.name === 'end' ||
          (key.ctrl && !key.meta && !key.option && (key.name ?? '').toLowerCase() === 'e')
        ) {
          key.preventDefault?.()
          setCursorPosition(value.length)
          return
        }

        if (key.name === 'backspace' && !key.ctrl && !key.meta && !key.option) {
          key.preventDefault?.()
          if (clamped <= 0) return
          const nextValue = value.slice(0, clamped - 1) + value.slice(clamped)
          applyChange(nextValue, clamped - 1)
          return
        }

        if (key.name === 'delete' && !key.ctrl && !key.meta && !key.option) {
          key.preventDefault?.()
          if (clamped >= value.length) return
          const nextValue = value.slice(0, clamped) + value.slice(clamped + 1)
          applyChange(nextValue, clamped)
          return
        }

        if (handlePasteKey(key)) return

        if (
          key.sequence &&
          key.sequence.length >= 1 &&
          !key.ctrl &&
          !key.meta &&
          !key.option &&
          !CONTROL_CHAR_REGEX.test(key.sequence)
        ) {
          key.preventDefault?.()
          insertAtCursor(key.sequence)
        }
      },
      [focused, cursorPosition, value, applyChange, insertAtCursor, handlePasteKey],
    ),
  )

  const isPlaceholder = value.length === 0 && Boolean(placeholder)
  const displayValue = isPlaceholder
    ? placeholder ?? ''
    : masked && value.length > 0
      ? '•'.repeat(value.length)
      : value

  const beforeCursor = displayValue.slice(0, cursorPosition)
  const activeChar = displayValue[cursorPosition]
  const afterCursor = displayValue.slice(cursorPosition + 1)
  const isBlockCursor = activeChar !== undefined && activeChar !== ' '

  return (
    <box
      focusable={focused}
      focused={focused}
      onPaste={handlePasteEvent}
    >
      <text style={{ fg: isPlaceholder ? theme.muted : theme.foreground }}>
        {isPlaceholder ? (
          <>
            <InputCursor
              visible={true}
              focused={focused}
              char={'▍'}
              color={theme.info}
              activeChar={' '}
            />
            {displayValue}
          </>
        ) : (
          <>
            {beforeCursor}
            <InputCursor
              visible={true}
              focused={focused}
              char={isBlockCursor ? activeChar : '▍'}
              color={isBlockCursor ? undefined : theme.info}
              backgroundColor={isBlockCursor ? theme.info : undefined}
              activeChar={activeChar}
            />
            {afterCursor}
          </>
        )}
      </text>
    </box>
  )
}