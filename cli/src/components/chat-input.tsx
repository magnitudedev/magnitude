import { useKeyboard } from '@opentui/react'
import { useState, useCallback, memo, useEffect, useRef } from 'react'
import { InputCursor } from './multiline-input'
import { BOX_CHARS } from '../utils/ui-constants'
import { useTheme } from '../hooks/use-theme'
import type { InputValue } from '../utils/strings'
import { decodePasteBytes } from '@opentui/core'
import type { ScrollBoxRenderable } from '@opentui/core'

interface ChatInputProps {
  value?: string
  cursorPosition?: number
  onChange?: (value: InputValue) => void
  onSubmit: (message: string) => void
  onPaste: (text?: string) => void
  disabled?: boolean
  placeholder?: string
}

export const ChatInput = memo(function ChatInput({
  value: controlledValue,
  cursorPosition: controlledCursor,
  onChange,
  onSubmit,
  onPaste,
  disabled = false,
  placeholder = 'Type a message...'
}: ChatInputProps) {
  const theme = useTheme()
  const [internalValue, setInternalValue] = useState('')
  const [internalCursor, setInternalCursor] = useState(0)
  const [lastActivity, setLastActivity] = useState(Date.now())
  const scrollBoxRef = useRef<ScrollBoxRenderable | null>(null)

  const value = controlledValue !== undefined ? controlledValue : internalValue
  const cursorPosition = controlledCursor !== undefined ? controlledCursor : internalCursor

  useEffect(() => {
    setLastActivity(Date.now())
  }, [value, cursorPosition])

  const updateValue = useCallback((newValue: string, newCursor: number) => {
    if (onChange) {
      onChange({
        text: newValue,
        cursorPosition: newCursor,
        lastEditDueToNav: false,
        pasteSegments: [],
        selectedPasteSegmentId: null,
      mentionSegments: [],
      selectedMentionSegmentId: null,
      })
    } else {
      setInternalValue(newValue)
      setInternalCursor(newCursor)
    }
  }, [onChange])

  const insertTextAtCursor = useCallback((text: string) => {
    const newValue = value.slice(0, cursorPosition) + text + value.slice(cursorPosition)
    const newCursor = cursorPosition + text.length
    updateValue(newValue, newCursor)
  }, [value, cursorPosition, updateValue])

  useKeyboard(
    useCallback(
      (key) => {
        if (disabled) return

        // Shift+Enter inserts newline
        if ((key.name === 'return' || key.name === 'enter') && key.shift) {
          insertTextAtCursor('\n')
          return
        }

        // Plain Enter submits
        if (key.name === 'return' || key.name === 'enter') {
          if (value.trim()) {
            onSubmit(value.trim())
            updateValue('', 0)
          }
          return
        }

        // Backspace
        if (key.name === 'backspace') {
          if (cursorPosition > 0) {
            const newValue = value.slice(0, cursorPosition - 1) + value.slice(cursorPosition)
            updateValue(newValue, cursorPosition - 1)
          }
          return
        }

        // Delete
        if (key.name === 'delete') {
          if (cursorPosition < value.length) {
            const newValue = value.slice(0, cursorPosition) + value.slice(cursorPosition + 1)
            updateValue(newValue, cursorPosition)
          }
          return
        }

        // Left arrow
        if (key.name === 'left') {
          const newCursor = Math.max(0, cursorPosition - 1)
          if (onChange) {
            onChange({
              text: value,
              cursorPosition: newCursor,
              lastEditDueToNav: true,
              pasteSegments: [],
              selectedPasteSegmentId: null,
      mentionSegments: [],
      selectedMentionSegmentId: null,
            })
          } else {
            setInternalCursor(newCursor)
          }
          return
        }

        // Right arrow
        if (key.name === 'right') {
          const newCursor = Math.min(value.length, cursorPosition + 1)
          if (onChange) {
            onChange({
              text: value,
              cursorPosition: newCursor,
              lastEditDueToNav: true,
              pasteSegments: [],
              selectedPasteSegmentId: null,
      mentionSegments: [],
      selectedMentionSegmentId: null,
            })
          } else {
            setInternalCursor(newCursor)
          }
          return
        }

        // Home / Ctrl+A
        if (key.name === 'home' || (key.ctrl && key.name === 'a')) {
          if (onChange) {
            onChange({
              text: value,
              cursorPosition: 0,
              lastEditDueToNav: true,
              pasteSegments: [],
              selectedPasteSegmentId: null,
      mentionSegments: [],
      selectedMentionSegmentId: null,
            })
          } else {
            setInternalCursor(0)
          }
          return
        }

        // End / Ctrl+E
        if (key.name === 'end' || (key.ctrl && key.name === 'e')) {
          if (onChange) {
            onChange({
              text: value,
              cursorPosition: value.length,
              lastEditDueToNav: true,
              pasteSegments: [],
              selectedPasteSegmentId: null,
      mentionSegments: [],
      selectedMentionSegmentId: null,
            })
          } else {
            setInternalCursor(value.length)
          }
          return
        }

        // Clear line with Ctrl+U
        if (key.ctrl && key.name === 'u') {
          updateValue('', 0)
          return
        }

        // Paste with Ctrl+V (Cmd+V triggers bracketed paste via onPaste event)
        if (key.ctrl && key.name === 'v') {
          // Call onPaste with no argument - it will read from clipboard
          onPaste()
          return
        }

        // Regular character input
        if (
          key.sequence &&
          key.sequence.length >= 1 &&
          !key.ctrl &&
          !key.meta &&
          key.sequence.charCodeAt(0) >= 32
        ) {
          insertTextAtCursor(key.sequence)
        }
      },
      [value, cursorPosition, onSubmit, onPaste, disabled, insertTextAtCursor, updateValue, onChange]
    )
  )

  const isPlaceholder = value.length === 0
  const displayValue = isPlaceholder ? placeholder : value
  const beforeCursor = displayValue.slice(0, cursorPosition)
  const afterCursor = displayValue.slice(cursorPosition)
  const borderColor = disabled ? theme.border : theme.primary
  const textColor = disabled ? theme.muted : isPlaceholder ? theme.muted : undefined

  return (
    <box
      style={{
        width: '100%',
        borderStyle: 'single',
        borderColor,
        customBorderChars: BOX_CHARS,
        paddingLeft: 0,
        paddingRight: 0,
        flexDirection: 'row',
        alignItems: 'center',
        flexShrink: 0,
      }}
    >
      <scrollbox
        ref={scrollBoxRef}
        scrollX={false}
        stickyScroll={true}
        stickyStart="bottom"
        scrollbarOptions={{ visible: false }}
        verticalScrollbarOptions={{ visible: false }}
        onPaste={(event) => onPaste(decodePasteBytes(event.bytes))}
        style={{
          flexGrow: 1,
          flexShrink: 0,
          rootOptions: {
            width: '100%',
            height: 1,
            backgroundColor: 'transparent',
            flexGrow: 1,
            flexShrink: 0,
          },
          wrapperOptions: {
            paddingLeft: 1,
            paddingRight: 1,
            border: false,
          },
          contentOptions: {
            justifyContent: 'flex-start',
          },
        }}
      >
        <text style={{ fg: textColor, wrapMode: 'none' }}>
          {!disabled ? (
            <>
              {beforeCursor}
              <InputCursor
                visible={true}
                focused={!disabled}
                shouldBlink={true}
                color={theme.primary}
                key={lastActivity}
              />
              {afterCursor}
            </>
          ) : (
            displayValue
          )}
        </text>
      </scrollbox>
    </box>
  )
})
