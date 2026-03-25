import { memo, useState } from 'react'
import { useTheme } from '../hooks/use-theme'
import { BOX_CHARS } from '../utils/ui-constants'

import { Button } from './button'
import { DiffView } from './diff-view'
import type { ApprovalRequestMessage } from '@magnitudedev/agent'

interface ApprovalRequestProps {
  message: ApprovalRequestMessage
  onApprove?: () => void
  onReject?: () => void
}

function getLineCount(input: unknown): number | null {
  if (input && typeof input === 'object' && 'content' in input) {
    return (input as { content: string }).content.split('\n').length
  }
  return null
}

export const ApprovalRequest = memo(function ApprovalRequest({ message, onApprove, onReject }: ApprovalRequestProps) {
  const theme = useTheme()
  const [approveHovered, setApproveHovered] = useState(false)
  const [rejectHovered, setRejectHovered] = useState(false)

  // Approved state — tool shows up in think block, no need for separate message
  if (message.status === 'approved') {
    return null
  }

  // Build mode transition — custom UI with "Build Now" / "Keep Planning"
  if (message.toolKey === 'requestBuild') {
    if (message.status === 'rejected') {
      return (
        <box style={{
          marginBottom: 1,
          paddingLeft: 1,
          paddingRight: 1,
          flexDirection: 'column',
          borderStyle: 'single',
          borderColor: theme.border,
          customBorderChars: BOX_CHARS,
        }}>
          <text>
            <span style={{ fg: theme.muted }}>{'↩ '}</span>
            <span style={{ fg: theme.foreground }}>{'Keep Planning'}</span>
          </text>
        </box>
      )
    }

    // Pending state
    return (
      <box style={{
        flexDirection: 'column',
        marginBottom: 1,
        borderStyle: 'single',
        borderColor: theme.border,
        customBorderChars: BOX_CHARS,
        paddingLeft: 1,
        paddingRight: 1,
      }}>
        <text>
          <span style={{ fg: theme.success }}>{'▶ '}</span>
          <span style={{ fg: theme.foreground }}>{'Begin Build Mode'}</span>
        </text>
        <text style={{ fg: theme.muted, marginTop: 0 }}>{message.reason}</text>
        <box style={{ flexDirection: 'row', gap: 1, marginTop: 1 }}>
          <Button
            onClick={onApprove}
            onMouseOver={() => setApproveHovered(true)}
            onMouseOut={() => setApproveHovered(false)}
            style={{
              borderStyle: 'single',
              borderColor: approveHovered ? theme.success : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}
          >
            <text style={{ fg: theme.success }}>Build Now (A)</text>
          </Button>
          <Button
            onClick={onReject}
            onMouseOver={() => setRejectHovered(true)}
            onMouseOut={() => setRejectHovered(false)}
            style={{
              borderStyle: 'single',
              borderColor: rejectHovered ? theme.muted : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}
          >
            <text style={{ fg: theme.muted }}>Keep Planning (D)</text>
          </Button>
        </box>
      </box>
    )
  }

  // Rejected state — two-line card with no border
  if (message.status === 'rejected') {
    return (
      <box style={{
        marginBottom: 1,
        paddingLeft: 1,
        paddingRight: 1,
        flexDirection: 'column',
        borderStyle: 'single',
        borderColor: theme.border,
        customBorderChars: BOX_CHARS,
      }}>
        <text>
          <span style={{ fg: theme.error }}>{'✗ '}</span>
          <span style={{ fg: theme.foreground }}>{'User Rejected '}</span>
          {renderRejectedToolInfo(message.toolKey, message.input, theme)}
        </text>
        <text style={{ fg: theme.muted }}>What would you like to do instead?</text>
      </box>
    )
  }

  // Pending state — approval card with neutral border
  const lineCount = getLineCount(message.input)
  const writeContent = getWriteContent(message)
  const writeDiffs = writeContent
    ? [{ startLine: 1, removedLines: [] as string[], addedLines: writeContent.split('\n'), contextBefore: [] as string[], contextAfter: [] as string[] }]
    : null
  const hasContentPreview = !!writeDiffs

  return (
    <box style={{
      flexDirection: 'column',
      marginBottom: 1,
      borderStyle: 'single',
      borderColor: theme.border,
      customBorderChars: BOX_CHARS,
      paddingLeft: 1,
      paddingRight: 1,
    }}>
      {renderPendingToolInfo(message.toolKey, message.input, theme, lineCount)}
      {writeDiffs && (
        <box style={{ flexDirection: 'column', marginTop: 1 }}>
          <DiffView diffs={writeDiffs} />
        </box>
      )}
      <text style={{ fg: theme.muted, marginTop: hasContentPreview ? 1 : 0 }}>{message.reason}</text>
      <box style={{ flexDirection: 'row', gap: 1, marginTop: 1 }}>
        <Button
          onClick={onApprove}
          onMouseOver={() => setApproveHovered(true)}
          onMouseOut={() => setApproveHovered(false)}
          style={{
            borderStyle: 'single',
            borderColor: approveHovered ? theme.success : theme.border,
            customBorderChars: BOX_CHARS,
            paddingLeft: 1,
            paddingRight: 1,
          }}
        >
          <text style={{ fg: theme.success }}>Approve (A)</text>
        </Button>
        <Button
          onClick={onReject}
          onMouseOver={() => setRejectHovered(true)}
          onMouseOut={() => setRejectHovered(false)}
          style={{
            borderStyle: 'single',
            borderColor: rejectHovered ? theme.error : theme.border,
            customBorderChars: BOX_CHARS,
            paddingLeft: 1,
            paddingRight: 1,
          }}
        >
          <text style={{ fg: theme.error }}>Deny (D)</text>
        </Button>
      </box>
    </box>
  )
})

function getWriteContent(message: ApprovalRequestMessage): string | null {
  if (message.toolKey === 'fileWrite' && message.input && typeof message.input === 'object' && 'content' in message.input) {
    return (message.input as { content: string }).content
  }
  return null
}

function renderPendingToolInfo(toolKey: string, input: unknown, theme: ReturnType<typeof useTheme>, lineCount: number | null) {
  if (toolKey === 'shell' && input && typeof input === 'object' && 'command' in input) {
    const cmd = (input as { command: string }).command
    const shortCmd = cmd.length > 80 ? cmd.slice(0, 77) + '...' : cmd
    return (
      <text>
        <span style={{ fg: theme.muted }}>{'$ '}</span>
        <span style={{ fg: theme.foreground }}>{shortCmd}</span>
      </text>
    )
  }

  if (toolKey === 'fileWrite' && input && typeof input === 'object' && 'path' in input) {
    const path = (input as { path: string }).path
    const lineInfo = lineCount !== null ? ` · ${lineCount} ${lineCount === 1 ? 'line' : 'lines'}` : ''
    return (
      <text>
        <span style={{ fg: theme.info }}>{'✎ '}</span>
        <span style={{ fg: theme.foreground }}>Write </span>
        <span style={{ fg: theme.muted }}>{path}</span>
        {lineInfo && <span style={{ fg: theme.info }}>{lineInfo}</span>}
      </text>
    )
  }

  if (toolKey === 'fileEdit' && input && typeof input === 'object' && 'path' in input) {
    const path = (input as { path: string }).path
    return (
      <text>
        <span style={{ fg: theme.info }}>{'✎ '}</span>
        <span style={{ fg: theme.foreground }}>Edit </span>
        <span style={{ fg: theme.muted }}>{path}</span>

      </text>
    )
  }

  return <text style={{ fg: theme.foreground }}>{toolKey}</text>
}

function renderRejectedToolInfo(toolKey: string, input: unknown, theme: ReturnType<typeof useTheme>) {
  if (toolKey === 'shell' && input && typeof input === 'object' && 'command' in input) {
    const cmd = (input as { command: string }).command
    const shortCmd = cmd.length > 80 ? cmd.slice(0, 77) + '...' : cmd
    return (
      <>
        <span style={{ fg: theme.muted }}>{'$ '}</span>
        <span style={{ fg: theme.foreground }}>{shortCmd}</span>
      </>
    )
  }

  if (toolKey === 'fileWrite' && input && typeof input === 'object' && 'path' in input) {
    const path = (input as { path: string }).path
    return (
      <>
        <span style={{ fg: theme.foreground }}>{'Write '}</span>
        <span style={{ fg: theme.muted }}>{path}</span>
      </>
    )
  }

  if (toolKey === 'fileEdit' && input && typeof input === 'object' && 'path' in input) {
    const path = (input as { path: string }).path
    return (
      <>
        <span style={{ fg: theme.foreground }}>{'Edit '}</span>
        <span style={{ fg: theme.muted }}>{path}</span>
      </>
    )
  }

  return <span style={{ fg: theme.muted }}>{toolKey}</span>
}
