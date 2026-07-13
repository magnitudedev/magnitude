import { memo } from 'react'
import { Option } from 'effect'
import type { ActionId } from '../../types/ui-actions'
import type { DisplayMessage } from '@magnitudedev/sdk'
import { UserMessage } from './messages/user-message'
import { QueuedUserMessage } from './messages/queued-user-message'
import { AssistantMessage } from './messages/assistant-message'
import { AgentCommunicationCard } from './messages/agent-communication-card'
import { ErrorMessage } from './messages/error-message'
import { useTheme } from '../../hooks/use-theme'
import { green, red, violet } from '../../utils/theme'
import { TextAttributes } from '@opentui/core'

interface MessageViewProps {
  message: DisplayMessage
  isStreaming: boolean
  isInterrupted?: boolean
  nextMessageInterrupted?: boolean
  onFileClick?: (path: string, section?: string) => void
  onForkExpand?: (forkId: string) => void
  onErrorAction?: (actionId: ActionId) => void
  mode?: 'default' | 'transcript'
}

const WorkerResumedRow = ({ message }: { message: Extract<DisplayMessage, { type: 'worker_resumed' }> }) => {
  const theme = useTheme()
  return (
    <box style={{ marginBottom: 1 }}>
      <text>
        <span style={{ fg: violet[300] }}>▶ </span>
        <span style={{ fg: theme.muted }}>Worker </span>
        <span style={{ fg: theme.foreground }}>{message.workerRole ? `${message.workerRole.charAt(0).toUpperCase()}${message.workerRole.slice(1)}` : message.workerId}</span>
        <span style={{ fg: theme.muted }}> resumed</span>
      </text>
    </box>
  )
}

const StatusIndicatorRow = ({ message }: { message: Extract<DisplayMessage, { type: 'status_indicator' }> }) => {
  const theme = useTheme()
  return (
    <box style={{ marginBottom: 1 }}>
      <text attributes={TextAttributes.DIM}>
        <span style={{ fg: theme.muted }}>{message.message}</span>
      </text>
    </box>
  )
}

const GoalStatusRow = ({ message }: { message: Extract<DisplayMessage, { type: 'goal_status' }> }) => {
  const theme = useTheme()
  const label = message.status === 'started' ? 'Goal started' : 'Goal finished'
  const detail = message.status === 'started'
    ? Option.getOrNull(message.objective)
    : Option.getOrNull(message.evidence)

  return (
    <box style={{ marginBottom: 1 }}>
      <text>
        <span style={{ fg: green[300] }}>{label}</span>
        {detail ? <span style={{ fg: theme.muted }}> · {detail}</span> : null}
      </text>
    </box>
  )
}

export const MessageView = memo(function MessageView({
  message,
  isStreaming,
  isInterrupted,
  nextMessageInterrupted,
  onFileClick,
  onForkExpand,
  onErrorAction,
  mode = 'default',
}: MessageViewProps) {
  const theme = useTheme()
  const isUserType = message.type === 'user_message' || message.type === 'queued_user_message'

  const content = (() => {
    switch (message.type) {
      case 'user_message':
        return <UserMessage content={message.content} timestamp={message.timestamp} taskMode={message.taskMode} attachments={message.attachments.filter((attachment: Extract<DisplayMessage, { type: 'user_message' }>['attachments'][number]) => attachment.type === 'image')} />

      case 'queued_user_message':
        return <QueuedUserMessage content={message.content} />

      case 'assistant_message':
        return (
          <AssistantMessage
            content={message.content}
            isStreaming={isStreaming}
            isInterrupted={isInterrupted}
            onFileClick={onFileClick}
          />
        )

      case 'thinking': {
        // Thinking is hidden in default mode
        if (mode === 'default') return null
        // In transcript mode, show thinking content (dim, italic)
        return (
          <box style={{ marginBottom: 1 }}>
            <text attributes={TextAttributes.ITALIC}>
              <span style={{ fg: theme.muted }}>{message.content}</span>
            </text>
          </box>
        )
      }

      case 'tool':
        // Projection emits tools as `tool_step`/`tool_summary` entries, not as
        // `message` entries, so this branch is unreachable. Defensive null.
        return null

      case 'status_indicator':
        return <StatusIndicatorRow message={message} />

      case 'goal_status':
        return <GoalStatusRow message={message} />

      case 'worker_resumed':
        return <WorkerResumedRow message={message} />

      case 'worker_finished':
      case 'worker_killed':
      case 'worker_user_killed':
        // These are data messages, not visually rendered inline
        return null

      case 'interrupted': {
        let interruptText: string
        if (message.context === 'fork') {
          interruptText = '■ Agent stopped'
        } else if (message.allKilled) {
          interruptText = '■ All agents interrupted. What would you like to do?'
        } else {
          interruptText = '■ Lead interrupted. What would you like to do?'
        }
        const noBottomGap = nextMessageInterrupted
        return (
          <box style={{ marginBottom: noBottomGap ? 0 : 1 }}>
            <text style={{ fg: red[400] }}>{interruptText}</text>
          </box>
        )
      }

      case 'error':
        return <ErrorMessage message={message.message} timestamp={message.timestamp} cta={Option.getOrUndefined(message.cta)} onAction={(actionId) => onErrorAction?.(actionId as ActionId)} />

      case 'agent_communication':
        return (
          <box style={{ marginBottom: 1 }}>
            <AgentCommunicationCard message={message} onFileClick={onFileClick} />
          </box>
        )

    }
  })()

  if (content === null) return null

  if (isUserType) {
    return content
  }

  return <box style={{ paddingLeft: 1 }}>{content}</box>
})
