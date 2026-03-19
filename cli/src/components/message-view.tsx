import { memo } from 'react'
import type { DisplayMessage } from '@magnitudedev/agent'
import { UserMessage } from './user-message'
import { QueuedUserMessage } from './queued-user-message'
import { AssistantMessage } from './assistant-message'
import { ThinkBlock } from './think-block'
import { InlineForkActivity } from './inline-fork-activity'
import { ApprovalRequest } from './approval-request'
import { AgentCommunicationCard } from './agent-communication-card'
import { ErrorMessage } from './error-message'
import { useTheme } from '../hooks/use-theme'

interface MessageViewProps {
  message: DisplayMessage
  isStreaming: boolean
  isInterrupted?: boolean
  isCollapsed?: boolean
  onToggleCollapse?: () => void
  hideThinkBlockHeader?: boolean
  onThinkBlockHeaderRef?: (ref: any) => void
  pendingApproval?: boolean
  onApprove?: () => void
  onReject?: () => void
  onWorkApprove?: () => void
  onWorkReject?: () => void
  inputHasText?: boolean
  onArtifactClick?: (name: string, section?: string) => void
  onForkExpand?: (forkId: string) => void
}

export const MessageView = memo(function MessageView({
  message,
  isStreaming,
  isInterrupted,
  isCollapsed,
  onToggleCollapse,
  hideThinkBlockHeader,
  onThinkBlockHeaderRef,
  pendingApproval,
  onApprove,
  onReject,
  onWorkApprove,
  onWorkReject,
  inputHasText,
  onArtifactClick,
  onForkExpand
}: MessageViewProps) {
  const theme = useTheme()
  // User messages have their own border structure providing left offset
  const isUserType = message.type === 'user_message' || message.type === 'queued_user_message'

  const content = (() => {
    switch (message.type) {
      case 'user_message':
        return <UserMessage content={message.content} timestamp={message.timestamp} taskMode={message.taskMode} attachments={message.attachments} />

      case 'queued_user_message':
        return <QueuedUserMessage content={message.content} />

      case 'assistant_message':
        return (
          <AssistantMessage
            content={message.content}
            isStreaming={isStreaming}
            isInterrupted={isInterrupted}
            onOpenArtifact={onArtifactClick}
          />
        )

      case 'think_block':
        return (
          <ThinkBlock
            block={message}
            isCollapsed={isCollapsed ?? false}
            onToggle={onToggleCollapse ?? (() => {})}
            timerStartTime={message.status === 'active' ? message.timestamp : null}
            hideHeader={hideThinkBlockHeader}
            onHeaderRef={onThinkBlockHeaderRef}
            pendingApproval={pendingApproval}
            onArtifactClick={onArtifactClick}
            isInterrupted={isInterrupted}
          />
        )

      case 'interrupted': {
        let interruptText: string
        if (message.context === 'fork') {
          interruptText = '[Stopped] · Agent was stopped by user'
        } else if (message.allKilled) {
          interruptText = '[Interrupted] · All agents were stopped. What would you like to do?'
        } else {
          interruptText = '[Interrupted] · What would you like to do instead?'
        }
        return (
          <box style={{ marginBottom: 1 }}>
            <text style={{ fg: theme.warning }}>{interruptText}</text>
          </box>
        )
      }

      case 'unexpected_error':
        return <ErrorMessage tag={message.tag} message={message.message} timestamp={message.timestamp} />

      case 'fork_activity':
        return (
          <InlineForkActivity
            message={message}
            onExpand={onForkExpand ?? (() => {})}
            onArtifactClick={onArtifactClick}
          />
        )

      case 'approval_request':
        return <ApprovalRequest message={message} onApprove={onApprove} onReject={onReject} />

      case 'agent_communication':
        return (
          <box style={{ marginBottom: 1 }}>
            <AgentCommunicationCard message={message} />
          </box>
        )

    }
  })()

  if (isUserType) {
    return content
  }

  return <box style={{ paddingLeft: 1 }}>{content}</box>
})
