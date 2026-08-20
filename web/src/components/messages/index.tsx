/**
 * Message dispatcher — maps DisplayMessage.type to the correct component.
 *
 * Uses the union discriminant `type` field to route to the appropriate
 * message component. The accepted timeline owns grouping; renderer preferences
 * may omit entries before dispatch. Null branches remain defensive for entries
 * represented by dedicated timeline components.
 */
import { memo, type ReactNode } from "react"
import type { DisplayMessage, WorkSummaryMessage } from "@magnitudedev/sdk"

import { UserMessage } from "./user-message"
import { QueuedUserMessage } from "./queued-user-message"
import { AssistantMessage } from "./assistant-message"
import { ThinkingMessage } from "./thinking-message"
import { StatusIndicator } from "./status-indicator"
import { GoalStatus } from "./goal-status"
import { InterruptedMessage } from "./interrupted"
import { ErrorMessage } from "./error-message"
import { AgentCommunication } from "./agent-communication"
import { UserBashCommand } from "./user-bash-command"
import { WorkSummary } from "./work-summary"
import type { AssistantResponseFooter } from "../assistant-response-presentation"

export interface MessageDispatchProps {
  message: DisplayMessage
  isStreaming?: boolean
  isInterrupted?: boolean
  assistantWorkSummary?: WorkSummaryMessage | null
  assistantResponseFooter?: AssistantResponseFooter | null
}

/** Render a single (non-clustered) message by dispatching on its type */
function MessageDispatchImpl({
  message,
  isStreaming = false,
  isInterrupted = false,
  assistantWorkSummary = null,
  assistantResponseFooter = null,
}: MessageDispatchProps): ReactNode {
  switch (message.type) {
    case "user_message":
      return <UserMessage message={message} />
    case "queued_user_message":
      return <QueuedUserMessage message={message} />
    case "user_bash_command":
      return <UserBashCommand message={message} />
    case "assistant_message":
      return (
        <AssistantMessage
          message={message}
          isStreaming={isStreaming}
          isInterrupted={isInterrupted}
          responseFooter={assistantResponseFooter}
          workSummary={assistantWorkSummary}
        />
      )
    case "thinking":
      return <ThinkingMessage key={message.phase} message={message} />
    case "status_indicator":
      return <StatusIndicator message={message} />
    case "work_summary":
      return <WorkSummary message={message} />
    case "goal_status":
      return <GoalStatus message={message} />
    case "interrupted":
      return <InterruptedMessage message={message} />
    case "error":
      return <ErrorMessage message={message} />
    case "agent_communication":
      return <AgentCommunication message={message} />
    case "tool":
    case "worker_resumed":
    case "worker_finished":
    case "worker_killed":
    case "worker_user_killed":
    case "fork_result":
    case "fork_activity":
      return null
    default:
      return null
  }
}

export const MessageDispatch = memo(
  MessageDispatchImpl,
  (prev, next) =>
    prev.message === next.message &&
    prev.isStreaming === next.isStreaming &&
    prev.isInterrupted === next.isInterrupted &&
    prev.assistantWorkSummary === next.assistantWorkSummary &&
    prev.assistantResponseFooter === next.assistantResponseFooter
)
