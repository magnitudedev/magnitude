/**
 * Message dispatcher — maps DisplayMessage.type to the correct component.
 *
 * Uses the union discriminant `type` field to route to the appropriate
 * message component. Projection owns visibility/grouping; the null branches
 * below are defensive — projection never emits these as message entries.
 */
import { memo, type ReactNode } from "react"
import type { DisplayMessage } from "@magnitudedev/sdk"

import { UserMessage } from "./user-message"
import { QueuedUserMessage } from "./queued-user-message"
import { AssistantMessage } from "./assistant-message"
import { ThinkingMessage } from "./thinking-message"
import { StatusIndicator } from "./status-indicator"
import { GoalStatus } from "./goal-status"
import { InterruptedMessage } from "./interrupted"
import { ErrorMessage } from "./error-message"
import { AgentCommunication } from "./agent-communication"

export interface MessageDispatchProps {
  message: DisplayMessage
  isStreaming?: boolean
  isInterrupted?: boolean
  mode?: "default" | "transcript"
}

/** Render a single (non-clustered) message by dispatching on its type */
function MessageDispatchImpl({
  message,
  isStreaming = false,
  isInterrupted = false,
  mode = "default",
}: MessageDispatchProps): ReactNode {
  switch (message.type) {
    case "user_message":
      return <UserMessage message={message} />
    case "queued_user_message":
      return <QueuedUserMessage message={message} />
    case "assistant_message":
      return (
        <AssistantMessage
          message={message}
          isStreaming={isStreaming}
          isInterrupted={isInterrupted}
        />
      )
    case "thinking":
      return <ThinkingMessage message={message} mode={mode} />
    case "status_indicator":
      return <StatusIndicator message={message} />
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
    prev.mode === next.mode,
)
