/**
 * Assistant message — spec §9.3.3
 *
 * Renders markdown content. Streaming cursor when active.
 * Interrupted state shows a divider below the partial response.
 */
import { type ReactNode } from "react"
import type { AssistantMessage as AssistantMessageType } from "@magnitudedev/sdk"
import { MarkdownContent } from "../markdown-content"
import { InterruptedDivider } from "./interrupted"

export interface AssistantMessageProps {
  message: AssistantMessageType
  isStreaming?: boolean
  isInterrupted?: boolean
}

export function AssistantMessage({
  message,
  isStreaming = false,
  isInterrupted = false,
}: AssistantMessageProps): ReactNode {
  return (
    <div style={{ paddingLeft: "12px", paddingTop: "2px", paddingBottom: "2px", maxWidth: "min(860px, 100%)" }}>
      <MarkdownContent
        content={message.content}
        isStreaming={isStreaming}
        showCursor={isStreaming && !isInterrupted}
      />
      {isInterrupted && <InterruptedDivider />}
    </div>
  )
}
