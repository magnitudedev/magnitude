/**
 * Assistant message — spec §9.3.3
 *
 * Renders markdown content. Streaming cursor when active.
 * Interrupted state shows a divider below the partial response.
 */
import { type ReactNode } from "react"
import type { AssistantMessage as AssistantMessageType } from "@magnitudedev/sdk"
import { stripTrailingLineBreaks } from "@magnitudedev/client-common"
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
    <div className="[padding-left:12px] [padding-top:2px] [padding-bottom:2px] [max-width:min(860px,_100%)]">
      <MarkdownContent
        content={stripTrailingLineBreaks(message.content)}
        isStreaming={isStreaming}
        showCursor={isStreaming && !isInterrupted}
      />
      {isInterrupted && <InterruptedDivider />}
    </div>
  )
}
