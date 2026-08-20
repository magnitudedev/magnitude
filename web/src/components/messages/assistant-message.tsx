/**
 * Assistant message — spec §9.3.3
 *
 * Renders markdown content. Streaming cursor when active.
 * Interrupted state shows a divider below the partial response.
 */
import { type ReactNode } from "react"
import type {
  AssistantMessage as AssistantMessageType,
  WorkSummaryMessage,
} from "@magnitudedev/sdk"
import { stripTrailingLineBreaks } from "@magnitudedev/client-common"
import { MarkdownContent } from "../markdown-content"
import { InterruptedDivider } from "./interrupted"
import { CopyButton, RelativeTimestamp } from "./shared"
import { WorkSummary } from "./work-summary"
import type { AssistantResponseFooter } from "../assistant-response-presentation"
export interface AssistantMessageProps {
  message: AssistantMessageType
  isStreaming?: boolean
  isInterrupted?: boolean
  responseFooter?: AssistantResponseFooter | null
  workSummary?: WorkSummaryMessage | null
}
export function AssistantMessage({
  message,
  isStreaming = false,
  isInterrupted = false,
  responseFooter = null,
  workSummary = null,
}: AssistantMessageProps): ReactNode {
  return (
    <div className="group/assistant max-w-[min(860px,100%)] py-0.5">
      <MarkdownContent
        content={stripTrailingLineBreaks(message.content)}
        isStreaming={isStreaming}
        showCursor={isStreaming && !isInterrupted}
      />
      {isInterrupted && <InterruptedDivider />}
      {workSummary !== null && (
        <div className="mt-2">
          <WorkSummary message={workSummary} />
        </div>
      )}
      {responseFooter !== null && (
        <div
          data-assistant-metadata=""
          className={`${
            responseFooter.isLatest
              ? "opacity-100"
              : "opacity-0 group-hover/assistant:opacity-100 group-focus-within/assistant:opacity-100"
          } mt-1.5 flex min-h-5 items-center gap-2 transition-opacity duration-100`}
        >
          <CopyButton text={responseFooter.content} label="Copy response" iconOnly />
          <RelativeTimestamp ts={responseFooter.timestamp} />
        </div>
      )}
    </div>
  )
}
