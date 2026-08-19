/**
 * User message — spec §9.3.1
 *
 * Compact right-aligned bubble. Attachments row. Metadata with copy + timestamp.
 */
import { useState, type ReactNode } from "react"
import type {
  UserMessage as UserMessageType,
  DisplayAttachment,
} from "@magnitudedev/sdk"
import { CopyButton, Timestamp, AttachmentPill } from "./shared"
export function UserMessage({
  message,
}: {
  message: UserMessageType
}): ReactNode {
  const [hovered, setHovered] = useState(false)
  const hasContent = message.content.trim().length > 0
  const showMetadata = hovered || message.attachments.length > 0
  return (
    <div
      data-task-mode={message.taskMode}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="flex flex-col items-end"
    >
      <div className="flex w-[min(720px,72%)] flex-col items-end">
        {hasContent && (
          <div
            data-user-message-content=""
            className={`${
              message.taskMode
                ? "border-orange-700 dark:border-orange-400"
                : "border-slate-300 dark:border-slate-750"
            } max-w-full rounded-lg border bg-slate-100 px-[11px] py-2 transition-colors duration-100 dark:bg-slate-800`}
          >
            <div className="whitespace-pre-wrap font-sans text-[14px] leading-[1.55] text-slate-900 [word-break:break-word] dark:text-slate-200">
              {message.content}
            </div>
          </div>
        )}

        {message.attachments.length > 0 && (
          <div className={`${hasContent ? "mt-2.5" : ""} flex max-w-full flex-wrap justify-end gap-2`}>
            {message.attachments.map((attachment: DisplayAttachment, index: number) => (
              <AttachmentPill key={index} attachment={attachment} />
            ))}
          </div>
        )}

        <div
          className={`${
            showMetadata ? "opacity-100" : "opacity-0"
          } mt-2 flex min-h-[22px] items-center justify-end gap-2 px-0.5 transition-opacity duration-100`}
        >
          {hasContent && <CopyButton text={message.content} />}
          <Timestamp ts={message.timestamp} />
        </div>
      </div>
    </div>
  )
}
