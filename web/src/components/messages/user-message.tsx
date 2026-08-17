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
  const showMetadata = hovered || message.attachments.length > 0
  return (
    <div
      data-task-mode={message.taskMode}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="flex flex-col items-end"
    >
      <div
        className={`${
          message.taskMode
            ? "border-orange-700 dark:border-orange-400"
            : "border-slate-300 dark:border-slate-750"
        } max-w-[min(720px,72%)] rounded-lg border bg-slate-100 px-[11px] py-2 transition-colors duration-100 dark:bg-slate-800`}
      >
        <div className="font-sans text-[14px] text-slate-900 dark:text-slate-200 leading-[1.55] whitespace-pre-wrap [word-break:break-word]">
          {message.content}
        </div>
      </div>
      <div
        className={`${
          showMetadata ? "opacity-[1]" : "opacity-[0]"
        }  flex items-center justify-between [gap:12px] [width:min(720px,_72%)] [min-height:22px] [margin-top:3px] [padding:0_2px] [transition:opacity_100ms_ease]`}
      >
        <div className="flex items-center [gap:4px] flex-wrap min-w-0">
          {message.attachments.map((a: DisplayAttachment, i: number) => (
            <AttachmentPill key={i} attachment={a} />
          ))}
        </div>
        <div className="flex items-center [gap:8px] shrink-0">
          <CopyButton text={message.content} />
          <Timestamp ts={message.timestamp} />
        </div>
      </div>
    </div>
  )
}
