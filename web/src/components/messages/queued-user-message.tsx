/**
 * Queued user message — spec §9.3.2
 *
 * Identical to user message but dimmed text, Clock icon + "Queued".
 */
import { useState, type ReactNode } from "react"
import type {
  QueuedUserMessage as QueuedUserMessageType,
  DisplayAttachment,
} from "@magnitudedev/sdk"
import { CopyButton, QueuedIndicator, AttachmentPill } from "./shared"
export function QueuedUserMessage({
  message,
}: {
  message: QueuedUserMessageType
}): ReactNode {
  const [hovered, setHovered] = useState(false)
  const showMetadata = hovered || message.attachments.length > 0
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="flex flex-col items-end"
    >
      <div className="[max-width:min(720px,_72%)] bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-750 rounded-[8px] [padding:8px_11px] opacity-[0.7]">
        <div className="font-sans text-[14px] text-slate-500 leading-[1.55] whitespace-pre-wrap [word-break:break-word]">
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
          <QueuedIndicator />
        </div>
      </div>
    </div>
  )
}
