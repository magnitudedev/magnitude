import { Button } from "@/components/ui/button"

/**
 * Shared utilities for message components — copy button, timestamp, attachment pill.
 */
import { useState, useSyncExternalStore, type ReactNode } from "react"
import {
  Copy,
  Check,
  Clock,
  FileText,
  Folder,
  Image as ImageIcon,
} from "lucide-react"
import type { DisplayAttachment } from "@magnitudedev/sdk"
import {
  formatShortTimestamp,
  getTickSnapshot,
  subscribeTick,
} from "@magnitudedev/client-common"
import { formatMessageRelativeTime } from "@/lib/message-relative-time"

/** Copy button with icon-swap feedback */
export function CopyButton({
  text,
  label = "Copy",
  iconOnly = false,
}: {
  text: string
  label?: string
  iconOnly?: boolean
}): ReactNode {
  const [copied, setCopied] = useState(false)
  return (
    <Button variant="unstyled" size="unstyled"
      onClick={() => {
        navigator.clipboard?.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      }}
      aria-label={label || "Copy to clipboard"}
      className="text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 data-[copied=true]:text-green-700 data-[copied=true]:hover:text-green-700 dark:data-[copied=true]:text-green-500 dark:data-[copied=true]:hover:text-green-500 [background:transparent] border-0 cursor-pointer flex items-center [gap:4px] text-[13px] font-sans [padding:0px]"
      data-copied={copied ? "true" : "false"}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
      {!iconOnly && label}
    </Button>
  )
}

/** Timestamp display */
export function Timestamp({ ts }: { ts: number }): ReactNode {
  return (
    <span className="font-sans text-[13px] text-slate-500">
      {formatShortTimestamp(ts)}
    </span>
  )
}

/** Live relative timestamp for conversational message metadata. */
export function RelativeTimestamp({ ts }: { ts: number }): ReactNode {
  useSyncExternalStore(subscribeTick, getTickSnapshot, getTickSnapshot)
  return (
    <span className="whitespace-nowrap font-sans text-[12px] text-slate-500">
      {formatMessageRelativeTime(ts, Date.now())}
    </span>
  )
}

/** Queued indicator with clock icon */
export function QueuedIndicator(): ReactNode {
  return (
    <span className="flex items-center [gap:4px] text-slate-500 font-sans text-[13px]">
      <Clock size={14} />
      Queued
    </span>
  )
}

/** Attachment pill for user messages */
export function AttachmentPill({
  attachment,
}: {
  attachment: DisplayAttachment
}): ReactNode {
  if (attachment.type === "image") {
    return (
      <div className="flex h-13 min-w-48 max-w-64 items-center gap-2.5 rounded-lg border border-slate-300 bg-white px-2.5 dark:border-slate-750 dark:bg-slate-850">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-slate-150 text-blue-600 dark:bg-slate-750 dark:text-blue-400">
          <ImageIcon size={17} />
        </span>
        <span className="min-w-0 font-sans">
          <span className="block truncate text-[12px] font-medium leading-4 text-slate-800 dark:text-slate-200">
            {attachment.filename}
          </span>
          <span className="mt-0.5 block text-[10px] leading-4 text-slate-500 dark:text-slate-400">
            Image · {attachment.width}×{attachment.height}
          </span>
        </span>
      </div>
    )
  }
  const Icon = attachment.type === "mention_directory" ? Folder : FileText
  const rangeSuffix =
    attachment.type === "mention_file_range"
      ? `:${attachment.startLine}-${attachment.endLine}`
      : ""
  const label = attachment.path.startsWith("$M/attachments/")
    ? attachment.path.slice("$M/attachments/".length)
    : attachment.path
  return (
    <div
      className="flex h-13 min-w-44 max-w-64 items-center gap-2.5 rounded-lg border border-slate-300 bg-white px-2.5 dark:border-slate-750 dark:bg-slate-850"
      title={attachment.path}
    >
      <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-slate-150 text-slate-600 dark:bg-slate-750 dark:text-slate-300">
        <Icon size={17} />
      </span>
      <span className="min-w-0 font-sans">
        <span className="block truncate text-[12px] font-medium leading-4 text-slate-800 dark:text-slate-200">
          {label}{rangeSuffix}
        </span>
        <span className="mt-0.5 block text-[10px] leading-4 text-slate-500 dark:text-slate-400">
          {attachment.type === "mention_directory"
            ? "Folder"
            : attachment.type === "mention_file_range"
            ? `Lines ${attachment.startLine}–${attachment.endLine}`
            : "File"}
        </span>
      </span>
    </div>
  )
}

/** Left gutter wrapper for non-user messages */
export function Gutter({ children }: { children: ReactNode }): ReactNode {
  return <div className="[padding-left:12px]">{children}</div>
}
