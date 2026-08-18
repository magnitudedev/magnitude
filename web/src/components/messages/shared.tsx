import { Button } from "@/components/ui/button"

/**
 * Shared utilities for message components — copy button, timestamp, attachment pill.
 */
import { useState, type ReactNode } from "react"
import {
  Copy,
  Check,
  Clock,
  FileText,
  Folder,
  Image as ImageIcon,
} from "lucide-react"
import type { DisplayAttachment } from "@magnitudedev/sdk"
import { formatShortTimestamp } from "@magnitudedev/client-common"

/** Copy button with icon-swap feedback */
export function CopyButton({
  text,
  label = "Copy",
}: {
  text: string
  label?: string
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
      {label}
    </Button>
  )
}

/** Timestamp display */
export function Timestamp({ ts }: { ts: number }): ReactNode {
  return (
    <span className="text-slate-500 font-sans text-[13px]">
      {formatShortTimestamp(ts)}
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
      <span className="inline-flex items-center [gap:4px] bg-white dark:bg-slate-850 border border-slate-300 dark:border-slate-750 rounded-[4px] [padding:2px_6px] text-[11px] text-slate-600 dark:text-slate-400">
        <ImageIcon size={14} />
        {attachment.filename}
        <span className="text-slate-500">
          {attachment.width}×{attachment.height}
        </span>
      </span>
    )
  }
  const Icon = attachment.type === "mention_directory" ? Folder : FileText
  const rangeSuffix =
    attachment.type === "mention_file_range"
      ? `:${attachment.startLine}-${attachment.endLine}`
      : ""
  return (
    <span className="inline-flex items-center [gap:4px] bg-white dark:bg-slate-850 border border-slate-300 dark:border-slate-750 rounded-[4px] [padding:2px_6px] text-[11px] text-slate-600 dark:text-slate-400">
      <Icon size={14} />
      {attachment.path}
      {rangeSuffix && <span className="text-slate-500">{rangeSuffix}</span>}
    </span>
  )
}

/** Left gutter wrapper for non-user messages */
export function Gutter({ children }: { children: ReactNode }): ReactNode {
  return <div className="[padding-left:12px]">{children}</div>
}
