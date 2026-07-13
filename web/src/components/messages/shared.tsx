/**
 * Shared utilities for message components — copy button, timestamp, attachment pill.
 */
import { useState, type ReactNode } from "react"
import { Copy, Check, Clock, FileText, Folder, Image as ImageIcon } from "lucide-react"
import type { DisplayAttachment } from "@magnitudedev/sdk"
import { formatShortTimestamp } from "@magnitudedev/client-common"

/** Copy button with icon-swap feedback */
export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }): ReactNode {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      }}
      aria-label={label || "Copy to clipboard"}
      className="hover-fg-copy"
      data-copied={copied ? "true" : "false"}
      style={{
        background: "transparent",
        border: "none",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        gap: "4px",
        fontSize: "13px",
        fontFamily: "var(--font-sans)",
        padding: 0,
      }}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
      {label}
    </button>
  )
}

/** Timestamp display */
export function Timestamp({ ts }: { ts: number }): ReactNode {
  return (
    <span style={{ color: "var(--fg-tertiary)", fontFamily: "var(--font-sans)", fontSize: "13px" }}>
      {formatShortTimestamp(ts)}
    </span>
  )
}

/** Queued indicator with clock icon */
export function QueuedIndicator(): ReactNode {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: "4px", color: "var(--fg-tertiary)", fontFamily: "var(--font-sans)", fontSize: "13px" }}>
      <Clock size={14} />
      Queued
    </span>
  )
}

/** Attachment pill for user messages */
export function AttachmentPill({ attachment }: { attachment: DisplayAttachment }): ReactNode {
  if (attachment.type === "image") {
    return (
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "4px",
          background: "var(--bg-surface)",
          border: "1px solid var(--border-default)",
          borderRadius: "4px",
          padding: "2px 6px",
          fontSize: "11px",
          color: "var(--fg-secondary)",
        }}
      >
        <ImageIcon size={14} />
        {attachment.filename}
        <span style={{ color: "var(--fg-tertiary)" }}>{attachment.width}×{attachment.height}</span>
      </span>
    )
  }
  const Icon = attachment.type === "mention_directory" ? Folder : FileText
  const rangeSuffix = attachment.type === "mention_file_range"
    ? `:${attachment.startLine}-${attachment.endLine}`
    : ""
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        background: "var(--bg-surface)",
        border: "1px solid var(--border-default)",
        borderRadius: "4px",
        padding: "2px 6px",
        fontSize: "11px",
        color: "var(--fg-secondary)",
      }}
    >
      <Icon size={14} />
      {attachment.path}
      {rangeSuffix && <span style={{ color: "var(--fg-tertiary)" }}>{rangeSuffix}</span>}
    </span>
  )
}

/** Left gutter wrapper for non-user messages */
export function Gutter({ children }: { children: ReactNode }): ReactNode {
  return (
    <div style={{ paddingLeft: "12px" }}>
      {children}
    </div>
  )
}
