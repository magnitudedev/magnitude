/**
 * User message — spec §9.3.1
 *
 * Compact right-aligned bubble. Attachments row. Metadata with copy + timestamp.
 */
import { useState, type ReactNode } from "react"
import type { UserMessage as UserMessageType, DisplayAttachment } from "@magnitudedev/sdk"
import { CopyButton, Timestamp, AttachmentPill } from "./shared"

export function UserMessage({ message }: { message: UserMessageType }): ReactNode {
  const [hovered, setHovered] = useState(false)
  const showMetadata = hovered || message.attachments.length > 0
  const borderColor = message.taskMode ? "var(--line-bash)" : "var(--line-user)"

  return (
    <div
      data-task-mode={message.taskMode}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
      }}
    >
      <div
        style={{
          maxWidth: "min(720px, 72%)",
          background: "var(--bg-surface-elevated)",
          border: `1px solid ${message.taskMode ? borderColor : "var(--border-default)"}`,
          borderRadius: 8,
          padding: "8px 11px",
          transition: "background 100ms ease",
        }}
      >
        <div
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: "14px",
            color: "var(--fg-primary)",
            lineHeight: 1.55,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {message.content}
        </div>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          width: "min(720px, 72%)",
          minHeight: 22,
          marginTop: "3px",
          padding: "0 2px",
          opacity: showMetadata ? 1 : 0,
          transition: "opacity 100ms ease",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "4px", flexWrap: "wrap", minWidth: 0 }}>
          {message.attachments.map((a: DisplayAttachment, i: number) => (
            <AttachmentPill key={i} attachment={a} />
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
          <CopyButton text={message.content} />
          <Timestamp ts={message.timestamp} />
        </div>
      </div>
    </div>
  )
}
