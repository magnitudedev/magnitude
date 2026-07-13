/**
 * Error message — spec §9.3.10
 *
 * Box with red-tinted background, left border accent, [Error] tag,
 * message body, optional CTA (URL or Action button).
 */
import { type ReactNode } from "react"
import { Option } from "effect"
import type { ErrorDisplayMessage as ErrorType } from "@magnitudedev/sdk"
import { CopyButton, Timestamp } from "./shared"

type ErrorCtaValue = Option.Option.Value<ErrorType["cta"]>

function ErrorCta({ cta }: { cta: ErrorCtaValue }): ReactNode {
  if (cta.kind === "url") {
    return (
      <div style={{ marginTop: "6px", display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "13px", color: "var(--fg-secondary)" }}>
          {cta.label}:
        </span>
        <a
          href={cta.url}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "var(--accent-primary)", textDecoration: "underline", fontFamily: "var(--font-mono)", fontSize: "13px" }}
        >
          {cta.url}
        </a>
        <CopyButton text={cta.url} label="" />
      </div>
    )
  }
  // action
  return (
    <div style={{ marginTop: "6px" }}>
      <button
        className="hover-danger-button"
        style={{
          border: "1px solid var(--accent-error)",
          borderRadius: "4px",
          background: "transparent",
          color: "var(--accent-error)",
          fontFamily: "var(--font-mono)",
          fontSize: "13px",
          padding: "4px 10px",
          cursor: "pointer",
        }}
      >
        {cta.label} ({cta.chord})
      </button>
    </div>
  )
}

export function ErrorMessage({ message }: { message: ErrorType }): ReactNode {
  const cta = Option.getOrNull(message.cta)
  return (
    <div>
      <div
        style={{
          background: "var(--tint-error)",
          border: "1px solid var(--accent-error)",
          borderLeft: "3px solid var(--accent-error)",
          borderRadius: "0 4px 4px 0",
          padding: "10px 12px",
        }}
      >
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "13px", color: "var(--accent-error)", fontWeight: 600 }}>
          [Error]
        </div>
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "14px",
            color: "var(--fg-primary)",
            whiteSpace: "pre-wrap",
            marginTop: "4px",
            lineHeight: 1.5,
          }}
        >
          {message.message}
        </div>
        {cta && <ErrorCta cta={cta} />}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginTop: "4px",
          padding: "0 2px",
        }}
      >
        <CopyButton text={message.message} />
        <Timestamp ts={message.timestamp} />
      </div>
    </div>
  )
}
