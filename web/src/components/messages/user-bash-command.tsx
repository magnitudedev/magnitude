import type { ReactNode } from "react"
import type { UserBashCommandMessage } from "@magnitudedev/sdk"
import { CopyButton, Timestamp } from "./shared"

export function UserBashCommand({ message }: { message: UserBashCommandMessage }): ReactNode {
  const output = [message.stdout, message.stderr].filter(Boolean).join("\n")
  const failed = message.exitCode !== 0

  return (
    <div>
      <div
        style={{
          border: "1px solid var(--border-default)",
          borderLeft: "3px solid var(--line-bash)",
          borderRadius: "0 4px 4px 0",
          padding: "8px 11px",
          fontFamily: "var(--font-mono)",
          fontSize: "13px",
        }}
      >
        <div style={{ color: "var(--fg-primary)", fontWeight: 600 }}>
          <span style={{ color: "var(--line-bash)" }}>$ </span>
          {message.command}
          <span style={{ color: failed ? "var(--accent-error)" : "var(--accent-success)", marginLeft: 8 }}>
            {failed ? `Exit ${message.exitCode}` : "✓"}
          </span>
        </div>
        {output && (
          <pre
            style={{
              color: failed ? "var(--accent-error)" : "var(--fg-secondary)",
              margin: "6px 0 0",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {output}
          </pre>
        )}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4, padding: "0 2px" }}>
        <CopyButton text={message.command} />
        <Timestamp ts={message.timestamp} />
      </div>
    </div>
  )
}
