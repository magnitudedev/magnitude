/**
 * SidebarEmptyState & SidebarLoadingState — spec §10
 *
 * Empty state: MessageSquare icon, "No sessions found", helpful subtitle.
 * Loading state: skeleton rows sized to match real session rows.
 */
import { MessageSquare } from "lucide-react"
import type { ReactNode } from "react"

// ── Empty state ──

export function SidebarEmptyState({ searchQuery }: { searchQuery?: string }): ReactNode {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "32px 16px",
        textAlign: "center",
      }}
    >
      <MessageSquare size={24} style={{ color: "var(--fg-tertiary)", marginBottom: "8px" }} />
      <div
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: 14,
          color: "var(--fg-secondary)",
        }}
      >
        No sessions found
      </div>
      <div
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: 12,
          color: "var(--fg-tertiary)",
          marginTop: "4px",
        }}
      >
        {searchQuery ? "Try a different search" : "Create a new session to get started"}
      </div>
    </div>
  )
}

// ── Loading state (skeleton rows) ──

const SKELETON_COUNT = 5

export function SidebarLoadingState(): ReactNode {
  return (
    <>
      {Array.from({ length: SKELETON_COUNT }, (_, i) => (
        <div
          key={i}
          className="session-item"
          aria-hidden="true"
          style={{
            cursor: "default",
            pointerEvents: "none",
          }}
        >
          <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 3 }}>
            <div
              style={{
                height: 22,
                display: "flex",
                alignItems: "center",
                gap: 8,
                minWidth: 0,
              }}
            >
              <div
                style={{
                  height: 14,
                  width: `${i % 3 === 0 ? 68 : i % 3 === 1 ? 54 : 78}%`,
                  borderRadius: 3,
                  background: "var(--bg-surface-elevated)",
                  opacity: 0.56,
                }}
              />
              <div
                style={{
                  height: 12,
                  width: 32,
                  borderRadius: 3,
                  background: "var(--bg-surface-elevated)",
                  opacity: 0.38,
                  flexShrink: 0,
                }}
              />
            </div>
            <div
              style={{
                height: 19,
                display: "flex",
                alignItems: "center",
                gap: 8,
                minWidth: 0,
              }}
            >
              <div
                style={{
                  height: 12,
                  width: `${i % 2 === 0 ? 56 : 66}%`,
                  borderRadius: 3,
                  background: "var(--bg-surface-elevated)",
                  opacity: 0.34,
                }}
              />
              <div
                style={{
                  height: 12,
                  width: 28,
                  borderRadius: 3,
                  background: "var(--bg-surface-elevated)",
                  opacity: 0.28,
                  flexShrink: 0,
                }}
              />
            </div>
          </div>
        </div>
      ))}
    </>
  )
}
