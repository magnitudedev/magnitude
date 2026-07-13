/**
 * WorkStatusBarSkeleton — placeholder mirroring the real WorkStatusBar
 * dimensions to prevent layout shift during session loading.
 *
 * Same outer box (border, radius, bg, minHeight 34px). Left: a 9px circle
 * + a 120px shimmer bar. Right: a 60px shimmer bar.
 */
import type { ReactNode } from "react"

const SHIMMER_BG = "linear-gradient(90deg, var(--border-subtle) 25%, var(--bg-surface) 50%, var(--border-subtle) 75%)"

export function WorkStatusBarSkeleton(): ReactNode {
  return (
    <div
      style={{
        margin: 0,
        border: "1px solid var(--border-default)",
        borderRadius: 6,
        background: "var(--bg-surface)",
        minHeight: 34,
        padding: "0 10px",
        display: "flex",
        alignItems: "center",
        gap: 8,
        flexShrink: 0,
        overflow: "hidden",
      }}
    >
      {/* Left: status dot + status text bar */}
      <div
        style={{
          width: 9,
          height: 9,
          borderRadius: "50%",
          background: "var(--border-subtle)",
          flexShrink: 0,
        }}
      />
      <div
        className="animate-shimmer"
        style={{
          width: 120,
          height: 12,
          borderRadius: 3,
          background: SHIMMER_BG,
          backgroundSize: "200% 100%",
          flexShrink: 0,
        }}
      />
      {/* Right: task count bar */}
      <div
        className="animate-shimmer"
        style={{
          marginLeft: "auto",
          width: 60,
          height: 12,
          borderRadius: 3,
          background: SHIMMER_BG,
          backgroundSize: "200% 100%",
          flexShrink: 0,
        }}
      />
    </div>
  )
}
