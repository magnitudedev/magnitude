/**
 * WorkingTimer / status bar — spec §9.4
 *
 * Uses useSyncExternalStore for the timer tick — a module-level store
 * that increments every second. Components read the current tick value
 * and React handles the subscription lifecycle.
 */
import { useSyncExternalStore } from "react"
import { Option } from "effect"
import { Circle, Loader2, Square } from "lucide-react"
import { formatElapsedMs } from "@magnitudedev/client-common"
import type { DisplayActorWork } from "@magnitudedev/sdk"
import { subscribeTick, getTickSnapshot, subscribeNoop } from "@magnitudedev/client-common"

export interface WorkingTimerProps {
  work: DisplayActorWork | null
  interruptedText?: string | null
}

export function WorkingTimer({ work, interruptedText }: WorkingTimerProps): React.ReactNode {
  const active = work?.phase === "working"
  const activity = work?.activity ?? null

  // Subscribe to tick store — only subscribes (starts interval) while active
  const tick = useSyncExternalStore(
    active ? subscribeTick : subscribeNoop,
    getTickSnapshot,
  )
  void tick // forces re-render every second while active

  const elapsed = active && work
    ? Math.max(0, Date.now() - (work.activeSince ?? Date.now()))
    : 0
  const completedDuration = !active && work && work.lastWorkMs > 0 ? work.lastWorkMs : null

  if (interruptedText) {
    return (
      <div className="working-timer" style={{ height: 22, padding: "0 12px", display: "flex", alignItems: "center", flexShrink: 0, background: "var(--bg-base)", gap: 6 }}>
        <Square size={10} fill="currentColor" style={{ color: "var(--accent-error)", flexShrink: 0 }} />
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--accent-error)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {interruptedText}
        </span>
      </div>
    )
  }

  if (!active && completedDuration !== null && completedDuration > 0) {
    return (
      <div className="working-timer" style={{ height: 22, padding: "0 12px", display: "flex", alignItems: "center", flexShrink: 0, background: "var(--bg-base)", gap: 6 }}>
        <Circle size={8} fill="currentColor" style={{ color: "var(--fg-tertiary)", flexShrink: 0 }} />
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--fg-secondary)" }}>
          Worked for {formatElapsedMs(completedDuration)}
        </span>
      </div>
    )
  }

  const hasActivity = activity !== null
  if (!active && !hasActivity) return null

  return (
    <div className="working-timer" style={{ height: 22, padding: "0 12px", display: "flex", alignItems: "center", flexShrink: 0, background: "var(--bg-base)", gap: 8, overflow: "hidden" }}>
      <Circle size={8} fill="currentColor" className="animate-pulse-dot" style={{ color: active ? "var(--accent-primary)" : "var(--fg-tertiary)", flexShrink: 0 }} />
      {active && <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--fg-secondary)", flexShrink: 0 }}>{formatElapsedMs(elapsed)}</span>}
      {active && work && work.activeChildCount > 0 && <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--fg-secondary)", flexShrink: 0 }}>· {work.activeChildCount} worker{work.activeChildCount > 1 ? "s" : ""} running</span>}
      {activity && (
        <span style={{ display: "flex", alignItems: "center", gap: 4, fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--fg-secondary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", ...(activity.kind === "thinking" ? { animation: "thinking-pulse 0.8s ease-in-out infinite alternate" } : activity.kind === "advisor" ? { opacity: 0.7 } : {}) }}>
          {activity.kind === "tool" && Option.getOrNull(activity.decorator) === "spinner" && <Loader2 size={14} className="animate-spin" style={{ color: "var(--fg-secondary)", flexShrink: 0 }} />}
          {activity.kind === "advisor" && <Circle size={8} fill="currentColor" style={{ color: "var(--fg-tertiary)", flexShrink: 0 }} />}
          <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{activity.message}{activity.kind === "advisor" && ""}</span>
        </span>
      )}
    </div>
  )
}
