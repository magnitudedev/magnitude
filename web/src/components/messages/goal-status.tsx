/**
 * Goal status message — spec §9.3.7
 *
 * Started: Target icon, "Goal started" + optional objective.
 * Finished: CheckCircle2 icon, "Goal finished" + optional evidence.
 */
import { type ReactNode } from "react"
import { Option } from "effect"
import { Target, CheckCircle2 } from "lucide-react"
import type { GoalStatusMessage as GoalStatusType } from "@magnitudedev/sdk"

export function GoalStatus({ message }: { message: GoalStatusType }): ReactNode {
  const objective = Option.getOrNull(message.objective)
  const evidence = Option.getOrNull(message.evidence)
  if (message.status === "started") {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: "6px", padding: "2px 0" }}>
        <Target size={14} style={{ color: "var(--accent-success)", flexShrink: 0 }} />
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "13px", color: "var(--accent-success)" }}>
          Goal started
        </span>
        {objective && (
          <span style={{ fontFamily: "var(--font-sans)", fontSize: "13px", color: "var(--fg-secondary)" }}>
            · {objective}
          </span>
        )}
      </div>
    )
  }
  // finished
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "6px", padding: "2px 0" }}>
      <CheckCircle2 size={14} style={{ color: "var(--accent-success)", flexShrink: 0 }} />
      <span style={{ fontFamily: "var(--font-sans)", fontSize: "13px", color: "var(--accent-success)" }}>
        Goal finished
      </span>
      {evidence && (
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "13px", color: "var(--fg-secondary)" }}>
          · {evidence}
        </span>
      )}
    </div>
  )
}
