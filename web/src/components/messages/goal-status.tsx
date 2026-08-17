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
export function GoalStatus({
  message,
}: {
  message: GoalStatusType
}): ReactNode {
  const objective = Option.getOrNull(message.objective)
  const evidence = Option.getOrNull(message.evidence)
  if (message.status === "started") {
    return (
      <div className="flex items-center [gap:6px] [padding:2px_0]">
        <Target
          size={14}
          className="text-green-700 dark:text-green-500 shrink-0"
        />
        <span className="font-sans text-[13px] text-green-700 dark:text-green-500">
          Goal started
        </span>
        {objective && (
          <span className="font-sans text-[13px] text-slate-600 dark:text-slate-400">
            · {objective}
          </span>
        )}
      </div>
    )
  }
  // finished
  return (
    <div className="flex items-center [gap:6px] [padding:2px_0]">
      <CheckCircle2
        size={14}
        className="text-green-700 dark:text-green-500 shrink-0"
      />
      <span className="font-sans text-[13px] text-green-700 dark:text-green-500">
        Goal finished
      </span>
      {evidence && (
        <span className="font-sans text-[13px] text-slate-600 dark:text-slate-400">
          · {evidence}
        </span>
      )}
    </div>
  )
}
