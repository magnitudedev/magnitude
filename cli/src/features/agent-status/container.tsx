/**
 * AgentStatus feature container (spec §5.6) — "what is the agent doing".
 * Reads the root timeline and task rows from display state; renders the
 * activity rail and task list. Fork expansion goes through the display
 * shape hook so the worker timeline is requested from the agent.
 */
import { useMemo, type ReactNode } from 'react'
import {
  useDisplayState,
  getFork,
  useDisplayViewController,
  useSlotProfiles,
  isDisplayRootStatusActive,
  type LocalModelLoadActivity,
} from '@magnitudedev/client-common'
import type { TaskDisplayRow, InterruptedMessage, SlotId } from '@magnitudedev/sdk'
import { ActivityRail } from './activity-rail'
import { ActivityRailSlot } from './activity-rail-slot'
import { TaskList } from './task-list'
import { useTheme } from '../../hooks/use-theme'

export function ActivityRailContainer({
  modelLoadActivity,
  onStopModel,
  width,
  agentActivityEnabled = true,
}: {
  readonly modelLoadActivity: LocalModelLoadActivity | null
  readonly onStopModel: (slotId: SlotId) => void
  readonly width: number
  readonly agentActivityEnabled?: boolean
}): ReactNode {
  const theme = useTheme()
  const timeline = useDisplayState((state) => getFork(state, null) ?? null)
  const rootActor = useDisplayState((state) => state.actors["root"] ?? null)

  const interrupted: InterruptedMessage | null = useMemo(() => {
    // Root interrupt from timeline statusSlot
    if (timeline) {
      const slot = timeline.presentation.statusSlot
      if (slot.kind === 'interrupted') {
        const message = timeline.messages.byId[slot.messageId]
        if (message?.type === 'interrupted') return message
      }
    }
    return null
  }, [timeline])

  const slotWidth = Math.max(0, width - 3)
  const activityWidth = Math.max(0, slotWidth - 3)

  const rootStatus = rootActor?.kind === 'root' ? rootActor.status : null
  const hasWork = agentActivityEnabled && rootStatus !== null
    && isDisplayRootStatusActive(rootStatus)
  const visibleInterrupted = agentActivityEnabled ? interrupted : null
  if (!hasWork && modelLoadActivity === null && visibleInterrupted === null) {
    return null
  }

  return (
    <ActivityRailSlot width={slotWidth} color={theme.accent}>
      <ActivityRail
        status={agentActivityEnabled ? rootStatus : null}
        width={activityWidth}
        modelLoadActivity={modelLoadActivity}
        onStopModel={onStopModel}
        interruptedMessage={visibleInterrupted}
      />
    </ActivityRailSlot>
  )
}

export function TaskListContainer(): ReactNode {
  // Selector returns the store's stable tasks ref; the row list is derived
  // in a memo. Building arrays inside a store selector makes the snapshot
  // unstable and loops useSyncExternalStore's commit check.
  const taskState = useDisplayState((state) => state.tasks)
  const actors = useDisplayState((state) => state.actors)
  const tasks = useMemo(
    (): readonly TaskDisplayRow[] =>
      taskState.order
        .map((id) => taskState.byId[id])
        .filter((row): row is TaskDisplayRow => row !== undefined),
    [taskState],
  )
  const { profiles } = useSlotProfiles()
  const { pushFork } = useDisplayViewController()

  if (tasks.length === 0) return null

  return (
    <TaskList
      tasks={tasks}
      actors={actors}
      taskSummary={taskState.summary}
      pushForkOverlay={pushFork}
      slotProfiles={profiles}
    />
  )
}
