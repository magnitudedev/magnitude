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
  useSlotProfiles,
  useDisplayViewController,
  findSlotProfile,
  type LocalModelLoadActivity,
} from '@magnitudedev/client-common'
import { PRIMARY_SLOT_ID, ROLE_TO_SLOT, SECONDARY_SLOT_ID } from '@magnitudedev/sdk'
import { Option } from 'effect'
import type { TaskDisplayRow, InterruptedMessage } from '@magnitudedev/sdk'
import { ActivityRail } from './activity-rail'
import { TaskList } from './task-list'
import { ContextUsageBar, contextUsageWidth } from './context-usage-bar'

export function AgentStatusRowContainer({
  modelLoadActivity,
  width,
}: {
  readonly modelLoadActivity: LocalModelLoadActivity | null
  readonly width: number
}): ReactNode {
  const timeline = useDisplayState((state) => getFork(state, null) ?? null)
  const rootActor = useDisplayState((state) => state.actors["root"] ?? null)
  const modelRequestActivity = useDisplayState(
    (state) => state.modelRequests.root ?? null,
  )
  const { profiles, rootProfile } = useSlotProfiles()

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

  // Map advisor role to its slot (primary) for model display
  const advisorSlot = ROLE_TO_SLOT.advisor
  const advisorSlotId = advisorSlot === 'primary' ? PRIMARY_SLOT_ID : SECONDARY_SLOT_ID
  const advisorProfile = profiles
    ? Option.getOrNull(findSlotProfile(profiles, advisorSlotId))
    : null
  const context = rootActor?.context ?? null
  const contextHardCap = rootProfile?.contextWindow ?? null
  const tokenUsage = context && context.tokenEstimate > 0 ? context.tokenEstimate : null
  const isCompacting = context?.isCompacting ?? false
  const reservedContextWidth = contextUsageWidth(tokenUsage, contextHardCap, isCompacting)
  const activityWidth = Math.max(0, width - reservedContextWidth - 3)

  return (
    <box style={{
      flexDirection: 'row',
      flexShrink: 0,
      alignItems: 'center',
      width: '100%',
      paddingRight: 2,
    }}>
      <box style={{ width: activityWidth, minWidth: 0, overflow: 'hidden' }}>
        <ActivityRail
          work={rootActor?.work ?? null}
          width={activityWidth}
          waitsForGenerationProgress={rootProfile?.providerId === "local"}
          modelLoadActivity={modelLoadActivity}
          modelRequestActivity={modelRequestActivity}
          interruptedMessage={interrupted}
          advisorModelName={advisorProfile?.modelDisplayName ?? null}
        />
      </box>
      <box style={{ flexGrow: 1, minWidth: 0 }} />
      <ContextUsageBar
        tokenUsage={tokenUsage}
        hardCap={contextHardCap}
        isCompacting={isCompacting}
      />
    </box>
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
