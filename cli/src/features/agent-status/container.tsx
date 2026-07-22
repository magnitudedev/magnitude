/**
 * AgentStatus feature container (spec §5.6) — "what is the agent doing".
 * Reads the root timeline and task rows from display state; renders the
 * working timer and task list. Fork expansion goes through the display
 * shape hook so the worker timeline is requested from the agent.
 */
import { useMemo, type ReactNode } from 'react'
import {
  useDisplayState,
  getFork,
  useSlotProfiles,
  useDisplayViewController,
  findSlotProfile,
} from '@magnitudedev/client-common'
import { PRIMARY_SLOT_ID, ROLE_TO_SLOT, SECONDARY_SLOT_ID } from '@magnitudedev/sdk'
import { Option } from 'effect'
import type { TaskDisplayRow, InterruptedMessage } from '@magnitudedev/sdk'
import { WorkingTimer } from './working-timer'
import { TaskList } from './task-list'

export function WorkingTimerContainer(): ReactNode {
  const timeline = useDisplayState((state) => getFork(state, null) ?? null)
  const rootActor = useDisplayState((state) => state.actors["root"] ?? null)
  const { profiles } = useSlotProfiles()

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

  if (!timeline && !rootActor) return null

  // Map advisor role to its slot (primary) for model display
  const advisorSlot = ROLE_TO_SLOT.advisor
  const advisorSlotId = advisorSlot === 'primary' ? PRIMARY_SLOT_ID : SECONDARY_SLOT_ID
  const advisorProfile = profiles
    ? Option.getOrNull(findSlotProfile(profiles, advisorSlotId))
    : null
  return (
    <WorkingTimer
      work={rootActor?.work ?? null}
      interruptedMessage={interrupted}
      advisorModelName={advisorProfile?.modelDisplayName ?? null}
    />
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
