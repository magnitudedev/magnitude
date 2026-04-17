import type { TaskWorkerSnapshot } from '@magnitudedev/agent'
import type { TaskAssigneeSlot, TaskDisplayRow, WorkerSlotDisplay } from './types'

export function formatWorkerLabel(args: { role: string | null; agentId: string }): string {
  const { role, agentId } = args
  return role ? `[${role}] ${agentId}` : agentId
}

export function deriveWorkerSlotDisplay(snapshot: TaskWorkerSnapshot): WorkerSlotDisplay | null {
  if (snapshot.assignee.kind === 'user') {
    return {
      kind: 'user',
      label: 'user',
      tone: 'warning',
    }
  }

  if (snapshot.workerState.status === 'spawning') {
    if (!snapshot.workerState.role) return null
    return {
      kind: 'worker',
      variant: 'spawning',
      label: `[${snapshot.workerState.role}]`,
      icon: '+',
      tone: 'active',
      interactiveForkId: null,
      timer: null,
      resumed: false,
      continuityKey: null,
      ghostEligible: false,
    }
  }

  if (snapshot.assignee.kind !== 'worker') return null

  const label = formatWorkerLabel({
    role: snapshot.assignee.role,
    agentId: snapshot.assignee.agentId,
  })

  switch (snapshot.workerState.status) {
    case 'working':
      return {
        kind: 'worker',
        variant: 'working',
        label,
        icon: '●',
        tone: 'active',
        interactiveForkId: snapshot.assignee.forkId,
        workerState: snapshot.workerState,
        resumed: snapshot.workerState.resumeCount > 0,
        continuityKey: snapshot.workerState.forkId,
        ghostEligible: true,
      }

    case 'idle':
      return {
        kind: 'worker',
        variant: 'idle',
        label,
        icon: '●',
        tone: 'muted',
        interactiveForkId: snapshot.assignee.forkId,
        workerState: snapshot.workerState,
        resumed: snapshot.workerState.resumeCount > 0,
        continuityKey: snapshot.workerState.forkId,
        ghostEligible: true,
      }

    case 'killing':
      return {
        kind: 'worker',
        variant: 'killing',
        label,
        icon: '✕',
        tone: 'danger',
        interactiveForkId: snapshot.assignee.forkId,
        timer: null,
        resumed: false,
        continuityKey: snapshot.workerState.forkId,
        ghostEligible: true,
      }

    case 'unassigned':
      return null
  }
}

export function deriveTaskDisplayRow(snapshot: TaskWorkerSnapshot): TaskDisplayRow {
  const assignee: TaskAssigneeSlot = deriveWorkerSlotDisplay(snapshot) ?? { kind: 'none' }

  return {
    rowId: `task:${snapshot.taskId}`,
    kind: 'task',
    taskId: snapshot.taskId,
    title: snapshot.title,
    status: snapshot.status,
    parentId: snapshot.parentId,
    depth: snapshot.depth,
    updatedAt: snapshot.updatedAt,
    assignee,
  }
}