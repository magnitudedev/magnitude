import type { TaskStatus, WorkerState } from '@magnitudedev/agent'

export type WorkerSlotTone = 'active' | 'muted' | 'danger' | 'warning' | 'ghost'

export type TaskAssigneeSlot =
  | {
      kind: 'none'
    }
  | {
      kind: 'worker'
      variant: 'spawning'
      label: `[${string}]`
      icon: '+'
      tone: 'active'
      interactiveForkId: null
      timer: null
      resumed: false
      continuityKey: null
      ghostEligible: false
    }
  | {
      kind: 'worker'
      variant: 'working'
      label: string
      icon: '●'
      tone: 'active'
      interactiveForkId: string
      workerState: WorkerState
      resumed: boolean
      continuityKey: string
      ghostEligible: true
    }
  | {
      kind: 'worker'
      variant: 'idle'
      label: string
      icon: '●'
      tone: 'muted'
      interactiveForkId: string
      workerState: WorkerState
      resumed: boolean
      continuityKey: string
      ghostEligible: true
    }
  | {
      kind: 'worker'
      variant: 'killing'
      label: string
      icon: '✕'
      tone: 'danger'
      interactiveForkId: string
      timer: null
      resumed: false
      continuityKey: string
      ghostEligible: true
    }
  | {
      kind: 'user'
      label: 'user'
      tone: 'warning'
    }
  | {
      kind: 'ghost'
      icon: '✕'
      label: string
      tone: 'ghost'
      expiresAt: number
    }

export type WorkerSlotDisplay = Extract<TaskAssigneeSlot, { kind: 'worker' } | { kind: 'user' }>
export type GhostSlotDisplay = Extract<TaskAssigneeSlot, { kind: 'ghost' }>

export type TaskDisplayRow = {
  rowId: string
  kind: 'task'
  taskId: string
  title: string
  taskType: string
  status: TaskStatus
  parentId: string | null
  depth: number
  updatedAt: number
  assignee: TaskAssigneeSlot
}

export type GhostOverlay = {
  taskId: string
  expiresAt: number
  label: string
}

export type VisibleWorkerContinuity =
  | {
      continuityKey: string
      taskId: string
      label: string
      lastKnownIndex: number
    }
  | null

export type TaskListTaskRow = TaskDisplayRow
export type TaskListItem = TaskDisplayRow