import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { cancelTaskTool, cancelTaskXmlBinding } from '../tools/task-tools'

export interface CancelTaskState extends BaseState {
  toolKey: 'cancelTask'
  taskId?: string
  cancelledCount?: number
  workersKilled?: number
}

const initial: Omit<CancelTaskState, 'phase' | 'toolKey'> = {
  taskId: undefined,
  cancelledCount: undefined,
  workersKilled: undefined,
}

export const cancelTaskModel = defineStateModel('cancelTask', {
  tool: cancelTaskTool,
  binding: cancelTaskXmlBinding,
})({
  initial,
  reduce: (state, event): CancelTaskState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return {
          ...state,
          phase: 'streaming',
          taskId: event.streaming.taskId?.value ?? state.taskId,
        }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
        return { ...state, phase: 'executing' }
      case 'parseError':
        return { ...state, phase: 'error' }
      case 'completed':
        return {
          ...state,
          phase: 'completed',
          taskId: event.output.taskId ?? state.taskId,
          cancelledCount: event.output.cancelledCount ?? state.cancelledCount,
          workersKilled: event.output.workersKilled ?? state.workersKilled,
        }
      case 'error':
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
