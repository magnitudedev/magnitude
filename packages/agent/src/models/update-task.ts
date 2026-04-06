import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { updateTaskTool, updateTaskXmlBinding } from '../tools/task-tools'

export interface UpdateTaskState extends BaseState {
  toolKey: 'updateTask'
  id?: string
  status?: 'pending' | 'completed' | 'cancelled'
}

const initial: Omit<UpdateTaskState, 'phase' | 'toolKey'> = {
  id: undefined,
  status: undefined,
}

function isValidUpdateTaskStatus(value: string | undefined): value is NonNullable<UpdateTaskState['status']> {
  return value === 'pending' || value === 'completed' || value === 'cancelled'
}

export const updateTaskModel = defineStateModel('updateTask', {
  tool: updateTaskTool,
  binding: updateTaskXmlBinding,
})({
  initial,
  reduce: (state, event): UpdateTaskState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return {
          ...state,
          phase: 'streaming',
          id: event.streaming.id?.value ?? state.id,
          status: (() => {
            const streamedStatus = event.streaming.status?.value
            return streamedStatus && isValidUpdateTaskStatus(streamedStatus)
              ? streamedStatus
              : state.status
          })(),
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
        return { ...state, phase: 'completed', id: event.output.id, status: event.output.status }
      case 'error':
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
