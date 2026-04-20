import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { updateTaskTool } from '../tools/task-tools'

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

export const updateTaskModel = defineStateModel('updateTask', updateTaskTool)({
  initial,
  reduce: (state, event): UpdateTaskState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        if (event.field === 'id') return { ...state, phase: 'streaming', id: (state.id ?? '') + event.delta }
        return state
      case 'ToolInputReady':
        return {
          ...state,
          phase: 'streaming',
          id: event.input.id,
          status: isValidUpdateTaskStatus(event.input.status) ? event.input.status : state.status,
        }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', id: event.result.output.id, status: event.result.output.status }
          case 'Error':
            return { ...state, phase: 'error' }
          case 'Rejected':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
        }
      }
      case 'ToolInputParseError':
        return { ...state, phase: 'error' }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
