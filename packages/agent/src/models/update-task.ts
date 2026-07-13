import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { updateTaskTool } from '../tools/task-tools'
import { UpdateTaskStateSchema, UpdateTaskStatusSchema, type UpdateTaskState, type UpdateTaskStatus } from './tool-state'

export { UpdateTaskStateSchema, UpdateTaskStatusSchema, type UpdateTaskState, type UpdateTaskStatus } from './tool-state'

const initial: Omit<UpdateTaskState, 'phase' | 'errorMessage'> = {
  taskId: Option.none(),
  status: Option.none(),
}

function isValidUpdateTaskStatus(value: string): value is UpdateTaskStatus {
  return value === 'pending' || value === 'completed' || value === 'cancelled'
}

export const updateTaskModel = defineStateModel(updateTaskTool)({
  state: UpdateTaskStateSchema,
  initial,
  reduce: (state, event): UpdateTaskState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        if (event.field === 'taskId') return { ...state, phase: 'streaming', taskId: Option.some(Option.getOrElse(state.taskId, () => '') + event.delta) }
        return state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          taskId: typeof event.input.taskId === 'string' ? Option.some(event.input.taskId) : state.taskId,
          status: typeof event.input.status === 'string' && isValidUpdateTaskStatus(event.input.status)
            ? Option.some(event.input.status)
            : state.status,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', taskId: Option.some(event.result.output.taskId), status: Option.some(event.result.output.status) }
          case 'Error':
            return { ...state, phase: 'error' }
          case 'Denied':
            return { ...state, phase: 'rejected', errorMessage: Option.some(String(event.result.denial)) }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
          default:
            return state
        }
      }
      case 'ToolInputRejected':
        return { ...state, phase: 'error', errorMessage: Option.some(event.issue.message) }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
