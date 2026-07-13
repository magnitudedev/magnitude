import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { killWorkerTool } from '../tools/task-tools'
import { KillWorkerStateSchema, type KillWorkerState } from './tool-state'

export { KillWorkerStateSchema, type KillWorkerState } from './tool-state'

const initial: Omit<KillWorkerState, 'phase' | 'errorMessage'> = {
  taskId: Option.none(),
}

export const killWorkerModel = defineStateModel(killWorkerTool)({
  state: KillWorkerStateSchema,
  initial,
  reduce: (state, event): KillWorkerState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        return event.field === 'taskId'
          ? { ...state, phase: 'streaming', taskId: Option.some(Option.getOrElse(state.taskId, () => '') + event.delta) }
          : state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          taskId: typeof event.input.taskId === 'string' ? Option.some(event.input.taskId) : state.taskId,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', taskId: Option.some(event.result.output.taskId) }
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
