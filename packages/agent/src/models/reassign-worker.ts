import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { reassignWorkerTool } from '../tools/task-tools'
import { ReassignWorkerStateSchema, type ReassignWorkerState } from './tool-state'

export { ReassignWorkerStateSchema, type ReassignWorkerState } from './tool-state'

const initial: Omit<ReassignWorkerState, 'phase' | 'errorMessage'> = {
  agentId: Option.none(),
  taskId: Option.none(),
}

export const reassignWorkerModel = defineStateModel(reassignWorkerTool)({
  state: ReassignWorkerStateSchema,
  initial,
  reduce: (state, event): ReassignWorkerState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        if (event.field === 'agentId') return { ...state, phase: 'streaming', agentId: Option.some(Option.getOrElse(state.agentId, () => '') + event.delta) }
        if (event.field === 'taskId') return { ...state, phase: 'streaming', taskId: Option.some(Option.getOrElse(state.taskId, () => '') + event.delta) }
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          agentId: typeof event.input.agentId === 'string' ? Option.some(event.input.agentId) : state.agentId,
          taskId: typeof event.input.taskId === 'string' ? Option.some(event.input.taskId) : state.taskId,
        }
      case 'ToolExecutionEnded':
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', agentId: Option.some(event.result.output.agentId), taskId: Option.some(event.result.output.taskId) }
          case 'Error':
            return { ...state, phase: 'error' }
          case 'Denied':
            return { ...state, phase: 'rejected', errorMessage: Option.some(String(event.result.denial)) }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
          default:
            return state
        }
      case 'ToolInputRejected':
        return { ...state, phase: 'error', errorMessage: Option.some(event.issue.message) }
      default:
        return state
    }
  },
})
