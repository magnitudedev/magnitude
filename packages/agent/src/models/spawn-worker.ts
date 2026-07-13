import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { spawnWorkerTool } from '../tools/task-tools'
import { SpawnWorkerStateSchema, type SpawnWorkerState } from './tool-state'

export { SpawnWorkerStateSchema, type SpawnWorkerState } from './tool-state'

const initial: Omit<SpawnWorkerState, 'phase' | 'errorMessage'> = {
  taskId: Option.none(),
  role: Option.none(),
  agentId: Option.none(),
  message: Option.none(),
  yield: Option.none(),
  title: Option.none(),
}

export const spawnWorkerModel = defineStateModel(spawnWorkerTool)({
  state: SpawnWorkerStateSchema,
  initial,
  reduce: (state, event): SpawnWorkerState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        if (event.field === 'taskId') return { ...state, phase: 'streaming', taskId: Option.some(Option.getOrElse(state.taskId, () => '') + event.delta) }
        if (event.field === 'agentId') return { ...state, phase: 'streaming', agentId: Option.some(Option.getOrElse(state.agentId, () => '') + event.delta) }
        if (event.field === 'message') return { ...state, phase: 'streaming', message: Option.some(Option.getOrElse(state.message, () => '') + event.delta) }
        if (event.field === 'role') return { ...state, phase: 'streaming', role: Option.some(Option.getOrElse(state.role, () => '') + event.delta) }
        return state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          taskId: typeof event.input.taskId === 'string' ? Option.some(event.input.taskId) : state.taskId,
          agentId: typeof event.input.agentId === 'string' ? Option.some(event.input.agentId) : state.agentId,
          message: typeof event.input.message === 'string' ? Option.some(event.input.message) : state.message,
          role: typeof event.input.role === 'string' ? Option.some(event.input.role) : state.role,
          yield: typeof event.input.yield === 'boolean' ? Option.some(event.input.yield) : state.yield,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return {
              ...state,
              phase: 'completed',
              taskId: Option.some(event.result.output.taskId),
              agentId: Option.some(event.result.output.agentId),
              title: Option.some(event.result.output.title),
              yield: typeof event.result.output.yield === 'boolean' ? Option.some(event.result.output.yield) : state.yield,
            }
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
