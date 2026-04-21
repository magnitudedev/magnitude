import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { spawnWorkerTool } from '../tools/task-tools'

export interface SpawnWorkerState extends BaseState {
  toolKey: 'spawnWorker'
  id?: string
  message?: string
  title?: string
}

const initial: Omit<SpawnWorkerState, 'phase' | 'toolKey'> = {
  id: undefined,
  message: undefined,
  title: undefined,
}

export const spawnWorkerModel = defineStateModel('spawnWorker', spawnWorkerTool)({
  initial,
  reduce: (state, event): SpawnWorkerState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        if (event.field === 'id') return { ...state, phase: 'streaming', id: (state.id ?? '') + event.delta }
        if (event.field === 'message') return { ...state, phase: 'streaming', message: (state.message ?? '') + event.delta }
        return state
      case 'ToolInputReady':
        return {
          ...state,
          phase: 'streaming',
          id: event.input.id ?? state.id,
          message: event.input.message ?? state.message,
        }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', id: event.result.output.id, title: event.result.output.title }
          case 'Error':
            return { ...state, phase: 'error' }
          case 'Rejected':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
        }
      }
      case 'ToolParseError':
        return { ...state, phase: 'error' }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
