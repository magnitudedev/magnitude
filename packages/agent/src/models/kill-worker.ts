import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { killWorkerTool } from '../tools/task-tools'

export interface KillWorkerState extends BaseState {
  toolKey: 'killWorker'
  id?: string
}

const initial: Omit<KillWorkerState, 'phase' | 'toolKey'> = {
  id: undefined,
}

export const killWorkerModel = defineStateModel('killWorker', killWorkerTool)({
  initial,
  reduce: (state, event): KillWorkerState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        return event.field === 'id'
          ? { ...state, phase: 'streaming', id: (state.id ?? '') + event.delta }
          : state
      case 'ToolInputReady':
        return { ...state, phase: 'streaming', id: event.input.id }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', id: event.result.output.id }
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
