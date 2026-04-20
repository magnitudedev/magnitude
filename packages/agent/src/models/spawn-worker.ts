import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { spawnWorkerTool } from '../tools/task-tools'

export interface SpawnWorkerState extends BaseState {
  toolKey: 'spawnWorker'
  id?: string
  message?: string
}

const initial: Omit<SpawnWorkerState, 'phase' | 'toolKey'> = {
  id: undefined,
  message: undefined,
}

export const spawnWorkerModel = defineStateModel('spawnWorker', spawnWorkerTool)({
  initial,
  reduce: (state, event): SpawnWorkerState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
        return {
          ...state,
          phase: 'streaming',
          id: event.streaming.id?.value ?? state.id,
          message: event.streaming.message?.value ?? state.message,
        }
      case 'inputReady':
        return {
          ...state,
          phase: 'streaming',
          id: event.input.id ?? state.id,
          message: event.input.message ?? state.message,
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
        return { ...state, phase: 'completed', id: event.output.id }
      case 'error':
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
