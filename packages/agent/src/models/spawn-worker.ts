import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { spawnWorkerTool, spawnWorkerXmlBinding } from '../tools/task-tools'

export interface SpawnWorkerState extends BaseState {
  toolKey: 'spawnWorker'
  id?: string
  role?: string
}

const initial: Omit<SpawnWorkerState, 'phase' | 'toolKey'> = {
  id: undefined,
  role: undefined,
}

export const spawnWorkerModel = defineStateModel('spawnWorker', {
  tool: spawnWorkerTool,
  binding: spawnWorkerXmlBinding,
})({
  initial,
  reduce: (state, event): SpawnWorkerState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return {
          ...state,
          phase: 'streaming',
          id: event.streaming.id?.value ?? state.id,
          role: event.streaming.role?.value ?? state.role,
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
        return { ...state, phase: 'completed', id: event.output.id, role: event.output.role }
      case 'error':
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
