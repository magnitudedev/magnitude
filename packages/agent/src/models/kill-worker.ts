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
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming', id: event.streaming.id?.value ?? state.id }
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
