import { defineStateModel, type BaseState, type Phase } from '@magnitudedev/tools'
import { phaseVerdictTool, phaseVerdictXmlBinding } from '../tools/phase-verdict'

export interface PhaseVerdictState extends BaseState {
  toolKey: 'phase-verdict'
}

export const phaseVerdictModel = defineStateModel('phase-verdict', {
  tool: phaseVerdictTool,
  binding: phaseVerdictXmlBinding,
})({
  initial: {},
  reduce: (state, event): PhaseVerdictState => {
    switch (event.type) {
      case 'started':
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming' as Phase }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
      case 'parseError':
        return { ...state, phase: 'executing' as Phase }
      case 'completed':
        return { ...state, phase: 'completed' as Phase }
      case 'error':
        return { ...state, phase: 'error' as Phase }
      case 'rejected':
        return { ...state, phase: 'rejected' as Phase }
      case 'interrupted':
        return { ...state, phase: 'interrupted' as Phase }
    }
  },
})