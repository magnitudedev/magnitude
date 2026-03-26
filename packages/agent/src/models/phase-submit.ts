import { defineStateModel, type BaseState, type Phase } from '@magnitudedev/tools'
import { phaseSubmitTool, phaseSubmitXmlBinding } from '../tools/phase-submit'

export interface PhaseSubmitState extends BaseState {
  toolKey: 'phaseSubmit'
  output?: string
  errorMessage?: string
}

export const phaseSubmitModel = defineStateModel('phaseSubmit', {
  tool: phaseSubmitTool,
  binding: phaseSubmitXmlBinding,
})({
  initial: {
    output: undefined as string | undefined,
    errorMessage: undefined as string | undefined,
  },
  reduce: (state, event) => {
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
        return { ...state, phase: 'completed' as Phase, output: String(event.output), errorMessage: undefined }
      case 'error':
        return { ...state, phase: 'error' as Phase, errorMessage: event.error.message }
      case 'rejected':
        return { ...state, phase: 'rejected' as Phase }
      case 'interrupted':
        return { ...state, phase: 'interrupted' as Phase }
    }
  },
})