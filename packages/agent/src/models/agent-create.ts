import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { agentCreateTool, agentCreateXmlBinding } from '../tools/agent-tools'

export interface AgentCreateState extends BaseState {
  toolKey: 'agentCreate'
  agentId?: string
}

const initial: Omit<AgentCreateState, 'phase' | 'toolKey'> = {
  agentId: undefined,
}

export const agentCreateModel = defineStateModel('agentCreate', {
  tool: agentCreateTool,
  binding: agentCreateXmlBinding,
})({
  initial,
  reduce: (state, event): AgentCreateState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming', agentId: event.streaming.agentId?.value ?? state.agentId }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
      case 'parseError':
        return { ...state, phase: 'executing' }
      case 'completed':
        return { ...state, phase: 'completed', agentId: event.output.agentId }
      case 'error':
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
