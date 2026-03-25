import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { agentKillTool, agentKillXmlBinding } from '../tools/agent-tools'

export interface AgentKillState extends BaseState {
  toolKey: 'agentKill'
  agentId?: string
}

const initial: Omit<AgentKillState, 'phase' | 'toolKey'> = {
  agentId: undefined,
}

export const agentKillModel = defineStateModel('agentKill', {
  tool: agentKillTool,
  binding: agentKillXmlBinding,
})({
  initial,
  reduce: (state, event): AgentKillState => {
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
