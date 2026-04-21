import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { agentCreateTool } from '../tools/agent-tools'

export interface AgentCreateState extends BaseState {
  toolKey: 'agentCreate'
  agentId?: string
}

const initial: Omit<AgentCreateState, 'phase' | 'toolKey'> = {
  agentId: undefined,
}

export const agentCreateModel = defineStateModel('agentCreate', agentCreateTool)({
  initial,
  reduce: (state, event): AgentCreateState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        return event.field === 'agentId'
          ? { ...state, phase: 'streaming', agentId: (state.agentId ?? '') + event.delta }
          : state
      case 'ToolInputReady':
        return { ...state, phase: 'streaming', agentId: event.input.agentId }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', agentId: event.result.output.agentId }
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
